import os
from typing import TypedDict, Annotated
from operator import add

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 1. Initialization and Env Setup
load_dotenv(override=True)
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0
)



class MemoryState(TypedDict):
    messages: Annotated[list[BaseMessage], add]
    user_name: str


def chat_node(state: MemoryState) -> dict:
    messages = state["messages"]
    user_name = state.get("user_name", "User")

    # Inject contextual knowledge dynamically from the state into the system instructions
    system_prompt = SystemMessage(
        content=f"You are a helpful assistant. You are talking to {user_name}."
    )
    full_history = [system_prompt] + messages

    response = llm.invoke(full_history)
    return {"messages": [response]}


# Instantiate in-memory checkpointer to persist context across unique thread IDs
memory_checkpointer = MemorySaver()

workflow = StateGraph(MemoryState)
workflow.add_node("chat_node", chat_node)
workflow.add_edge(START, "chat_node")
workflow.add_edge("chat_node", END)

# Compile the graph enriched with persistent checkpointer memory
app = workflow.compile(checkpointer=memory_checkpointer)

if __name__ == "__main__":
    print("--- Running Stateful Thread Memory Agent ---")

    # Thread 1: Chat interaction for Alex
    config_1 = {"configurable": {"thread_id": "session_alex"}}
    initial_input = {
        "messages": [HumanMessage(content="Hi, my name is Alex.")],
        "user_name": "Alex",
    }

    print("\n--- Turn 1 (Thread Alex) ---")
    events = app.stream(initial_input, config=config_1)
    for event in events:
        for node, state in event.items():
            print(state["messages"][-1].content)

    # Thread 1: Secondary contextual query
    follow_up_input = {
        "messages": [HumanMessage(content="What is my name again?")]
    }
    print("\n--- Turn 2 (Thread Alex - Context Retention) ---")
    events = app.stream(follow_up_input, config=config_1)
    for event in events:
        for node, state in event.items():
            print(state["messages"][-1].content)

    # Thread 2: Separated isolated user environment
    config_2 = {"configurable": {"thread_id": "session_bob"}}
    bob_input = {
        "messages": [HumanMessage(content="What is my name?")],
        "user_name": "Bob",
    }
    print("\n--- Turn 3 (Thread Bob - Isolated Context) ---")
    events = app.stream(bob_input, config=config_2)
    for event in events:
        for node, state in event.items():
            print(state["messages"][-1].content)
