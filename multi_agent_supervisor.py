import os
from typing import TypedDict, Annotated, Literal
from operator import add

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

# 1. Initialization and Env Setup
load_dotenv(override=True)
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0
)


# 1. Define Supervisor Routing Schema
class RouterSchema(BaseModel):
    """Decides the next worker node to execute or exits if the goal is met."""

    next_step: Literal["code_agent", "qa_agent", "FINISH"] = Field(
        description="The next specialized agent node to call, or FINISH if fully complete."
    )


# 2. Define Core State
class TeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add]
    next_step: str
    code_artifact: str


# 3. Define Specialized Node Agents
def supervisor_node(state: TeamState) -> dict:
    """Orchestrates system objectives and assigns micro-tasks."""
    messages = state["messages"]

    system_instruction = (
        "You are the System Supervisor. Manage two workers: 'code_agent' (writes raw python code) "
        "and 'qa_agent' (reviews, checks bugs, and suggests fixes). Review the history and assign "
        "the next node, or select FINISH if the code artifact is generated, verified, and complete."
    )

    structured_llm = llm.with_structured_output(RouterSchema)
    response = structured_llm.invoke([AIMessage(content=system_instruction)] + messages)

    return {"next_step": response.next_step}


def code_agent_node(state: TeamState) -> dict:
    """Focuses completely on generating functional python structures."""
    messages = state["messages"]

    prompt = (
        "You are a coding agent. Generate clean, minimal Python code based on requests. "
        "Output ONLY raw Python code inside a markdown block. Do not write general explanations."
    )

    response = llm.invoke([AIMessage(content=prompt)] + messages)
    return {
        "messages": [AIMessage(content=response.content, name="code_agent")],
        "code_artifact": response.content,
    }


def qa_agent_node(state: TeamState) -> dict:
    """Focuses entirely on evaluating bugs, optimizations, and syntax formatting."""
    code_to_test = state.get("code_artifact", "No code found.")

    prompt = (
        f"You are a QA Engineer. Review the following Python code for any syntax errors or logic bugs:\n\n{code_to_test}\n\n"
        f"If the code looks perfect, respond explicitly with: 'PASSED QA'. Otherwise, list required bug fixes."
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content, name="qa_agent")]}


# 4. Define Supervisor Conditional Link
def route_next(state: TeamState) -> Literal["code_agent", "qa_agent", END]:
    """Inspects the supervisor state value to route execution."""
    target = state["next_step"]
    if target == "FINISH":
        return END
    return target


# 5. Connect the Orchestration Topology
workflow = StateGraph(TeamState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("code_agent", code_agent_node)
workflow.add_node("qa_agent", qa_agent_node)

# Connect worker leaf nodes straight back to supervisor for reassessment
workflow.add_edge("code_agent", "supervisor")
workflow.add_edge("qa_agent", "supervisor")

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges("supervisor", route_next)

app = workflow.compile()

if __name__ == "__main__":
    print("--- Running Multi-Agent Supervisor Workgroup ---")
    task = "Write a fast python function to calculate the fibonacci sequence using dynamic programming."
    inputs = {"messages": [HumanMessage(content=task)]}

    for chunk in app.stream(inputs, {"recursion_limit": 15}):
        for node, state in chunk.items():
            print(f"\n[Active Node System: {node}]")
            if "next_step" in state:
                print(f"Supervisor choice -> Next Up: {state['next_step']}")
            if "messages" in state and state["messages"]:
                print(f"Message Segment:\n{state['messages'][-1].content}")
