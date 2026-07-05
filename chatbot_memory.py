import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# FIX: Core native module replaces the legacy community setup
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()


def run_memory_tutorial():
    print("--- Running Tutorial 3: Chatbot with Memory ---")

    # Initialize the modern production model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly AI companion."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}")
    ])

    chain = chat_prompt | llm
    session_store = {}

    # Retrieves or instantiates isolated memory lists per session ID
    def get_session_history(session_id: str):
        if session_id not in session_store:
            session_store[session_id] = InMemoryChatMessageHistory()
        return session_store[session_id]

    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    user_config = {"configurable": {"session_id": "user_session_789"}}

    # Execution Turn 1
    print("User: Hi, my name is Alex.")
    with_history.invoke({"input": "Hi, my name is Alex."}, config=user_config)

    # Execution Turn 2 (Gemini maps context from Turn 1 natively)
    print("User: What is my name?")
    response = with_history.invoke({"input": "What is my name?"}, config=user_config)

    print(f"\n[Gemini Response]:\n{response.content}")


if __name__ == "__main__":
    run_memory_tutorial()
