import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables from .env file
load_dotenv()


def run_foundations_tutorial():
    print("--- Running Tutorial 1: Foundations ---")

    # Initialize Gemini model (requires GOOGLE_API_KEY in environment)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    # Structure system and user instructions
    messages = [
        SystemMessage(content="You are a helpful data science assistant."),
        HumanMessage(content="Explain overfitting in one sentence."),
    ]

    # Request completion
    print("Sending request to Gemini...")
    response = llm.invoke(messages)

    print("\n[Gemini Response]:")
    print(response.content)


if __name__ == "__main__":
    run_foundations_tutorial()
