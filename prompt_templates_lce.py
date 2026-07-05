import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def run_lcel_tutorial():
    print("--- Running Tutorial 2: LCEL & Prompt Templates ---")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    # 1. Define a dynamic prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert chef. Provide a recipe based only on the ingredients listed."),
        ("user", "What can I make with: {ingredients}?")
    ])

    # 2. Add an output parser to extract the clean text string directly
    output_parser = StrOutputParser()

    # 3. Assemble the pipeline using LangChain Expression Language (LCEL)
    recipe_chain = prompt_template | llm | output_parser

    # 4. Invoke the declarative chain
    print("Generating recipe...")
    result = recipe_chain.invoke({"ingredients": "eggs, tomatoes, avocado"})

    print("\n[Gemini Response]:")
    print(result)


if __name__ == "__main__":
    run_lcel_tutorial()
