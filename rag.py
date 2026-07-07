import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()


def create_mock_policy_file():
    with open("company_policy.txt", "w") as f:
        f.write(
            "Company Policy Update: Employees are eligible for 25 days of paid annual leave.\n"
            "Remote work is approved up to 3 days per week with manager coordination."
        )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_rag_tutorial():
    print("--- Running Tutorial 4: Native LCEL RAG ---")
    create_mock_policy_file()

    # 1. Initialize Gemini Model and the Fixed Embedding
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    # FIX: Native production string matching Google AI Studio channel specifications
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Extract context and load vector store
    with open("company_policy.txt", "r") as f:
        text_content = f.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    docs = text_splitter.create_documents([text_content])

    # Chroma indexing will now pass execution safely
    vector_store = Chroma.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    # 3. Create your custom Q&A prompt template
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's question using ONLY the provided context:\n\n{context}",
            ),
            ("user", "{input}"),
        ]
    )

    # 4. Build a clean, readable LCEL Chain
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # 5. Invoke the chain directly
    query = "How many days of paid annual leave do I get?"
    print(f"Querying: '{query}'")
    answer = rag_chain.invoke(query)

    print(f"\n[Gemini Answer]:\n{answer}")

    # if os.path.exists("company_policy.txt"):
    #   os.remove("company_policy.txt")


if __name__ == "__main__":
    run_rag_tutorial()
