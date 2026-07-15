import os
from typing import List, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------------
# 1. Set up the LLM (Claude Opus 4.1) and embedding model (Voyage AI)
# ---------------------------------------------------------------------------
llm = ChatAnthropic(
    model="claude-opus-4-8",  # swap for your exact model id
    temperature=0.3,
    max_tokens=1024,
)

embeddings = VoyageAIEmbeddings(
    model="voyage-3-large",  # Anthropic-recommended embeddings
)

# ---------------------------------------------------------------------------
# 2. Load / prepare documents and build the vector store
# ---------------------------------------------------------------------------
raw_texts = [
    "LangGraph is a library for building stateful, multi-actor applications "
    "with LLMs, built on top of LangChain.",
    "Claude Opus 4 is Anthropic's most capable model, ideal for complex "
    "reasoning, coding, and agentic tasks.",
    "Voyage AI provides high-quality embedding models recommended by "
    "Anthropic for retrieval-augmented generation.",
    "RAG (Retrieval-Augmented Generation) combines a retriever that fetches "
    "relevant documents with a generator LLM that produces answers.",
]

docs = [Document(page_content=t) for t in raw_texts]

# Split into chunks (useful for longer documents)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Create a Chroma vector store from the chunks
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="rag_demo",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ---------------------------------------------------------------------------
# 3. Define the LangGraph state
# ---------------------------------------------------------------------------
class RAGState(TypedDict):
    question: str
    documents: List[Document]
    answer: str


# ---------------------------------------------------------------------------
# 4. Define the graph nodes
# ---------------------------------------------------------------------------
def retrieve(state: RAGState) -> RAGState:
    """Retrieve relevant documents for the question."""
    docs = retriever.invoke(state["question"])
    return {"documents": docs}


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the question using ONLY the "
            "provided context. If the answer is not in the context, say you "
            "don't know.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def generate(state: RAGState) -> RAGState:
    """Generate an answer using Claude Opus 4 from the retrieved context."""
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    messages = prompt.invoke(
        {"context": context, "question": state["question"]}
    )
    response = llm.invoke(messages)
    return {"answer": response.content}


# ---------------------------------------------------------------------------
# 5. Build and compile the graph
# ---------------------------------------------------------------------------
graph_builder = StateGraph(RAGState)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

# ---------------------------------------------------------------------------
# 6. Run it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = graph.invoke({"question": "What embedding model does Anthropic recommend for RAG?"})
    print("Answer:\n", result["answer"])
    print("\nRetrieved sources:")
    for i, doc in enumerate(result["documents"], 1):
        print(f"{i}. {doc.page_content[:80]}...")