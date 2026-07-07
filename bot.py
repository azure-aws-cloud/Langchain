import os
import shutil
import gc
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

app = FastAPI(title="Gemini RAG REST API", version="1.0")

# Enable CORS for Angular frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Default Angular URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_DIR = "./chroma_db"
COLLECTION_NAME = "gemini_knowledge_base"


# --- Core RAG Dependencies ---


def get_embeddings_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", task_type="retrieval_document"
    )


def get_vectorstore_instance():
    embeddings = get_embeddings_model()
    persistent_client = chromadb.PersistentClient(path=DB_DIR)
    return Chroma(
        client=persistent_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class QueryRequest(BaseModel):
    question: str


# --- API Endpoints ---


@app.get("/api/status")
async def get_status():
    """Checks if the local vector database contains indexed data."""
    if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
        try:
            vector_store = get_vectorstore_instance()
            count = vector_store._collection.count()
            if count > 0:
                return {
                    "initialized": True,
                    "document_count": count,
                    "message": "Database loaded successfully.",
                }
        except Exception as e:
            return {
                "initialized": False,
                "document_count": 0,
                "message": f"Error checking DB: {str(e)}",
            }
    return {"initialized": False, "document_count": 0, "message": "Database is empty."}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Receives files, chunks the content, and appends it to ChromaDB."""
    try:
        content = await file.read()
        raw_text = content.decode("utf-8")

        # Split documents cleanly
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.create_documents([raw_text])

        vector_store = get_vectorstore_instance()
        vector_store.add_documents(docs)

        return {
            "status": "success",
            "message": f"Indexed {len(docs)} chunks from {file.filename}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


@app.post("/api/chat")
async def chat_query(request: QueryRequest):
    """Executes generative QA over the local indexed store context."""
    try:
        vector_store = get_vectorstore_instance()
        if vector_store._collection.count() == 0:
            raise HTTPException(
                status_code=400,
                detail="Database is empty. Please upload documents first.",
            )

        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the user's question using ONLY the provided context snippets. "
                    "If the answer cannot be confidently formulated from the context, "
                    "respond with 'I cannot find that information in the provided source materials.'\n\n"
                    "Context:\n{context}",
                ),
                ("user", "{input}"),
            ]
        )

        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        response = rag_chain.invoke(request.question)
        print("the response from llm is ", response)
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG execution failure: {str(e)}")


@app.delete("/api/reset")
async def reset_database():
    """Wipes the local disk storage database completely."""
    try:
        if os.path.exists(DB_DIR):
            gc.collect()  # Release SQLite handles before deletion
            shutil.rmtree(DB_DIR)
        return {"status": "success", "message": "Database erased completely from disk."}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to reset database: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
