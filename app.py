import os
import shutil
import streamlit as st
import chromadb  # Direct import to configure clients explicitly
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load API key configurations
load_dotenv()

# Web Application UI Layout Configurations
st.set_page_config(page_title="Persistent Gemini RAG", page_icon="💾", layout="wide")
st.title("💾 On-Premise Persistent Gemini RAG Explorer")
st.subheader("Keep your indexed documents safely stored on-premise across application restarts.")

DB_DIR = "./chroma_db"
COLLECTION_NAME = "gemini_knowledge_base"


# --- RAG Core Functions ---

def get_embeddings_model():
    """Initializes the active, production-ready Gemini embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="retrieval_document"
    )


def format_docs(docs):
    """Joins chunk elements natively with spacing."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_vectorstore_instance():
    """Builds a secure, direct connection tracking the custom embedding dimensions explicitly."""
    embeddings = get_embeddings_model()

    # FIX: Initialize a native PersistentClient to block 384-fallback defaults
    persistent_client = chromadb.PersistentClient(path=DB_DIR)

    # Establish the collection wrapper tracking our Gemini function directly
    vector_store = Chroma(
        client=persistent_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )
    return vector_store


def process_text_into_persistent_store(raw_text: str):
    """Chunks text content and appends it cleanly to the explicit database instance."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([raw_text])

    vector_store = get_vectorstore_instance()
    # Safely index documents into the tracked collection layout
    vector_store.add_documents(docs)
    return vector_store.as_retriever(search_kwargs={"k": 2})


# --- Startup Verification Logic ---

# Check disk layer space on startup to safely pre-load data
if "retriever" not in st.session_state:
    if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
        try:
            vector_store = get_vectorstore_instance()
            # Verify if documents exist in the persistent collection layer
            if vector_store._collection.count() > 0:
                st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 2})
                st.toast("🎉 Detected existing database on disk! Pre-loaded successfully.", icon="💾")
            else:
                st.session_state.retriever = None
        except Exception as startup_err:
            st.error(f"Failed to auto-load index. Wiping corrupted metadata may be required: {startup_err}")
            st.session_state.retriever = None
    else:
        st.session_state.retriever = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Graphical Interface Layout ---

# Sidebar for Knowledge Source Uploads & Database Maintenance
with st.sidebar:
    st.header("🗂️ On-Premise Data Manager")

    # Real-time state indicators
    if st.session_state.retriever is not None:
        st.success("🟢 System Status: Ready (Database Loaded)")
    else:
        st.warning("🔴 System Status: No Knowledge Base Initialized")

    uploaded_file = st.file_uploader(
        "Append text materials to local disk storage:",
        type=["txt", "md", "log"],
        help="Upload new reference files to store inside your on-premise vector database."
    )

    if uploaded_file is not None:
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            with st.spinner("Chunking text layout and updating disk index..."):
                raw_context = uploaded_file.read().decode("utf-8")
                st.session_state.retriever = process_text_into_persistent_store(raw_context)
                st.session_state.last_uploaded = uploaded_file.name
                st.rerun()

    st.markdown("---")
    st.subheader("⚙️ Maintenance Panel")

    # Safe database wipe reset implementation
    if st.button("🗑️ Reset Database (Wipe Local Storage)", type="primary"):
        if os.path.exists(DB_DIR):
            # Close active SQLite handles implicitly before deleting files
            st.session_state.retriever = None
            import gc

            gc.collect()
            shutil.rmtree(DB_DIR)

        st.session_state.retriever = None
        st.session_state.messages = []
        if "last_uploaded" in st.session_state:
            del st.session_state.last_uploaded
        st.toast("Database folder erased completely from disk.", icon="💥")
        st.rerun()

# Primary Interactive Conversation Space
if st.session_state.retriever is None:
    st.info(
        "ℹ️ **Database Empty:** Please drag and drop a reference text file into the sidebar to establish your local database on-premise.")
else:
    # Render historical conversation components
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle incoming user query lines
    if user_query := st.chat_input("Ask a question anchored to your local disk databases..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Execute generative completions against retriever pipeline layers
        with st.chat_message("assistant"):
            with st.spinner("Searching on-premise data chunks..."):
                try:
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

                    rag_prompt = ChatPromptTemplate.from_messages([
                        ("system", "Answer the user's question using ONLY the provided context snippets. "
                                   "If the answer cannot be confidently formulated from the context, "
                                   "respond with 'I cannot find that information in the provided source materials.'\n\n"
                                   "Context:\n{context}"),
                        ("user", "{input}")
                    ])

                    rag_chain = (
                            {"context": st.session_state.retriever | format_docs, "input": RunnablePassthrough()}
                            | rag_prompt
                            | llm
                            | StrOutputParser()
                    )

                    response_output = rag_chain.invoke(user_query)
                    st.markdown(response_output)
                    st.session_state.messages.append({"role": "assistant", "content": response_output})

                except Exception as error_exception:
                    st.error(f"Execution Error: {str(error_exception)}")
