import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

# Load API key configuration
load_dotenv()

# App Configuration
st.set_page_config(page_title="Gemini RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 Gemini 2.5 Multi-Document RAG Explorer")
st.subheader(
    "Upload raw reference materials and query them seamlessly using native LangChain LCEL."
)


# --- Initialization and RAG Pipeline Functions ---


def process_text_into_vectorstore(raw_text: str):
    """Chunks text content and indexes it into an in-memory Chroma instance."""
    # Chunking document layout
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([raw_text])

    # Initialize standard production Gemini embedding strings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create local Chroma instance
    vector_store = Chroma.from_documents(docs, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 2})


def format_docs(docs):
    """Joins chunk elements natively with spacing."""
    return "\n\n".join(doc.page_content for doc in docs)


# Initialize application session properties if empty
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Graphical Interface Layout ---

# Sidebar for Knowledge Source Uploads
with st.sidebar:
    st.header("🗂️ Data Ingestion Center")
    uploaded_file = st.file_uploader(
        "Drop reference text documents below:",
        type=["txt", "md", "log"],
        help="Upload text base contexts to anchor Gemini's generation domain.",
    )

    if uploaded_file is not None:
        # Prevent repetitive parsing overhead loops
        if (
            "last_uploaded" not in st.session_state
            or st.session_state.last_uploaded != uploaded_file.name
        ):
            with st.spinner(
                "Analyzing text schema and generating vector index embeddings..."
            ):
                raw_context = uploaded_file.read().decode("utf-8")
                # Initialize new target retriever pipeline instance
                st.session_state.retriever = process_text_into_vectorstore(raw_context)
                st.session_state.last_uploaded = uploaded_file.name
                st.success(f"Successfully processed: '{uploaded_file.name}'!")

    st.markdown("---")
    st.info(
        "💡 **How to operate:** Ensure your `GOOGLE_API_KEY` environment variable is defined, "
        "upload a raw file above, and query it in the primary chat input area."
    )

# Primary Interactive Conversation Space
if st.session_state.retriever is None:
    st.warning(
        "⚠️ Access Denied: Please upload a reference file via the sidebar to initialize the retrieval model context."
    )
else:
    # Render historic session tracking components
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle incoming user message requests
    if user_query := st.chat_input(
        "Ask a question about the uploaded document context..."
    ):
        # Append and render the user's question
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Execute generative completion against retriever pipeline context
        with st.chat_message("assistant"):
            with st.spinner(
                "Searching vector index space & running generation model inference..."
            ):
                try:
                    # 1. Instantiate the foundational model instance
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash", temperature=0.1
                    )

                    # 2. Build target system architecture prompts
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

                    # 3. Assemble declarative chain properties natively
                    rag_chain = (
                        {
                            "context": st.session_state.retriever | format_docs,
                            "input": RunnablePassthrough(),
                        }
                        | rag_prompt
                        | llm
                        | StrOutputParser()
                    )

                    # 4. Invoke model inference execution
                    response_output = rag_chain.invoke(user_query)

                    # Render resulting string response safely back to web layout
                    st.markdown(response_output)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_output}
                    )

                except Exception as error_exception:
                    st.error(f"Execution Error occurred: {str(error_exception)}")
