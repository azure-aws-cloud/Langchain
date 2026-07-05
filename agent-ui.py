import streamlit as st
import requests

# Web Application UI Layout Configurations
st.set_page_config(page_title="Persistent Gemini RAG Client", page_icon="💾", layout="wide")
st.title("💾 On-Premise Persistent Gemini RAG Explorer")
st.subheader("Connected to secure FastAPI backend architecture.")

API_BASE_URL = "http://localhost:8000/api"

# --- Helper Functions to Consume REST API ---

def get_backend_status():
    """Fetches initialization status from the FastAPI endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        return {"initialized": False, "document_count": 0, "message": "Backend server is offline."}
    return {"initialized": False, "document_count": 0, "message": "Unknown backend error."}


def upload_file_to_backend(uploaded_file):
    """Sends the file payload directly to the FastAPI document parsing pipeline."""
    try:
        # Prepare the file payload matching FastAPI's expected parameter
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=30)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Upload communication failure: {str(e)}")
        return False


def query_rag_chat(question: str):
    """Submits the prompt string and returns the generated LLM response chunk."""
    try:
        payload = {"question": question}
        response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json().get("response")
        else:
            error_detail = response.json().get("detail", "Unknown server issue.")
            st.error(f"Backend Error: {error_detail}")
    except Exception as e:
        st.error(f"Chat connection failure: {str(e)}")
    return None


def reset_backend_database():
    """Triggers a clean disk wipe via the backend endpoint framework."""
    try:
        response = requests.delete(f"{API_BASE_URL}/reset", timeout=10)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Reset request failure: {str(e)}")
        return False


# --- App State Initializations ---

# Fetch status directly from the REST API layer on reload
db_status = get_backend_status()
is_initialized = db_status.get("initialized", False)
doc_count = db_status.get("document_count", 0)
backend_msg = db_status.get("message", "")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Graphical Interface Layout ---

# Sidebar for Knowledge Source Uploads & Database Maintenance
with st.sidebar:
    st.header("🗂️ On-Premise Data Manager")

    # Real-time backend state indicators
    if backend_msg == "Backend server is offline.":
        st.error("🔴 System Status: Backend Server Offline")
    elif is_initialized:
        st.success(f"🟢 System Status: Ready ({doc_count} Chunks Loaded)")
    else:
        st.warning("🔴 System Status: No Knowledge Base Initialized")

    uploaded_file = st.file_uploader(
        "Append text materials to local disk storage:",
        type=["txt", "md", "log"],
        help="Upload new reference files to store inside your on-premise vector database."
    )

    if uploaded_file is not None:
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            if backend_msg == "Backend server is offline.":
                st.error("Cannot upload. Please start your FastAPI server application first.")
            else:
                with st.spinner("Uploading and indexing text layout via REST API..."):
                    success = upload_file_to_backend(uploaded_file)
                    if success:
                        st.session_state.last_uploaded = uploaded_file.name
                        st.toast("🎉 Document processed and chunked into ChromaDB!", icon="💾")
                        st.rerun()

    st.markdown("---")
    st.subheader("⚙️ Maintenance Panel")

    # Safe database wipe reset implementation via REST call
    if st.button("🗑️ Reset Database (Wipe Local Storage)", type="primary"):
        with st.spinner("Erasing remote disk volumes..."):
            if reset_backend_database():
                st.session_state.messages = []
                if "last_uploaded" in st.session_state:
                    del st.session_state.last_uploaded
                st.toast("Database folder erased completely from disk.", icon="💥")
                st.rerun()

# Primary Interactive Conversation Space
if backend_msg == "Backend server is offline.":
    st.info("🔌 **Connection Error:** The UI cannot see the REST API server at `http://localhost:8000`. Please start it up.")
elif not is_initialized:
    st.info("ℹ️ **Database Empty:** Please drag and drop a reference text file into the sidebar to establish your local database on-premise.")
else:
    # Render historical conversation components from session history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle incoming user query lines
    if user_query := st.chat_input("Ask a question anchored to your local disk databases..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Execute generative completions against backend retriever layout
        with st.chat_message("assistant"):
            with st.spinner("Searching on-premise data chunks..."):
                response_output = query_rag_chat(user_query)
                if response_output:
                    st.markdown(response_output)
                    st.session_state.messages.append({"role": "assistant", "content": response_output})
