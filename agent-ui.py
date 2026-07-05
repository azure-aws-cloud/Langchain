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

db_status = get_backend_status()
is_initialized = db_status.get("initialized", False)
doc_count = db_status.get("document_count", 0)
backend_msg = db_status.get("message", "")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Persistent registry track for unique chat inputs to feed the sidebar shortcuts
if "command_history" not in st.session_state:
    st.session_state.command_history = []

# Tracks active query triggered via sidebar actions
active_query = None

# --- Graphical Interface Layout ---

# Sidebar for Knowledge Source Uploads, Command History & Database Maintenance
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

    # --- COMMAND HISTORY SHORTCUT PANEL ---
    st.markdown("---")
    st.subheader("📜 Command History Shortcuts")

    if not st.session_state.command_history:
        st.caption("No recent queries recorded yet. Ask a question below to see history options.")
    else:
        st.caption("Click a question below to instantly execute it:")
        # Render a list of unique shortcuts safely
        for idx, history_item in enumerate(st.session_state.command_history):
            # Enforce short dynamic labels to keep sidebar look compact
            btn_label = history_item if len(history_item) < 35 else f"{history_item[:32]}..."
            if st.button(f"💬 {btn_label}", key=f"hist_{idx}", use_container_width=True, help=history_item):
                active_query = history_item

    st.markdown("---")
    st.subheader("⚙️ Maintenance Panel")

    # Safe database wipe reset implementation via REST call
    if st.button("🗑️ Reset Database (Wipe Local Storage)", type="primary"):
        with st.spinner("Erasing remote disk volumes..."):
            if reset_backend_database():
                st.session_state.messages = []
                st.session_state.command_history = []
                if "last_uploaded" in st.session_state:
                    del st.session_state.last_uploaded
                st.toast("Database folder erased completely from disk.", icon="💥")
                st.rerun()

# Primary Interactive Conversation Space
if backend_msg == "Backend server is offline.":
    st.info(
        "🔌 **Connection Error:** The UI cannot see the REST API server at `http://localhost:8000`. Please start it up.")
elif not is_initialized:
    st.info(
        "ℹ️ **Database Empty:** Please drag and drop a reference text file into the sidebar to establish your local database on-premise.")
else:
    # Render historical conversation components from session history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Fall back to standard input row if no sidebar history action has been taken
    user_query = st.chat_input("Ask a question anchored to your local disk databases...")

    # Prioritise the sidebar history hook over chat input line
    if active_query:
        user_query = active_query

    # Execute generative completions against backend retriever layout if text query is valid
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Append query text safely into history stack tracking uniques explicitly
        if user_query not in st.session_state.command_history:
            st.session_state.command_history.append(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching on-premise data chunks..."):
                response_output = query_rag_chat(user_query)
                if response_output:
                    st.markdown(response_output)
                    st.session_state.messages.append({"role": "assistant", "content": response_output})
                    st.rerun()
