# rag_app/ui/streamlit_app.py

import os
import sys
import requests
import streamlit as st
from phoenix.client import Client
from auth.core import authenticate, show_user_info
from datetime import datetime

# Add parent directory to Python path so we can import config
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.settings import settings

# FastAPI backend URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
INGEST_ENDPOINT = f"{API_BASE_URL}/api/v1/ingest/"
BULK_INGEST_ENDPOINT = f"{API_BASE_URL}/api/v1/ingest/bulk"
QUERY_ENDPOINT = f"{API_BASE_URL}/api/v1/query/"
CHAT_SESSIONS_ENDPOINT = f"{API_BASE_URL}/api/v1/chat/sessions"
CHAT_MESSAGES_ENDPOINT = f"{API_BASE_URL}/api/v1/chat/sessions/{{chat_id}}/messages"
CHAT_DELETE_ENDPOINT = f"{API_BASE_URL}/api/v1/chat/sessions/{{chat_id}}"
DOCUMENTS_ENDPOINT = f"{API_BASE_URL}/api/v1/admin/documents/{{backend}}"
EVALUATION_GROUND_TRUTH_ENDPOINT = f"{API_BASE_URL}/api/v1/evaluation/generate-ground-truth"
EVALUATION_EVALUATE_ENDPOINT = f"{API_BASE_URL}/api/v1/evaluation/evaluate"
EVALUATION_FILES_ENDPOINT = f"{API_BASE_URL}/api/v1/evaluation/ground-truth-files"
EVALUATION_DELETE_FILE_ENDPOINT = f"{API_BASE_URL}/api/v1/evaluation/ground-truth-files/{{filename}}"
EVALUATION_PREVIEW_FILE_ENDPOINT = f"{API_BASE_URL}/api/v1/evaluation/ground-truth-files/{{filename}}/preview"
EVALUATION_RESULTS_ENDPOINT = f"{API_BASE_URL}/api/v1/evaluation/evaluation-results"
EVALUATION_RESULTS_PREVIEW_ENDPOINT = f"{API_BASE_URL}/api/v1/evaluation/evaluation-results/{{filename}}/preview"


auth_status, authenticator = authenticate()
show_user_info(authenticator)
# Page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Open WebUI-inspired dark theme
st.markdown("""
<style>
    /* Main container styling - Dark theme like Open WebUI */
    .main {
        background-color: #0f0f0f;
        padding: 2rem;
        color: #e5e5e5;
    }

    /* Overall app background */
    .stApp {
        background-color: #0f0f0f;
    }

    /* Toolbar - Make it black */
    [data-testid="stToolbar"],
    .stAppToolbar,
    .st-emotion-cache-14vh5up {
        background-color: #0f0f0f !important;
        border-bottom: 1px solid #2a2a2a !important;
    }

    /* Toolbar inner containers */
    [data-testid="stToolbar"] > div,
    .stAppToolbar > div,
    .st-emotion-cache-1j22a0y,
    .st-emotion-cache-70qvj9,
    .st-emotion-cache-scp8yw {
        background-color: transparent !important;
    }

    /* Toolbar buttons */
    [data-testid="stToolbar"] button,
    .stAppToolbar button {
        color: #e5e5e5 !important;
        background-color: transparent !important;
    }

    /* Toolbar actions container */
    [data-testid="stToolbarActions"],
    .stToolbarActions {
        background-color: transparent !important;
    }

    /* Main menu button */
    [data-testid="stMainMenu"],
    .stMainMenu {
        color: #e5e5e5 !important;
    }

    [data-testid="stMainMenu"] button {
        background-color: transparent !important;
    }

    /* Main menu icon (three dots) */
    [data-testid="stMainMenu"] svg {
        color: #e5e5e5 !important;
        fill: #e5e5e5 !important;
    }

    /* Deploy button */
    [data-testid="stAppDeployButton"] button {
        background-color: #3b82f6 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
    }

    [data-testid="stAppDeployButton"] button:hover {
        background-color: #2563eb !important;
    }

    /* Status widget (Rerun notification) - Dark theme */
    [data-testid="stStatusWidget"],
    .stStatusWidget {
        background-color: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        color: #e5e5e5 !important;
    }

    /* Status widget label */
    [data-testid="stStatusWidget"] label {
        color: #e5e5e5 !important;
    }

    /* Status widget icon */
    [data-testid="stStatusWidget"] span[data-testid="stIconMaterial"] {
        color: #3b82f6 !important;
    }

    /* Status widget buttons (Rerun, Always rerun) */
    [data-testid="stStatusWidget"] button {
        background-color: #2a2a2a !important;
        color: #e5e5e5 !important;
        border: 1px solid #404040 !important;
    }

    [data-testid="stStatusWidget"] button:hover {
        background-color: #3b82f6 !important;
        color: white !important;
        border-color: #3b82f6 !important;
    }

    /* Chat container - Dark theme */
    .stChatMessage {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }

    /* User message */
    .stChatMessage[data-testid="user-message"] {
        background-color: #1e3a5f;
        border: 1px solid #2c5282;
    }

    /* Sidebar styling - Dark theme */
    [data-testid="stSidebar"] {
        background-color: #171717;
        border-right: 1px solid #2a2a2a;
    }

    [data-testid="stSidebar"] .stMarkdown p {
        color: #e5e5e5;
    }

    [data-testid="stSidebar"] .stMarkdown span {
        color: #e5e5e5;
    }

    /* Headers - White text on dark */
    h1 {
        color: #ffffff;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        margin-bottom: 2rem;
    }

    h2, h3 {
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    h4, h5, h6 {
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Info boxes - Dark theme */
    .stAlert {
        border-radius: 10px;
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        color: #e5e5e5;
    }

    /* Buttons - Modern dark theme */
    .stButton > button {
        border-radius: 8px;
        background-color: #3b82f6;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }

    .stButton > button[kind="secondary"] {
        background-color: #2a2a2a;
        border: 1px solid #404040;
    }

    .stButton > button[kind="secondary"]:hover {
        background-color: #333333;
    }

    /* Input fields - Dark theme */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea textarea {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        color: #e5e5e5;
        padding: 0.5rem;
    }

    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 1px #3b82f6;
    }

    /* Metrics - Dark theme */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #3b82f6;
        font-weight: bold;
    }

    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #a0a0a0;
    }

    /* File uploader - Dark theme */
    .stFileUploader {
        background-color: #1a1a1a;
        border: 1px dashed #2a2a2a;
        border-radius: 10px;
        padding: 1rem;
    }

    /* Tabs - Dark theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #171717;
        border-radius: 10px;
        padding: 0.5rem;
        border: 1px solid #2a2a2a;
    }

    .stTabs [data-baseweb="tab"] {
        color: #a0a0a0;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #2a2a2a;
        color: #e5e5e5;
    }

    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }

    /* Expanders - Dark theme */
    .streamlit-expanderHeader {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        font-weight: 600;
        color: #e5e5e5;
    }

    .streamlit-expanderHeader:hover {
        background-color: #2a2a2a;
    }

    /* Code blocks - Dark theme */
    .stCodeBlock {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
    }

    code {
        background-color: #1a1a1a;
        color: #e5e5e5;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
    }

    /* Fix chat input - Dark theme */
    [data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 200pt !important;
    right: 0 !important;
    padding: 1rem 3rem !important;
    z-index: 1000 !important;
    margin: 0 !important;

    /* ✅ Gradient: black bottom → transparent top */
    background: linear-gradient(to top, rgba(0,0,0,1) 0%, rgba(0,0,0,0) 100%) !important;

    /* Optional: round top corners */
    border-radius: 12px 12px 0 0;
}


    [data-testid="stChatInput"] input {
        border: 1px solid #2a2a2a !important;
        color: #e5e5e5 !important;
    }

    /* Adjust for sidebar */
    [data-testid="stChatInput"] {
        margin-left: 7rem !important;
    }

    /* Bottom padding to prevent content overlap */
    .main {
        padding-bottom: 120px !important;
    }

    /* Dividers - Dark theme */
    hr {
        border-color: #2a2a2a;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }

    .badge-success {
        background-color: #10b981;
        color: white;
    }

    .badge-info {
        background-color: #3b82f6;
        color: white;
    }

    .badge-warning {
        background-color: #f59e0b;
        color: white;
    }

    /* Progress bars - Dark theme */
    .stProgress > div > div {
        background-color: #3b82f6;
    }

    /* Markdown text in main area */
    .stMarkdown {
        color: #e5e5e5;
    }

    /* Force all headings to be white */
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4,
    .stMarkdown h5,
    .stMarkdown h6 {
        color: #ffffff !important;
    }

    /* Streamlit specific heading components */
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMarkdownContainer"] h6 {
        color: #ffffff !important;
    }

    /* Streamlit title and subheader */
    .stTitle,
    .stSubheader,
    [class*="stTitle"],
    [class*="stSubheader"] {
        color: #ffffff !important;
    }

    /* Any element with these classes */
    .e1nzilvr0,
    .e1nzilvr1,
    .e1nzilvr2,
    .e1nzilvr3,
    .e1nzilvr4,
    .e1nzilvr5 {
        color: #ffffff !important;
    }

    /* Force all paragraph children that are actually headers */
    p > strong {
        color: #ffffff;
    }

    /* Super aggressive heading override - catches everything */
    .main h1,
    .main h2,
    .main h3,
    .main h4,
    .main h5,
    .main h6,
    div[data-testid="column"] h1,
    div[data-testid="column"] h2,
    div[data-testid="column"] h3,
    div[data-testid="column"] h4,
    div[data-testid="column"] h5,
    div[data-testid="column"] h6,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: #ffffff !important;
    }

    /* Captions */
    .stCaption {
        color: #a0a0a0;
    }

    /* Success/Error/Warning messages - Dark theme */
    .stSuccess {
        background-color: #1a3a1a;
        border: 1px solid #10b981;
        color: #10b981;
    }

    .stError {
        background-color: #3a1a1a;
        border: 1px solid #ef4444;
        color: #ef4444;
    }

    .stWarning {
        background-color: #3a2f1a;
        border: 1px solid #f59e0b;
        color: #f59e0b;
    }

    .stInfo {
        background-color: #1a2a3a;
        border: 1px solid #3b82f6;
        color: #3b82f6;
    }

    /* Multiselect - Dark theme */
    .stMultiSelect > div > div {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
    }

    .stMultiSelect [data-baseweb="tag"] {
        background-color: #3b82f6;
        color: white;
    }

    /* Radio buttons - Dark theme */
    .stRadio > label {
        color: #e5e5e5;
    }

    /* Slider - Dark theme */
    .stSlider > div > div > div {
        background-color: #3b82f6;
    }

    /* Checkbox - Dark theme */
    .stCheckbox > label {
        color: #e5e5e5;
    }

    /* Spinner - Dark theme */
    .stSpinner > div {
        border-top-color: #3b82f6;
    }

    /* Tables - Dark theme */
    table {
        background-color: #1a1a1a;
        color: #e5e5e5;
    }

    thead tr {
        background-color: #2a2a2a;
    }

    tbody tr {
        border-bottom: 1px solid #2a2a2a;
    }

    tbody tr:hover {
        background-color: #252525;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

if "total_documents" not in st.session_state:
    st.session_state.total_documents = 0

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []

if "selected_documents" not in st.session_state:
    st.session_state.selected_documents = []


def load_chat_sessions():
    """Load all chat sessions from the backend."""
    try:
        response = requests.get(CHAT_SESSIONS_ENDPOINT, timeout=10)
        if response.status_code == 200:
            st.session_state.chat_sessions = response.json()
        else:
            st.session_state.chat_sessions = []
    except Exception as e:
        st.error(f"Failed to load chat sessions: {str(e)}")
        st.session_state.chat_sessions = []


def load_chat_messages(chat_id: str):
    """Load messages for a specific chat session."""
    try:
        url = CHAT_MESSAGES_ENDPOINT.format(chat_id=chat_id)
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            messages = response.json()
            # Convert to Streamlit message format
            st.session_state.messages = []
            for msg in messages:
                message_dict = {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                if msg.get("sources"):
                    import json
                    try:
                        message_dict["sources"] = json.loads(msg["sources"])
                    except:
                        pass
                if msg.get("cost"):
                    message_dict["cost"] = msg["cost"]
                st.session_state.messages.append(message_dict)
    except Exception as e:
        st.error(f"Failed to load chat messages: {str(e)}")


def start_new_chat():
    """Start a new chat session."""
    st.session_state.current_chat_id = None
    st.session_state.messages = []
    st.session_state.chat_history = []


def delete_chat_session(chat_id: str):
    """Delete a chat session."""
    try:
        url = CHAT_DELETE_ENDPOINT.format(chat_id=chat_id)
        response = requests.delete(url, timeout=10)
        if response.status_code == 200:
            # Reload chat sessions
            load_chat_sessions()
            # If deleted chat was current, start new chat
            if st.session_state.current_chat_id == chat_id:
                start_new_chat()
            st.success("Chat deleted successfully!")
        else:
            st.error("Failed to delete chat")
    except Exception as e:
        st.error(f"Error deleting chat: {str(e)}")

def sidebar_prompt_version_selector(llm_backend: str):
    """
    Display prompt version selector in the Streamlit sidebar.
    Fetches available prompt versions from Phoenix based on the LLM backend.
    """
    client = Client()
    base_url = os.getenv("PHOENIX_BASE_URL", "http://localhost:6006")

    # Pick prompt identifier dynamically
    prompt_identifier = (
        settings.prompts.openai_prompt if llm_backend == "openai"
        else settings.prompts.ollama_prompt
    )

    st.subheader("🧠 Prompt Version")
    try:
        # Fetch available prompt versions from Phoenix REST API
        resp = requests.get(
            f"{base_url}/v1/prompts/{prompt_identifier}/versions",
            params={"limit": 20},
            timeout=10
        )

        if resp.status_code == 200:
            data = resp.json()
            versions = data.get("data", [])
        else:
            versions = []
            st.warning(f"Unable to fetch prompt versions ({resp.status_code})")

    except Exception as e:
        versions = []
        st.error(f"Error fetching prompt versions: {str(e)}")

    # Build selection dropdown
    if versions:
        version_labels = {
            v["id"]: f"{v['id']} ({v.get('model_name', 'unknown')} - {v.get('created_at', 'n/a')})"
            for v in versions
        }

        selected_vid = st.selectbox(
            "Select prompt version",
            options=list(version_labels.keys()),
            format_func=lambda vid: version_labels[vid],
            key=f"prompt_version_{llm_backend}"
        )

        st.session_state.selected_prompt_version_id = selected_vid
    else:
        st.info("No prompt versions available.")
        st.session_state.selected_prompt_version_id = None

    return st.session_state.get("selected_prompt_version_id")

def sidebar_settings():
    """Render sidebar with settings and document upload."""
    with st.sidebar:
        # Chat History Section
        st.title("💬 Chats")

        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            start_new_chat()
            st.rerun()

        st.divider()

        # Load chat sessions on first run
        if not st.session_state.chat_sessions:
            load_chat_sessions()

        # Display chat sessions
        if st.session_state.chat_sessions:
            st.subheader("📝 Chat History")
            for chat in st.session_state.chat_sessions:
                col1, col2 = st.columns([4, 1.5])
                with col1:
                    # Button to select chat
                    if st.button(
                        chat["title"],
                        key=f"chat_{chat['id']}",
                        use_container_width=True,
                        type="secondary" if st.session_state.current_chat_id != chat["id"] else "primary"
                    ):
                        st.session_state.current_chat_id = chat["id"]
                        load_chat_messages(chat["id"])
                        st.rerun()
                with col2:
                    # Delete button
                    if st.button("🗑️", key=f"delete_{chat['id']}"):
                        delete_chat_session(chat["id"])
                        st.rerun()
        else:
            st.info("No chat history yet. Start a new chat!")

        st.divider()
        st.title("⚙️ Settings")

        # Backend selection
        st.subheader("🔧 LLM Backend")
        llm_backend = st.selectbox(
            "Generation Model",
            options=["openai", "ollama"],
            index=0 if settings.llm_backend == "openai" else 1,
            help="Choose which LLM to use for answering questions",
            key="llm_backend"
        )

        if llm_backend == "openai":
            st.caption(f"🌐 Model: {settings.openai.model}")
        else:
            st.caption(f"🏠 Model: {settings.ollama.model}")
        if "last_backend" not in st.session_state or st.session_state.last_backend != llm_backend:
            st.session_state.selected_prompt_version_id = None
        st.session_state.last_backend = llm_backend

        sidebar_prompt_version_selector(llm_backend)


        # Display available documents with selector
        st.markdown("#### 📚 Document Filter")
        try:
            url = DOCUMENTS_ENDPOINT.format(backend=llm_backend)
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                docs_data = response.json()
                total_docs = docs_data.get("total_documents", 0)
                total_chunks = docs_data.get("total_chunks", 0)
                documents = docs_data.get("documents", [])

                if total_docs > 0:
                    st.caption(f"**{total_docs} documents** ({total_chunks} chunks)")

                    # Document selection
                    doc_filenames = [doc["filename"] for doc in documents]

                    # Add "All Documents" option
                    selected = st.multiselect(
                        "Select documents to search",
                        options=doc_filenames,
                        default=[],
                        help="Leave empty to search all documents, or select specific documents to limit the search",
                        key=f"doc_selector_{llm_backend}"
                    )

                    # Update session state
                    st.session_state.selected_documents = selected if selected else []

                    # Show selection status
                    if st.session_state.selected_documents:
                        st.success(f"🔍 Searching {len(st.session_state.selected_documents)} selected document(s)")
                    else:
                        st.info("🔍 Searching all documents")

                    # Show document details in expander
                    with st.expander("View Document Details"):
                        for doc in documents:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.text(doc["filename"])
                            with col2:
                                st.caption(f"{doc['chunk_count']} chunks")
                else:
                    st.info("No documents yet. Upload some documents to get started!")
            else:
                st.warning("Unable to load documents")
        except Exception as e:
            st.warning(f"Failed to load documents: {str(e)}")

        st.divider()

        # Retrieval settings
        st.subheader("🔍 Retrieval Settings")
        top_k = st.slider(
            "Top K chunks to retrieve",
            min_value=5,
            max_value=50,
            value=settings.retrieval.top_k,
            help="Number of chunks to retrieve before reranking"
        )

        rerank_top_m = st.slider(
            "Top M after reranking",
            min_value=1,
            max_value=10,
            value=settings.retrieval.rerank_top_m,
            help="Number of chunks to use for generation"
        )

        st.divider()


        return llm_backend, top_k


def chat_tab(llm_backend, top_k):
    """Chat interface tab."""
    # Container for chat messages
    chat_container = st.container()

    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show sources if available
                if message.get("sources"):
                    with st.expander("📚 View Sources"):
                        for idx, source in enumerate(message["sources"], 1):
                            metadata = source.get('metadata', {})
                            st.markdown(f"**Source {idx}**")

                            # Display file name
                            file_name = metadata.get('source_file', 'Unknown')
                            st.caption(f"**File:** {file_name}")

                            # Display page number if available
                            page_numbers = metadata.get('page_numbers', [])
                            page = metadata.get('page')
                            if page_numbers:
                                pages_str = ", ".join(map(str, page_numbers))
                                st.caption(f"**Page(s):** {pages_str}")
                            elif page:
                                st.caption(f"**Page:** {page}")

                            # Display text preview
                            st.text(source.get('text', '')[:200] + "...")
                            st.divider()

    # Chat input - placed after container to stay at bottom
    prompt = st.chat_input("Ask me anything about your documents...")

    if prompt:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from RAG system
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Make API request with selected documents filter
                    payload = {
                        "query": prompt,
                        "top_k": top_k,
                        "llm_backend": llm_backend,
                        "chat_session_id": st.session_state.current_chat_id,
                        "selected_documents": st.session_state.selected_documents if st.session_state.selected_documents else None,
                        "prompt_version_id": st.session_state.get("selected_prompt_version_id")
                    }

                    response = requests.post(QUERY_ENDPOINT, json=payload)

                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get("answer", "I couldn't generate an answer.")
                        used_chunks = result.get("used_chunks", [])
                        cost_usd = result.get("cost_usd", 0.0)
                        chat_session_id = result.get("chat_session_id")
                        retrieved_count = result.get("retrieved_count")
                        reranked_count = result.get("reranked_count")

                        # Update current chat ID if it was a new chat
                        if not st.session_state.current_chat_id and chat_session_id:
                            st.session_state.current_chat_id = chat_session_id
                            # Reload chat sessions to show the new chat
                            load_chat_sessions()

                        # Display answer
                        st.markdown(answer)

                        # Show rerank stats and cost
                        info_parts = []
                        if retrieved_count and reranked_count:
                            rerank_percentage = (reranked_count / retrieved_count) * 100
                            info_parts.append(f"🔄 Retrieved: {retrieved_count} → Reranked: {reranked_count} ({rerank_percentage:.1f}%)")
                        if cost_usd > 0:
                            info_parts.append(f"💰 Cost: ${cost_usd:.4f}")

                        if info_parts:
                            st.caption(" | ".join(info_parts))

                        # Add assistant message to chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": used_chunks,
                            "cost": cost_usd
                        })

                        # Update statistics
                        st.session_state.total_queries += 1

                        # Show sources
                        if used_chunks:
                            with st.expander("📚 View Sources"):
                                for idx, chunk in enumerate(used_chunks, 1):
                                    metadata = chunk.get('metadata', {})
                                    st.markdown(f"**Source {idx}**")

                                    # Display file name
                                    file_name = metadata.get('source_file', 'Unknown')
                                    st.caption(f"**File:** {file_name}")

                                    # Display page number if available
                                    page_numbers = metadata.get('page_numbers', [])
                                    page = metadata.get('page')
                                    if page_numbers:
                                        pages_str = ", ".join(map(str, page_numbers))
                                        st.caption(f"**Page(s):** {pages_str}")
                                    elif page:
                                        st.caption(f"**Page:** {page}")

                                    # Display text preview
                                    st.text(chunk.get('text', '')[:200] + "...")
                                    st.divider()
                    else:
                        error_msg = f"API request failed with status {response.status_code}"
                        st.error(f"❌ {error_msg}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(f"❌ {error_msg}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


def upload_tab():
    """Document upload interface tab."""
    st.markdown("### 📤 Upload Documents to Knowledge Base")
    st.markdown("Add documents to your RAG system for intelligent question answering. Supports PDF, DOCX, TXT, and Markdown files.")

    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Supported Formats", "4 Types")
    with col2:
        st.metric("Max File Size", "100 MB")
    with col3:
        st.metric("Processing", "Automatic")

    st.divider()

    # Upload mode selector
    upload_mode = st.radio(
        "Upload Mode",
        options=["Single File", "Multiple Files"],
        horizontal=True,
        help="Choose whether to upload one file or multiple files at once"
    )

    # Backend selection
    upload_backend = st.selectbox(
        "Embedding Backend",
        options=["openai", "ollama"],
        index=0 if settings.embedding_backend == "openai" else 1,
        help="Choose which embedding model to use for processing documents"
    )

    st.divider()

    if upload_mode == "Single File":
        # Single file upload
        st.markdown("#### 📄 Single File Upload")
        st.info("💡 Upload one document at a time for precise control over your knowledge base.")

        uploaded_file = st.file_uploader(
            "Choose a document",
            type=["pdf", "docx", "txt", "md"],
            help="Upload a single document to add to the knowledge base"
        )

        if uploaded_file is not None:
            # Display file info in a professional card
            st.markdown("##### 📋 File Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📝 Filename", uploaded_file.name, help="Name of the uploaded file")
            with col2:
                file_size_kb = uploaded_file.size / 1024
                if file_size_kb > 1024:
                    st.metric("💾 Size", f"{file_size_kb / 1024:.2f} MB", help="File size in megabytes")
                else:
                    st.metric("💾 Size", f"{file_size_kb:.2f} KB", help="File size in kilobytes")
            with col3:
                file_type = uploaded_file.type.split('/')[-1].upper()
                st.metric("📄 Type", file_type, help="Document file type")

            st.divider()

            # Upload button with better styling
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("📤 Upload & Process Document", use_container_width=True, type="primary"):
                    with st.spinner(f"🔄 Processing document with {upload_backend.upper()} embeddings..."):
                        progress_bar = st.progress(0)
                        try:
                            progress_bar.progress(20)
                            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                            data = {"backend": upload_backend}

                            progress_bar.progress(40)
                            response = requests.post(INGEST_ENDPOINT, files=files, data=data, timeout=300)
                            progress_bar.progress(80)

                            if response.status_code == 200:
                                result = response.json()
                                progress_bar.progress(100)
                                if result.get("success"):
                                    st.success(f"✅ {result.get('message')}")
                                    st.session_state.total_documents += 1

                                    # Show additional info if available
                                    if result.get('chunks_created'):
                                        st.info(f"📊 Created {result.get('chunks_created')} text chunks for retrieval")
                                else:
                                    st.error(f"❌ {result.get('message', 'Unknown error')}")
                            else:
                                st.error(f"❌ API request failed with status code: {response.status_code}")
                        except Exception as e:
                            st.error(f"❌ Error during upload: {str(e)}")
                        finally:
                            progress_bar.empty()
            with col2:
                if st.button("🔄 Clear", use_container_width=True):
                    st.rerun()

    else:
        # Multiple files upload
        st.markdown("#### 📚 Multiple Files Upload")
        st.info("💡 Upload multiple documents at once for efficient batch processing.")

        uploaded_files = st.file_uploader(
            "Choose documents",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help="Upload multiple documents to add to the knowledge base"
        )

        if uploaded_files and len(uploaded_files) > 0:
            # Display summary metrics
            st.markdown("##### 📊 Upload Summary")
            total_size = sum(f.size for f in uploaded_files)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📁 Files Selected", len(uploaded_files))
            with col2:
                st.metric("💾 Total Size", f"{total_size / (1024*1024):.2f} MB")
            with col3:
                file_types = set(f.type.split('/')[-1].upper() for f in uploaded_files)
                st.metric("📄 File Types", len(file_types))

            # Show file list
            with st.expander("📋 View All Selected Files", expanded=False):
                for idx, file in enumerate(uploaded_files, 1):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.markdown(f"**{idx}.** {file.name}")
                    with col2:
                        size_kb = file.size / 1024
                        if size_kb > 1024:
                            st.text(f"{size_kb / 1024:.2f} MB")
                        else:
                            st.text(f"{size_kb:.2f} KB")
                    with col3:
                        st.text(file.type.split('/')[-1].upper())

            st.divider()

            # Upload buttons
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("📤 Process All Documents", use_container_width=True, type="primary"):
                    with st.spinner(f"🔄 Processing {len(uploaded_files)} documents with {upload_backend.upper()} embeddings..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        try:
                            status_text.text("Preparing files for upload...")
                            progress_bar.progress(10)

                            # Prepare files for bulk upload
                            files = [
                                ('files', (file.name, file.getvalue(), file.type))
                                for file in uploaded_files
                            ]
                            data = {"backend": upload_backend}

                            status_text.text(f"Uploading {len(uploaded_files)} files to server...")
                            progress_bar.progress(30)

                            response = requests.post(BULK_INGEST_ENDPOINT, files=files, data=data, timeout=600)
                            progress_bar.progress(90)

                            if response.status_code == 200:
                                result = response.json()
                                progress_bar.progress(100)
                                status_text.text("✅ Upload complete!")

                                # Display summary with enhanced styling
                                st.markdown("---")
                                st.markdown("### 📊 Processing Results")

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("📁 Total Files", result.get("total_files", 0))
                                with col2:
                                    successful = result.get("successful", 0)
                                    st.metric("✅ Successful", successful, delta=f"+{successful}")
                                with col3:
                                    failed = result.get("failed", 0)
                                    st.metric("❌ Failed", failed, delta=f"-{failed}" if failed > 0 else "0")

                                # Update total documents count
                                st.session_state.total_documents += result.get("successful", 0)

                                st.divider()

                                # Display detailed results
                                st.markdown("#### 📋 Individual File Results")
                                results = result.get("results", [])

                                # Separate successful and failed results
                                successful_results = [r for r in results if r.get("success")]
                                failed_results = [r for r in results if not r.get("success")]

                                # Show successful results
                                if successful_results:
                                    with st.expander(f"✅ Successful Uploads ({len(successful_results)})", expanded=True):
                                        for file_result in successful_results:
                                            col1, col2 = st.columns([3, 1])
                                            with col1:
                                                st.success(f"**{file_result.get('filename')}**")
                                            with col2:
                                                if file_result.get('chunks_created'):
                                                    st.caption(f"📊 {file_result.get('chunks_created')} chunks")

                                # Show failed results
                                if failed_results:
                                    with st.expander(f"❌ Failed Uploads ({len(failed_results)})", expanded=True):
                                        for file_result in failed_results:
                                            st.error(f"**{file_result.get('filename')}**: {file_result.get('message')}")

                            else:
                                st.error(f"❌ API request failed with status code: {response.status_code}")
                        except Exception as e:
                            st.error(f"❌ Error during bulk upload: {str(e)}")
                            import traceback
                            with st.expander("🔍 Error Details"):
                                st.code(traceback.format_exc())
                        finally:
                            progress_bar.empty()
                            status_text.empty()

            with col2:
                if st.button("🔄 Clear", use_container_width=True):
                    st.rerun()


def prompt_tab():
    """Enhanced Prompt version details tab."""
    st.markdown("### 🧠 Prompt Template Management")

    version_id = st.session_state.get("selected_prompt_version_id")

    if not version_id:
        st.info("ℹ️ No prompt version selected. Please select a prompt version from the sidebar to view details.")

        # Show helpful information
        st.markdown("""
        ---
        #### What are Prompt Templates?

        Prompt templates define how the RAG system formats questions and context for the LLM.
        They control:
        - 📝 How the query is structured
        - 📚 How retrieved context is presented
        - 🔄 How conversation history is included
        - 🎯 The instruction style and tone

        **Select a version from the sidebar** to view and manage prompt templates.
        """)
        return

    try:
        client = Client()
        prompt_obj = client.prompts.get(prompt_version_id=version_id)

        # Use dict of internal attributes if available
        raw = prompt_obj.__dict__ if hasattr(prompt_obj, "__dict__") else {}

        prompt_id = raw.get("_id", version_id)
        model_name = raw.get("_model_name", "Unknown")
        created_at = raw.get("_created_at", "Unknown")
        template_obj = raw.get("_template", {})

        # Display prompt metadata in a professional card
        st.markdown("#### 📋 Prompt Metadata")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Version ID", prompt_id)
        with col2:
            st.metric("Model", model_name)
        with col3:
            if created_at != "Unknown":
                try:
                    from datetime import datetime
                    if isinstance(created_at, str):
                        created_at_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        formatted_date = created_at_obj.strftime("%Y-%m-%d %H:%M")
                    else:
                        formatted_date = str(created_at)
                    st.metric("Created", formatted_date)
                except:
                    st.metric("Created", "Unknown")
            else:
                st.metric("Created", "Unknown")

        st.divider()

        # Display template configuration
        st.markdown("#### ⚙️ Template Configuration")

        if template_obj:
            config_col1, config_col2 = st.columns(2)

            with config_col1:
                temperature = template_obj.get("temperature", "Not specified")
                st.markdown(f"**🌡️ Temperature:** `{temperature}`")

            with config_col2:
                max_tokens = template_obj.get("max_tokens", "Not specified")
                st.markdown(f"**📏 Max Tokens:** `{max_tokens}`")

        st.divider()

        # Display template messages
        st.markdown("#### 💬 Prompt Template")

        messages = template_obj.get("messages", [])

        if messages:
            # Display each message in the template
            for idx, msg in enumerate(messages, 1):
                role = msg.get("role", "unknown")
                content = msg.get("content", [])

                # Extract text from content
                if isinstance(content, list) and len(content) > 0:
                    text = content[0].get("text", "")
                else:
                    text = str(content)

                # Create an expander for each message
                with st.expander(f"📨 Message {idx}: {role.upper()}", expanded=(idx == 1)):
                    # Display role badge
                    role_color = {
                        "system": "info",
                        "user": "success",
                        "assistant": "warning"
                    }.get(role.lower(), "info")

                    st.markdown(f'<span class="status-badge badge-{role_color}">{role.upper()}</span>',
                              unsafe_allow_html=True)

                    # Display the text content
                    st.markdown("**Content:**")
                    st.code(text, language="text")

                    # Show character count
                    st.caption(f"📊 Length: {len(text)} characters")
        else:
            # Fallback: try to show full_text if messages extraction fails
            full_text = "\n".join(
                msg.get("content", [{}])[0].get("text", "")
                for msg in template_obj.get("messages", [])
            )
            st.code(full_text, language="text")

        st.divider()

        # Display variables if available
        st.markdown("#### 🔧 Available Variables")
        st.markdown("""
        The prompt template supports the following variables:
        - `{query}` - The user's question
        - `{context}` - Retrieved document chunks
        - `{conversation_history}` - Previous messages in the conversation
        """)

        # Show example usage
        with st.expander("📖 View Example Usage"):
            st.markdown("""
            **Example Query:**
            ```
            What is machine learning?
            ```

            **Context** will be automatically populated with relevant chunks from your documents.

            **Conversation History** maintains the context of the ongoing conversation.
            """)

    except Exception as e:
        st.error(f"❌ Could not load prompt details: {str(e)}")
        with st.expander("Error Details"):
            import traceback
            st.code(traceback.format_exc())


def main():
    """Main application."""
    # Title with emoji
    st.title("🤖 RAG Assistant")
    st.markdown("""
    <p style='text-align: center; color: #a0a0a0; font-size: 1rem; margin-top: -1rem; margin-bottom: 2rem;'>
    Intelligent Document Question-Answering System powered by RAG
    </p>
    """, unsafe_allow_html=True)

    # Get settings from sidebar
    llm_backend, top_k = sidebar_settings()


    # Create tabs - Reordered: Upload, Prompts, Chat
    tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "🧠 Prompt Templates", "💬 Chat"])

    with tab1:
        upload_tab()

    with tab2:
        prompt_tab()

    with tab3:
        if not st.session_state.messages:
            # Show welcome message when chat is empty
            st.markdown("""
            ### 💬 Welcome to RAG Chat!

            Start a conversation with your documents. The system will:
            - 🔍 Search through your uploaded documents
            - 📊 Retrieve the most relevant information
            - 🤖 Generate accurate, contextual answers
            - 📚 Provide source references

            **Type your question below to get started!**
            """)
        chat_tab(llm_backend, top_k)


if __name__ == "__main__":
    main()
