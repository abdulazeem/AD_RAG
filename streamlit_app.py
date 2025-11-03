# rag_app/ui/streamlit_app.py

import os
import sys
import requests
import streamlit as st
from phoenix.client import Client
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

# Page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }

    /* Chat container */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    /* Headers */
    h1 {
        color: white;
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    h2, h3 {
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.9);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid #667eea;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #667eea;
    }

    /* File uploader */
    .stFileUploader {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 1rem;
    }

    /* Fix chat input - keep at bottom */
    [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 0 !important;
        left: 300pt !important;
        right: 0 !important;
        padding: 1rem 3rem !important;
        background: linear-gradient(to top, rgba(102, 126, 234, 0.95), #ffffff !important;
        z-index: 1000 !important;
        margin: 0 !important;
    }

    /* Adjust for sidebar */
    [data-testid="stChatInput"] {
        margin-left: 21rem !important;
    }

    /* Bottom padding to prevent content overlap */
    .main {
        padding-bottom: 120px !important;
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
    st.markdown("Upload single or multiple documents to add them to your RAG knowledge base.")

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
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=["pdf", "docx", "txt", "md"],
            help="Upload a single document to add to the knowledge base"
        )

        if uploaded_file is not None:
            # Display file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Filename", uploaded_file.name)
            with col2:
                st.metric("Size", f"{uploaded_file.size / 1024:.2f} KB")
            with col3:
                st.metric("Type", uploaded_file.type.split('/')[-1].upper())

            st.divider()

            if st.button("📤 Upload & Ingest", use_container_width=True, type="primary"):
                with st.spinner(f"Ingesting document with {upload_backend.upper()}..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        data = {"backend": upload_backend}
                        response = requests.post(INGEST_ENDPOINT, files=files, data=data, timeout=300)

                        if response.status_code == 200:
                            result = response.json()
                            if result.get("success"):
                                st.success(f"✅ {result.get('message')}")
                                st.session_state.total_documents += 1

                            else:
                                st.error(f"❌ {result.get('message', 'Unknown error')}")
                        else:
                            st.error(f"❌ API request failed: {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    else:
        # Multiple files upload
        st.markdown("#### 📚 Multiple Files Upload")
        uploaded_files = st.file_uploader(
            "Choose documents",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help="Upload multiple documents to add to the knowledge base"
        )

        if uploaded_files and len(uploaded_files) > 0:
            # Display files info
            st.markdown(f"**Selected Files:** {len(uploaded_files)}")

            # Show file list
            with st.expander("View Selected Files"):
                for idx, file in enumerate(uploaded_files, 1):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.text(f"{idx}. {file.name}")
                    with col2:
                        st.text(f"{file.size / 1024:.2f} KB")
                    with col3:
                        st.text(file.type.split('/')[-1].upper())

            st.divider()

            if st.button("📤 Upload & Ingest All", use_container_width=True, type="primary"):
                with st.spinner(f"Ingesting {len(uploaded_files)} documents with {upload_backend.upper()}..."):
                    try:
                        # Prepare files for bulk upload
                        files = [
                            ('files', (file.name, file.getvalue(), file.type))
                            for file in uploaded_files
                        ]
                        data = {"backend": upload_backend}

                        response = requests.post(BULK_INGEST_ENDPOINT, files=files, data=data, timeout=600)

                        if response.status_code == 200:
                            result = response.json()

                            # Display summary
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Files", result.get("total_files", 0))
                            with col2:
                                st.metric("✅ Successful", result.get("successful", 0))
                            with col3:
                                st.metric("❌ Failed", result.get("failed", 0))

                            # Update total documents count
                            st.session_state.total_documents += result.get("successful", 0)

                            st.divider()

                            # Display detailed results
                            st.markdown("#### 📊 Detailed Results")
                            results = result.get("results", [])

                            for file_result in results:
                                if file_result.get("success"):
                                    with st.expander(f"✅ {file_result.get('filename')}", expanded=False):
                                        st.success(file_result.get('message'))
                                        if file_result.get('chunks_created'):
                                            st.caption(f"Chunks created: {file_result.get('chunks_created')}")
                                else:
                                    with st.expander(f"❌ {file_result.get('filename')}", expanded=False):
                                        st.error(file_result.get('message'))

                        else:
                            st.error(f"❌ API request failed: {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        import traceback
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())


def main():
    """Main application."""
    # Title with emoji
    st.title("🤖 RAG Assistant")

    # Get settings from sidebar
    llm_backend, top_k = sidebar_settings()

    # Create tabs
    tab1, tab2 = st.tabs(["📤 Upload", "💬 Chat"])

    with tab1:
        upload_tab()

    with tab2:
        st.markdown("### Ask questions about your documents!")
        chat_tab(llm_backend, top_k)


if __name__ == "__main__":
    main()
