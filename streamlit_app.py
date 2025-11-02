# rag_app/ui/streamlit_app.py

import os
import sys
import requests
import streamlit as st
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
RERANK_ENDPOINT = f"{API_BASE_URL}/api/v1/rerank/"
DOCUMENTS_ENDPOINT = f"{API_BASE_URL}/api/v1/admin/documents/{{backend}}"
EVALUATION_GROUND_TRUTH_ENDPOINT = f"{API_BASE_URL}/api/v1/evaluation/generate-ground-truth"
EVALUATION_EVALUATE_ENDPOINT = f"{API_BASE_URL}/api/v1/evaluation/evaluate"
EVALUATION_FILES_ENDPOINT = f"{API_BASE_URL}/api/v1/evaluation/ground-truth-files"
EVALUATION_DELETE_FILE_ENDPOINT = f"{API_BASE_URL}/api/v1/evaluation/ground-truth-files/{{filename}}"
EVALUATION_PREVIEW_FILE_ENDPOINT = f"{API_BASE_URL}/api/v1/evaluation/ground-truth-files/{{filename}}/preview"
EVALUATION_RESULTS_ENDPOINT = f"{API_BASE_URL}/api/v1/evaluation/evaluation-results"
EVALUATION_RESULTS_PREVIEW_ENDPOINT = f"{API_BASE_URL}/api/v1/evaluation/evaluation-results/{{filename}}/preview"

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

        # Statistics
        st.subheader("📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", st.session_state.total_queries)
        with col2:
            st.metric("Documents", st.session_state.total_documents)

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
                        "selected_documents": st.session_state.selected_documents if st.session_state.selected_documents else None
                    }
                    response = requests.post(QUERY_ENDPOINT, json=payload)

                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get("answer", "I couldn't generate an answer.")
                        used_chunks = result.get("used_chunks", [])
                        cost_usd = result.get("cost_usd", 0.0)
                        chat_session_id = result.get("chat_session_id")

                        # Update current chat ID if it was a new chat
                        if not st.session_state.current_chat_id and chat_session_id:
                            st.session_state.current_chat_id = chat_session_id
                            # Reload chat sessions to show the new chat
                            load_chat_sessions()

                        # Display answer
                        st.markdown(answer)

                        # Show cost if applicable
                        if cost_usd > 0:
                            st.caption(f"💰 Estimated cost: ${cost_usd:.4f}")

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


def rerank_tab(llm_backend):
    """Rerank interface tab."""
    st.markdown("### 🔄 Document Reranking")
    st.markdown("Enter a query and documents to rerank them by relevance.")

    # Query input
    query = st.text_input(
        "Query",
        placeholder="What is machine learning?",
        help="Enter the query to rank documents against"
    )

    # Documents input
    st.markdown("#### Documents")
    st.caption("Enter documents to rank (one per line)")

    documents_text = st.text_area(
        "Documents (one per line)",
        placeholder="Machine learning is a subset of AI...\nDeep learning uses neural networks...\nPython is a programming language...",
        height=200,
        label_visibility="collapsed"
    )

    # Additional settings
    col1, col2 = st.columns(2)
    with col1:
        rerank_backend = st.selectbox(
            "Reranking Backend",
            options=["openai", "ollama"],
            index=0 if llm_backend == "openai" else 1,
            help="LLM backend to use for reranking"
        )
    with col2:
        top_k = st.number_input(
            "Top K Results",
            min_value=1,
            max_value=50,
            value=5,
            help="Return only top K ranked documents"
        )

    # Rerank button
    if st.button("🔄 Rerank Documents", use_container_width=True, type="primary"):
        if not query:
            st.error("❌ Please enter a query")
        elif not documents_text.strip():
            st.error("❌ Please enter at least one document")
        else:
            # Parse documents
            documents = [doc.strip() for doc in documents_text.split('\n') if doc.strip()]

            if len(documents) == 0:
                st.error("❌ No valid documents found")
            else:
                with st.spinner("Reranking documents..."):
                    try:
                        # Make API request
                        payload = {
                            "query": query,
                            "documents": documents,
                            "llm_backend": rerank_backend,
                            "top_k": top_k
                        }
                        response = requests.post(RERANK_ENDPOINT, json=payload)

                        if response.status_code == 200:
                            result = response.json()
                            ranked_docs = result.get("ranked_documents", [])

                            # Display results
                            st.success(f"✅ Reranked {len(ranked_docs)} documents")

                            st.markdown("### 📊 Results")
                            for idx, doc in enumerate(ranked_docs, 1):
                                with st.container():
                                    col1, col2 = st.columns([0.9, 0.1])
                                    with col1:
                                        st.markdown(f"**#{idx} - Score: {doc['score']:.2f}**")
                                        st.markdown(f"*Original position: #{doc['original_index'] + 1}*")
                                    with col2:
                                        st.metric("", f"{doc['score']:.1f}")

                                    st.text_area(
                                        f"Document {idx}",
                                        value=doc['text'],
                                        height=100,
                                        disabled=True,
                                        label_visibility="collapsed"
                                    )
                                    st.divider()

                        else:
                            st.error(f"❌ API request failed: {response.status_code}")

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")


def evaluation_tab(llm_backend):
    """LLM-based evaluation interface tab."""
    st.markdown("### 📊 LLM-Based Evaluation System")
    st.markdown("Generate ground truth datasets and evaluate your RAG system using LLM-based metrics.")

    # Create sub-tabs for ground truth generation and evaluation
    eval_tab1, eval_tab2 = st.tabs(["🎯 Ground Truth Generation", "📈 RAG Evaluation"])

    with eval_tab1:
        st.markdown("#### Generate Ground Truth Dataset")
        st.markdown("Generate question-answer pairs from your knowledge base for evaluation.")

        # Backend selection
        col1, col2 = st.columns(2)
        with col1:
            gt_backend = st.selectbox(
                "Select Backend",
                options=["openai", "ollama"],
                index=0 if llm_backend == "openai" else 1,
                help="Choose which backend to use for generating ground truth",
                key="gt_backend"
            )
        with col2:
            num_samples = st.number_input(
                "Number of Q&A Pairs",
                min_value=1,
                max_value=50,
                value=10,
                help="Number of question-answer pairs to generate"
            )

        # Document selection for ground truth generation
        st.markdown("##### Document Selection (Optional)")
        try:
            docs_response = requests.get(DOCUMENTS_ENDPOINT.format(backend=gt_backend), timeout=10)
            if docs_response.status_code == 200:
                docs_data = docs_response.json()
                documents = docs_data.get("documents", [])

                if documents:
                    doc_filenames = [doc["filename"] for doc in documents]
                    gt_selected_docs = st.multiselect(
                        "Select specific documents for ground truth generation (leave empty for all)",
                        options=doc_filenames,
                        default=[],
                        help="Filter ground truth generation to specific documents",
                        key="gt_selected_docs"
                    )
                else:
                    st.warning("No documents available in this backend")
                    gt_selected_docs = []
            else:
                st.warning("Unable to load documents")
                gt_selected_docs = []
        except Exception as e:
            st.warning(f"Failed to load documents: {str(e)}")
            gt_selected_docs = []

        st.divider()

        if st.button("🎯 Generate Ground Truth", use_container_width=True, type="primary"):
            with st.spinner(f"Generating {num_samples} ground truth pairs..."):
                try:
                    payload = {
                        "backend": gt_backend,
                        "num_samples": num_samples,
                        "selected_documents": gt_selected_docs if gt_selected_docs else None
                    }
                    response = requests.post(EVALUATION_GROUND_TRUTH_ENDPOINT, json=payload, timeout=600)

                    if response.status_code == 200:
                        result = response.json()

                        if result.get("success"):
                            st.success(f"✅ {result.get('message')}")

                            # Display details
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Q&A Pairs Generated", result.get("num_generated", 0))
                            with col2:
                                st.metric("Backend Used", gt_backend.upper())

                            # Show file path
                            if result.get("file_path"):
                                st.info(f"📁 Saved to: `{result.get('file_path')}`")
                        else:
                            st.error(f"❌ {result.get('message', 'Generation failed')}")
                    else:
                        st.error(f"❌ API request failed: {response.status_code}")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

        st.divider()

        # List available ground truth files
        st.markdown("#### 📂 Available Ground Truth Files")
        try:
            response = requests.get(EVALUATION_FILES_ENDPOINT, timeout=10)
            if response.status_code == 200:
                files = response.json()

                if files:
                    st.caption(f"Found {len(files)} ground truth file(s)")

                    for file in files:
                        with st.expander(f"📄 {file['filename']}", expanded=False):
                            # File metadata
                            col1, col2 = st.columns(2)
                            with col1:
                                st.text(f"Created: {file['created_at']}")
                            with col2:
                                st.text(f"Size: {file['size_kb']:.2f} KB")
                            st.code(file['filepath'], language=None)

                            st.divider()

                            # Action buttons
                            col_view, col_delete = st.columns(2)

                            with col_view:
                                # View/Preview button
                                if st.button("👁️ View Preview", key=f"view_gt_{file['filename']}", use_container_width=True):
                                    try:
                                        preview_url = EVALUATION_PREVIEW_FILE_ENDPOINT.format(filename=file['filename'])
                                        preview_response = requests.get(preview_url, params={"rows": 10}, timeout=10)

                                        if preview_response.status_code == 200:
                                            preview_data = preview_response.json()

                                            st.markdown("##### 📊 Preview")
                                            st.caption(f"Total rows: {preview_data['total_rows']}")

                                            # Display preview as dataframe
                                            import pandas as pd
                                            df_preview = pd.DataFrame(preview_data['preview_data'])
                                            st.dataframe(df_preview, use_container_width=True, height=400)

                                        else:
                                            st.error(f"❌ Failed to load preview: {preview_response.status_code}")
                                    except Exception as e:
                                        st.error(f"❌ Error loading preview: {str(e)}")

                            with col_delete:
                                # Delete button
                                if st.button("🗑️ Delete", key=f"delete_gt_{file['filename']}", use_container_width=True, type="secondary"):
                                    # Confirm deletion
                                    if st.session_state.get(f"confirm_delete_{file['filename']}", False):
                                        try:
                                            delete_url = EVALUATION_DELETE_FILE_ENDPOINT.format(filename=file['filename'])
                                            delete_response = requests.delete(delete_url, timeout=10)

                                            if delete_response.status_code == 200:
                                                st.success(f"✅ Deleted {file['filename']}")
                                                st.rerun()
                                            else:
                                                st.error(f"❌ Failed to delete file: {delete_response.status_code}")
                                        except Exception as e:
                                            st.error(f"❌ Error deleting file: {str(e)}")
                                        # Reset confirmation
                                        st.session_state[f"confirm_delete_{file['filename']}"] = False
                                    else:
                                        # Set confirmation flag
                                        st.session_state[f"confirm_delete_{file['filename']}"] = True
                                        st.warning("⚠️ Click delete again to confirm")
                else:
                    st.info("No ground truth files yet. Generate one above!")
            else:
                st.warning("Unable to load ground truth files")
        except Exception as e:
            st.warning(f"Failed to load files: {str(e)}")

    with eval_tab2:
        st.markdown("#### Evaluate RAG System")
        st.markdown("Run LLM-based evaluation on your RAG system using a ground truth dataset.")

        # Load available ground truth files
        try:
            response = requests.get(EVALUATION_FILES_ENDPOINT, timeout=10)
            gt_files = []
            if response.status_code == 200:
                gt_files = response.json()
        except:
            gt_files = []

        if not gt_files:
            st.warning("⚠️ No ground truth files available. Please generate ground truth first in the previous tab.")
        else:
            # File selection
            selected_file = st.selectbox(
                "Select Ground Truth File",
                options=[f['filename'] for f in gt_files],
                help="Choose a ground truth file to evaluate against"
            )

            # Get file path
            selected_filepath = next(
                (f['filepath'] for f in gt_files if f['filename'] == selected_file),
                None
            )

            # Backend selection
            eval_backend = st.selectbox(
                "Evaluation Backend",
                options=["openai", "ollama"],
                index=0 if llm_backend == "openai" else 1,
                help="Choose which backend to use for evaluation",
                key="eval_backend"
            )

            # Document filter (optional)
            st.markdown("##### Document Filter (Optional)")
            try:
                eval_docs_response = requests.get(DOCUMENTS_ENDPOINT.format(backend=eval_backend), timeout=10)
                if eval_docs_response.status_code == 200:
                    eval_docs_data = eval_docs_response.json()
                    eval_documents = eval_docs_data.get("documents", [])

                    if eval_documents:
                        eval_doc_filenames = [doc["filename"] for doc in eval_documents]
                        filter_docs = st.multiselect(
                            "Select specific documents to evaluate (leave empty for all)",
                            options=eval_doc_filenames,
                            default=[],
                            help="Filter evaluation to specific documents",
                            key="eval_filter_docs"
                        )
                    else:
                        st.info("No documents available in this backend")
                        filter_docs = []
                else:
                    st.warning("Unable to load documents")
                    filter_docs = []
            except Exception as e:
                st.warning(f"Failed to load documents: {str(e)}")
                filter_docs = []

            st.divider()

            if st.button("📈 Run Evaluation", use_container_width=True, type="primary"):
                with st.spinner("Running LLM-based evaluation... This may take several minutes..."):
                    try:
                        payload = {
                            "ground_truth_file": selected_filepath,
                            "backend": eval_backend,
                            "selected_documents": filter_docs if filter_docs else None
                        }
                        response = requests.post(EVALUATION_EVALUATE_ENDPOINT, json=payload, timeout=1800)

                        if response.status_code == 200:
                            result = response.json()

                            if result.get("success"):
                                st.success(f"✅ {result.get('message')}")

                                # Show results file path
                                results_file_path = result.get("results_file_path")
                                if results_file_path:
                                    st.info(f"📊 Results saved to: `{results_file_path}`")

                                summary = result.get("results_summary", {})
                                eval_scores = result.get("evaluation_scores", {})

                                # Display summary metrics
                                st.markdown("### 📊 Evaluation Summary")
                                cols = st.columns(4)
                                with cols[0]:
                                    st.metric(
                                        "Questions Evaluated",
                                        summary.get("questions_evaluated", 0)
                                    )
                                with cols[1]:
                                    st.metric(
                                        "Faithful Responses",
                                        f"{summary.get('faithful_count', 0)} ({summary.get('faithful_percentage', 0.0):.1f}%)"
                                    )
                                with cols[2]:
                                    st.metric(
                                        "Avg Correctness",
                                        f"{summary.get('average_correctness', 0.0):.2f}/10"
                                    )
                                with cols[3]:
                                    st.metric(
                                        "Backend Used",
                                        eval_backend.upper()
                                    )

                                st.divider()

                                # Display correctness distribution
                                aggregate_metrics = eval_scores.get('aggregate_metrics', {})
                                if 'correctness_distribution' in aggregate_metrics:
                                    st.markdown("### 📊 Correctness Score Distribution")
                                    dist = aggregate_metrics['correctness_distribution']

                                    import pandas as pd
                                    dist_df = pd.DataFrame([
                                        {'Score Range': k, 'Count': v}
                                        for k, v in dist.items()
                                    ])

                                    col1, col2 = st.columns([2, 1])
                                    with col1:
                                        st.bar_chart(dist_df.set_index('Score Range'))
                                    with col2:
                                        st.dataframe(dist_df, use_container_width=True, hide_index=True)

                                st.divider()

                                # Per-question results
                                if 'evaluation_results' in eval_scores:
                                    st.markdown("### 🔍 Per-Question Results")

                                    per_question = eval_scores['evaluation_results']
                                    for idx, q_result in enumerate(per_question, 1):
                                        metrics = q_result.get('metrics', {})
                                        faithfulness = metrics.get('faithfulness', False)
                                        correctness = metrics.get('correctness', 0)

                                        # Color-code the expander title based on faithfulness
                                        status_emoji = "✅" if faithfulness else "❌"

                                        with st.expander(f"{status_emoji} Question {idx} (Score: {correctness}/10): {q_result.get('question', '')[:80]}..."):
                                            st.markdown(f"**Question:** {q_result.get('question', 'N/A')}")
                                            st.markdown(f"**Ground Truth Answer:** {q_result.get('ground_truth_answer', 'N/A')}")
                                            st.markdown(f"**AI Response:** {q_result.get('ai_response', 'N/A')}")
                                            st.markdown(f"**Contexts Used:** {q_result.get('contexts_used', 0)}")

                                            st.divider()

                                            # Show evaluation metrics
                                            score_cols = st.columns(2)
                                            with score_cols[0]:
                                                st.metric(
                                                    "Faithfulness",
                                                    "✅ True" if faithfulness else "❌ False"
                                                )
                                            with score_cols[1]:
                                                st.metric(
                                                    "Correctness Score",
                                                    f"{correctness}/10"
                                                )

                                            # Show justification
                                            st.markdown("**Evaluation Justification:**")
                                            st.info(metrics.get('justification', 'No justification provided'))

                            else:
                                st.error(f"❌ {result.get('message', 'Evaluation failed')}")

                        else:
                            st.error(f"❌ API request failed: {response.status_code}")

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        import traceback
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())

        st.divider()

        # View past evaluation results
        st.markdown("#### 📂 View Past Evaluation Results")
        try:
            response = requests.get(EVALUATION_RESULTS_ENDPOINT, timeout=10)
            if response.status_code == 200:
                result_files = response.json()

                if result_files:
                    st.caption(f"Found {len(result_files)} evaluation result file(s)")

                    for file in result_files:
                        with st.expander(f"📊 {file['filename']}", expanded=False):
                            # File metadata
                            col1, col2 = st.columns(2)
                            with col1:
                                st.text(f"Created: {file['created_at']}")
                            with col2:
                                st.text(f"Size: {file['size_kb']:.2f} KB")
                            st.code(file['filepath'], language=None)

                            st.divider()

                            # View button
                            if st.button("👁️ View Results", key=f"view_results_{file['filename']}", use_container_width=True):
                                try:
                                    preview_url = EVALUATION_RESULTS_PREVIEW_ENDPOINT.format(filename=file['filename'])
                                    preview_response = requests.get(preview_url, params={"rows": 100}, timeout=10)

                                    if preview_response.status_code == 200:
                                        preview_data = preview_response.json()

                                        st.markdown("##### 📊 Evaluation Results Preview")
                                        st.caption(f"Total rows: {preview_data['total_rows']}")

                                        # Display preview as dataframe
                                        import pandas as pd
                                        df_preview = pd.DataFrame(preview_data['preview_data'])
                                        st.dataframe(df_preview, use_container_width=True, height=600)

                                        # Download link
                                        st.info(f"💾 Full results available at: `{file['filepath']}`")

                                    else:
                                        st.error(f"❌ Failed to load results: {preview_response.status_code}")
                                except Exception as e:
                                    st.error(f"❌ Error loading results: {str(e)}")
                else:
                    st.info("No evaluation results yet. Run an evaluation above to create results!")
            else:
                st.warning("Unable to load evaluation results")
        except Exception as e:
            st.warning(f"Failed to load evaluation results: {str(e)}")


def main():
    """Main application."""
    # Title with emoji
    st.title("🤖 RAG Assistant")

    # Get settings from sidebar
    llm_backend, top_k = sidebar_settings()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "💬 Chat", "🔄 Rerank", "📊 Evaluation"])

    with tab1:
        upload_tab()

    with tab2:
        st.markdown("### Ask questions about your documents!")
        chat_tab(llm_backend, top_k)

    with tab3:
        rerank_tab(llm_backend)

    with tab4:
        evaluation_tab(llm_backend)


if __name__ == "__main__":
    main()
