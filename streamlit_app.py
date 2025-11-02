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
QUERY_ENDPOINT = f"{API_BASE_URL}/api/v1/query/"
CHAT_SESSIONS_ENDPOINT = f"{API_BASE_URL}/api/v1/chat/sessions"
CHAT_MESSAGES_ENDPOINT = f"{API_BASE_URL}/api/v1/chat/sessions/{{chat_id}}/messages"
CHAT_DELETE_ENDPOINT = f"{API_BASE_URL}/api/v1/chat/sessions/{{chat_id}}"
RERANK_ENDPOINT = f"{API_BASE_URL}/api/v1/rerank/"

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

        # Document upload section
        st.subheader("📄 Upload Documents")
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf", "docx", "txt"],
            help="Upload documents to add to the knowledge base"
        )

        upload_backend = st.selectbox(
            "Upload with backend",
            options=["openai", "ollama"],
            index=0 if settings.embedding_backend == "openai" else 1,
            key="upload_backend"
        )

        if uploaded_file is not None:
            if st.button("📤 Ingest Document", use_container_width=True):
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
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if available
            if message.get("sources"):
                with st.expander("📚 View Sources"):
                    for idx, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {idx}**")
                        st.caption(f"File: {source.get('metadata', {}).get('source_file', 'Unknown')}")
                        st.text(source.get('text', '')[:200] + "...")
                        st.divider()

    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from RAG system
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Make API request
                    payload = {
                        "query": prompt,
                        "top_k": top_k,
                        "llm_backend": llm_backend,
                        "chat_session_id": st.session_state.current_chat_id
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
                                    st.markdown(f"**Source {idx}**")
                                    st.caption(f"File: {chunk.get('metadata', {}).get('source_file', 'Unknown')}")
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


def main():
    """Main application."""
    # Title with emoji
    st.title("🤖 RAG Assistant")

    # Get settings from sidebar
    llm_backend, top_k = sidebar_settings()

    # Create tabs
    tab1, tab2 = st.tabs(["💬 Chat", "🔄 Rerank"])

    with tab1:
        st.markdown("### Ask questions about your documents!")
        chat_tab(llm_backend, top_k)

    with tab2:
        rerank_tab(llm_backend)


if __name__ == "__main__":
    main()
