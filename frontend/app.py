"""Streamlit frontend — Chat with your Documents UI."""

import os
import json
import streamlit as st
import httpx
from dotenv import load_dotenv

load_dotenv()

# Backend API URL — configurable via env for Docker
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Chat with your Documents",
    page_icon="📚",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .source-tag {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #667eea44;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.8rem;
        margin: 2px;
        display: inline-block;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #ffffff11;
    }
    .stChatMessage {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ────────────────────────────────────────────────

def check_backend() -> bool:
    """Check if the backend API is reachable."""
    try:
        r = httpx.get(f"{API_URL}/health", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


# Messages that should not be sent as conversation history
_SKIP_HISTORY_PREFIXES = (
    "No documents have been indexed yet",
    "⚠️ Backend is not connected",
    "❌ Failed to get a response",
    "I couldn't find the answer to that",
)


def _build_history() -> list[dict]:
    """Build filtered history excluding error/system messages."""
    history = []
    for m in st.session_state.messages:
        content = m["content"]
        if m["role"] == "assistant" and content.startswith(_SKIP_HISTORY_PREFIXES):
            continue
        history.append({"role": m["role"], "content": content})
    return history


def upload_file(file) -> dict | None:
    """Upload a file to the backend."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        r = httpx.post(f"{API_URL}/documents/upload", files=files, timeout=120.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None


def list_documents() -> list:
    """Fetch the list of indexed documents."""
    try:
        r = httpx.get(f"{API_URL}/documents/", timeout=10.0)
        r.raise_for_status()
        return r.json().get("documents", [])
    except Exception:
        return []


def delete_document(doc_id: str) -> bool:
    """Delete a document from the index."""
    try:
        r = httpx.delete(f"{API_URL}/documents/{doc_id}", timeout=10.0)
        return r.status_code == 200
    except Exception:
        return False


def list_chats() -> list:
    """Fetch the list of saved chats."""
    try:
        r = httpx.get(f"{API_URL}/chats", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def load_chat(chat_id: str) -> dict | None:
    """Fetch a specific chat history."""
    try:
        r = httpx.get(f"{API_URL}/chats/{chat_id}", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def delete_chat(chat_id: str) -> bool:
    """Delete a specific chat history."""
    try:
        r = httpx.delete(f"{API_URL}/chats/{chat_id}", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


def ask_question(query: str, top_k: int = 5) -> dict | None:
    """Send a question through the full RAG pipeline."""
    try:
        payload = {
            "query": query, 
            "top_k": top_k,
            "history": _build_history()
        }
        if st.session_state.get("chat_id"):
            payload["chat_id"] = st.session_state.chat_id
            
        r = httpx.post(
            f"{API_URL}/query/ask",
            json=payload,
            timeout=120.0,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Query failed: {e}")
        return None


def ask_question_stream(query: str, top_k: int = 5) -> dict | None:
    """Send a question and stream the answer tokens as SSE."""
    payload = {
        "query": query,
        "top_k": top_k,
        "history": _build_history(),
    }
    if st.session_state.get("chat_id"):
        payload["chat_id"] = st.session_state.chat_id

    answer_placeholder = st.empty()
    answer_text = ""
    event_name = "message"
    meta: dict = {}

    try:
        with httpx.stream(
            "POST",
            f"{API_URL}/query/ask/stream",
            json=payload,
            timeout=120.0,
        ) as response:
            response.raise_for_status()

            for raw_line in response.iter_lines():
                if raw_line is None:
                    continue
                line = raw_line.decode() if isinstance(raw_line, bytes) else raw_line

                if line.startswith("event: "):
                    event_name = line[7:].strip()
                    continue
                if not line.startswith("data: "):
                    continue

                data_str = line[6:].strip()
                data = {}
                if data_str:
                    try:
                        data = json.loads(data_str)
                    except Exception:
                        data = {}

                if event_name == "token":
                    token = data.get("text", "")
                    if token:
                        answer_text += token
                        answer_placeholder.markdown(answer_text + "▌")
                elif event_name == "meta":
                    meta = data
                elif event_name == "error":
                    detail = data.get("detail", "Unknown streaming error")
                    st.error(detail)
                    return None
                elif event_name == "done":
                    break
    except Exception as e:
        st.error(f"Query failed: {e}")
        return None

    answer_placeholder.markdown(answer_text if answer_text else " ")

    return {
        "query": query,
        "answer": meta.get("answer", answer_text),
        "sources": meta.get("sources", []),
        "model": meta.get("model", ""),
        "chat_id": meta.get("chat_id"),
        "chat_title": meta.get("chat_title"),
    }

def get_doc_info(doc_id: str) -> dict | None:
    try:
        r = httpx.get(f"{API_URL}/documents/{doc_id}/info", timeout=10.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

@st.dialog("Document Info")
def show_doc_info(doc_id: str):
    info = get_doc_info(doc_id)
    if info:
        st.metric("Chunks", info['chunk_count'])
        st.metric("Avg chars/chunk", info['avg_chars_per_chunk'])
        st.metric("Pages", info['pages_count'])
    else:
        st.error("Could not fetch info")

@st.dialog("Delete Document")
def confirm_delete(doc_id: str, filename: str):
    st.warning(f"Are you sure you want to delete **{filename}**?")
    _, col1, col2, _ = st.columns([1.5, 1, 1, 1.5])
    with col1:
        if st.button("Yes", type="primary", key="yes_del_doc", use_container_width=True):
            delete_document(doc_id)
            st.rerun()
    with col2:
        if st.button("No", key="no_del_doc", use_container_width=True):
            st.rerun()

@st.dialog("Delete Chat")
def confirm_delete_chat(chat_id: str, title: str):
    st.warning(f"Are you sure you want to delete chat: **{title}**?")
    _, col1, col2, _ = st.columns([1.5, 1, 1, 1.5])
    with col1:
        if st.button("Yes", type="primary", key="yes_del_chat", use_container_width=True):
            delete_chat(chat_id)
            if st.session_state.chat_id == chat_id:
                st.session_state.chat_id = None
                st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("No", key="no_del_chat", use_container_width=True):
            st.rerun()

# ── Session State ───────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None


# ── Sidebar ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📂 Document Management")

    # Connection status
    backend_ok = check_backend()
    if backend_ok:
        st.success("✅ Backend connected")
    else:
        st.error("❌ Backend not reachable")
        st.caption(f"Make sure the API is running at `{API_URL}`")

    st.divider()

    # Chat History
    st.markdown("### Chat History")
    
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.chat_id = None
        st.rerun()
        
    if backend_ok:
        chats = list_chats()
        if chats:
            st.markdown("<br>", unsafe_allow_html=True)
            for chat in chats:
                col1, col2 = st.columns([5, 1])
                with col1:
                    if st.button(f"{chat['title']}", key=f"chat_{chat['chat_id']}", use_container_width=True):
                        loaded_chat = load_chat(chat['chat_id'])
                        if loaded_chat:
                            st.session_state.chat_id = chat['chat_id']
                            # Convert dict to session messages
                            st.session_state.messages = []
                            for msg in loaded_chat['messages']:
                                st.session_state.messages.append({
                                    "role": msg['role'],
                                    "content": msg['content'],
                                    "sources": [] # We don't save sources to history JSON currently, though we could
                                })
                            st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_chat_{chat['chat_id']}"):
                        confirm_delete_chat(chat['chat_id'], chat['title'])
        else:
            st.caption("No chat history found.")

    st.divider()

    # File upload
    st.markdown("### Upload Documents")
    uploaded_file = st.file_uploader(
        "Choose a PDF, TXT, MD file",
        type=["pdf", "txt", "md"],
        help="Upload a document to index it for Q&A",
    )

    if uploaded_file and st.button("📤 Upload & Index", use_container_width=True):
        with st.spinner("Parsing and indexing document..."):
            result = upload_file(uploaded_file)
            if result:
                st.success(
                    f"✅ Indexed **{result['filename']}** — "
                    f"{result['chunks_created']} chunks created"
                )
                st.balloons()

    st.divider()

    # Document list
    st.markdown("### Indexed Documents")
    if backend_ok:
        docs = list_documents()
        if docs:
            for doc in docs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(
                        f"📄 **{doc['filename']}**  \n"
                        f"`{doc['chunk_count']} chunks`"
                    )
                with col2:
                    if st.button("🗑️", key=f"del_{doc['document_id']}"):
                        confirm_delete(doc["document_id"], doc["filename"])
                    if st.button("ℹ️", key=f"get_{doc['document_id']}_info"):
                        show_doc_info(doc['document_id'])

        else:
            st.caption("No documents indexed yet. Upload one above!")

    st.divider()

    # Settings
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Results to retrieve (top-K)", 1, 20, int(os.getenv("TOP_K", "5")))


# ── Main Chat Area ──────────────────────────────────────────────────

st.markdown(
    '<p class="main-header">Chat with your documents using AI-powered retrieval</p>',
    unsafe_allow_html=True,
)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for src in msg["sources"]:
                    st.markdown(
                        f'<span class="source-tag">'
                        f'📄 {src["source"]} — Page {src["page"]} '
                        f'(score: {src["score"]:.2f})'
                        f'</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(src["text"][:200] + "..." if len(src["text"]) > 200 else src["text"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if not backend_ok:
            response_text = "⚠️ Backend is not connected. Please start the API server."
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        else:
            result = ask_question_stream(prompt, top_k=top_k)

            if result:
                sources = result.get("sources", [])
                if sources:
                    with st.expander("📎 Sources"):
                        for src in sources:
                            st.markdown(
                                f'<span class="source-tag">'
                                f'📄 {src["source"]} — Page {src["page"]} '
                                f'(score: {src["score"]:.2f})'
                                f'</span>',
                                unsafe_allow_html=True,
                            )
                            st.caption(
                                src["text"][:200] + "..."
                                if len(src["text"]) > 200
                                else src["text"]
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"] if result else "Error generating response.",
                    "sources": result.get("sources", []) if result else [],
                })
                
                # Update chat ID if it's a new chat
                if result and result.get("chat_id") and not st.session_state.chat_id:
                    st.session_state.chat_id = result["chat_id"]
                    st.rerun() # Refresh to show in sidebar
            else:
                error_msg = "❌ Failed to get a response. Please try again."
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
