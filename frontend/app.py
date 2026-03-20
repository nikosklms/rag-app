"""Streamlit frontend — Chat with your Documents UI."""

import streamlit as st
import httpx

# Backend API URL
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="📚 DocMind — Chat with your Documents",
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


def ask_question(query: str, top_k: int = 5) -> dict | None:
    """Send a question through the full RAG pipeline."""
    try:
        r = httpx.post(
            f"{API_URL}/query/ask",
            json={"query": query, "top_k": top_k},
            timeout=120.0,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Query failed: {e}")
        return None


# ── Session State ───────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []


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

    # File upload
    st.markdown("### Upload Documents")
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
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
                        if delete_document(doc["document_id"]):
                            st.rerun()
        else:
            st.caption("No documents indexed yet. Upload one above!")

    st.divider()

    # Settings
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Results to retrieve (top-K)", 1, 20, 5)

    # Clear chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Chat Area ──────────────────────────────────────────────────

st.markdown('<p class="main-header">📚 DocMind</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Chat with your documents using AI-powered retrieval</p>',
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
            with st.spinner("Thinking..."):
                result = ask_question(prompt, top_k=top_k)

            if result:
                st.markdown(result["answer"])

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
            else:
                error_msg = "❌ Failed to get a response. Please try again."
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
