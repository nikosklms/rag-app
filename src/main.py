"""FastAPI application — RAG API with document management and querying."""

import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.ingestion.parser import parse_file
from src.ingestion.chunker import chunk_text, Chunk
from src.ingestion.embedder import Embedder
from src.retrieval.retriever import retrieve
from src.generation.generator import generate, generate_chat_title
import src.history_manager as history_manager
from src.models.schemas import (
    UploadResponse,
    DocumentListResponse,
    DocumentInfo,
    DocumentInfoResponse,
    DeleteResponse,
    QueryRequest,
    SearchResponse,
    AskResponse,
    HealthResponse,
    ChatSummary,
    ChatHistoryResponse,
    ChatMessage
)

app = FastAPI(
    title="RAG API",
    description="Chat with your Documents — Retrieval-Augmented Generation API",
    version="1.0.0",
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def sync_uploads():
    """Ensure all valid files in the upload directory are indexed in ChromaDB.
    
    This function discovers files dropped manually in the uploads folder and
    reindexes files that exist on disk but lost their database entries.
    """
    settings.ensure_dirs()
    upload_dir = Path(settings.upload_dir)
    
    # Get all currently indexed document IDs
    try:
        indexed_docs = {d["document_id"] for d in Embedder.list_documents()}
    except Exception as e:
        print(f"Warning: Could not connect to DB for sync: {e}")
        return
        
    for file_path in upload_dir.glob("*"):
        if not file_path.is_file():
            continue
            
        suffix = file_path.suffix.lower()
        if suffix not in (".pdf", ".txt", ".md"):
            continue
            
        filename_parts = file_path.name.split("_", 1)
        
        document_id = None
        original_filename = file_path.name
        
        # Check if the file already has an 8-char ID prefix
        if len(filename_parts) == 2 and len(filename_parts[0]) == 8:
            document_id = filename_parts[0]
            original_filename = filename_parts[1]
            
        # If the document_id is valid and already indexed, skip it
        if document_id and document_id in indexed_docs:
            continue
            
        # File is either entirely new or its ID is not in DB.
        if not document_id:
            # It's a completely new dropped file. Needs an ID prefix.
            document_id = Embedder.generate_document_id()
            new_path = file_path.parent / f"{document_id}_{original_filename}"
            try:
                file_path.rename(new_path)
                file_path = new_path
            except Exception as e:
                print(f"Failed to rename {file_path.name}: {e}")
                continue
                
        print(f"🔄 Auto-sync: Indexing {original_filename} (ID: {document_id})...")
        
        try:
            pages = parse_file(file_path)
            all_chunks = []
            chunk_idx = 0
            for page in pages:
                page_chunks = chunk_text(
                    text=page.text,
                    source=original_filename,
                    page=page.page,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                    start_index=chunk_idx,
                )
                all_chunks.extend(page_chunks)
                chunk_idx += len(page_chunks)
            
            Embedder.embed_chunks(all_chunks, document_id)
            print(f"✅ Auto-sync: Successfully indexed {original_filename}!")
        except Exception as e:
            print(f"❌ Auto-sync: Failed to index {original_filename}: {e}")


@app.on_event("startup")
async def startup():
    """Ensure directories exist on startup and sync orphaned uploads."""
    settings.ensure_dirs()
    await sync_uploads()


# ── Health ──────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        llm_provider=settings.llm_provider,
        documents_indexed=Embedder.get_document_count(),
    )


# ── Document Management ────────────────────────────────────────────


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a PDF or TXT document.

    The file is parsed, chunked, embedded, and stored in ChromaDB.
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".txt", ".md"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: .pdf, .txt, .md",
        )

    # Save uploaded file
    settings.ensure_dirs()
    document_id = Embedder.generate_document_id()
    file_path = Path(settings.upload_dir) / f"{document_id}_{file.filename}"

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Parse → Chunk → Embed
    try:
        pages = parse_file(file_path)

        all_chunks: list[Chunk] = []
        chunk_idx = 0
        for page in pages:
            page_chunks = chunk_text(
                text=page.text,
                source=page.source,
                page=page.page,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                start_index=chunk_idx,
            )
            all_chunks.extend(page_chunks)
            chunk_idx += len(page_chunks)

        chunks_stored = Embedder.embed_chunks(all_chunks, document_id)
    except Exception as e:
        # Clean up file on failure
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        chunks_created=chunks_stored,
    )


@app.get("/documents/", response_model=DocumentListResponse)
async def list_documents():
    """List all indexed documents."""
    docs = Embedder.list_documents()
    return DocumentListResponse(
        documents=[
            DocumentInfo(
                document_id=d["document_id"],
                filename=d["filename"],
                chunk_count=d["chunk_count"],
            )
            for d in docs
        ]
    )

@app.get("/documents/{document_id}/info", response_model=DocumentInfoResponse)
async def get_document_info(document_id: str):
    """Get info for a single document"""

    try:
        doc = Embedder.get_document_info(document_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentInfoResponse(**doc) # ** -> unpack


@app.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str):
    """Delete a document and all its indexed chunks."""
    # Check if document exists
    docs = Embedder.list_documents()
    doc_exists = any(d["document_id"] == document_id for d in docs)

    if not doc_exists:
        raise HTTPException(status_code=404, detail="Document not found")

    Embedder.delete_document(document_id)

    # Also delete the uploaded file if it exists
    upload_dir = Path(settings.upload_dir)
    for f in upload_dir.glob(f"{document_id}_*"):
        f.unlink(missing_ok=True)

    return DeleteResponse(document_id=document_id, deleted=True)


# ── Chat History ────────────────────────────────────────────────────

@app.get("/chats", response_model=list[ChatSummary])
async def list_chats():
    """List all saved chat sessions."""
    return history_manager.list_chats()

@app.get("/chats/{chat_id}", response_model=ChatHistoryResponse)
async def get_chat_history(chat_id: str):
    """Get the full history of a specific chat session."""
    chat = history_manager.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat

@app.delete("/chats/{chat_id}")
async def delete_chat_history(chat_id: str):
    """Delete a chat session."""
    if not history_manager.delete_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"message": "Chat deleted"}


# ── Query ───────────────────────────────────────────────────────────


@app.post("/query/search", response_model=SearchResponse)
async def search_query(request: QueryRequest):
    """Retrieval-only search — returns relevant chunks without LLM generation."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    results = retrieve(request.query, top_k=request.top_k)

    return SearchResponse(query=request.query, results=results)


@app.post("/query/ask", response_model=AskResponse)
async def ask_query(request: QueryRequest):
    """Full RAG pipeline — retrieves context and generates an answer."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Retrieve relevant chunks
    results = retrieve(request.query, top_k=request.top_k)

    if not results:
        return AskResponse(
            query=request.query,
            answer="No documents have been indexed yet. Please upload some documents first.",
            sources=[],
            model=settings.llm_provider,
        )

    # Generate answer
    try:
        answer, model = generate(request.query, results, request.history)
        
        chat_id = request.chat_id
        chat_title = None
        
        # Save to history manager
        if not chat_id:
            # First turn: create chat and title
            chat_title = generate_chat_title(request.query, answer)
            chat_id = history_manager.create_chat(chat_title)
            
        history_manager.append_messages(chat_id, [
            ChatMessage(role="user", content=request.query),
            ChatMessage(role="assistant", content=answer)
        ])
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM generation failed: {e}. Make sure your LLM provider ({settings.llm_provider}) is running.",
        )

    return AskResponse(
        query=request.query,
        answer=answer,
        sources=results,
        model=model,
        chat_id=chat_id,
        chat_title=chat_title
    )
