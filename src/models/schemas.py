"""Pydantic request/response models for all API endpoints."""

from pydantic import BaseModel, Field


# ── Document endpoints ──────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Response after uploading and indexing a document."""
    document_id: str
    filename: str
    chunks_created: int


class DocumentInfo(BaseModel):
    """Information about an indexed document."""
    document_id: str
    filename: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    """Response for listing all documents."""
    documents: list[DocumentInfo]


class DeleteResponse(BaseModel):
    """Response after deleting a document."""
    document_id: str
    deleted: bool


# ── Query endpoints ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for query endpoints."""
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalResult(BaseModel):
    """A single retrieval result with source info."""
    text: str
    source: str
    page: int
    chunk_index: int
    score: float


class SearchResponse(BaseModel):
    """Response for retrieval-only search."""
    query: str
    results: list[RetrievalResult]


class AskResponse(BaseModel):
    """Response for the full RAG pipeline."""
    query: str
    answer: str
    sources: list[RetrievalResult]
    model: str


# ── Health ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    llm_provider: str
    documents_indexed: int
