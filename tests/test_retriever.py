"""Tests for the retriever module."""

import pytest
import tempfile
import shutil
from unittest.mock import patch

from src.config import settings
from src.ingestion.chunker import Chunk
from src.ingestion.embedder import Embedder
from src.retrieval.retriever import retrieve


@pytest.fixture(autouse=True)
def isolate_chromadb():
    """Use a temporary directory for ChromaDB during tests to protect real data."""
    # Create a temp dir
    temp_dir = tempfile.mkdtemp()
    
    # Store old path
    old_path = settings.chroma_persist_dir
    settings.chroma_persist_dir = temp_dir
    
    # Force Embedder to recreate the client with the new path
    Embedder._client = None
    
    yield
    
    # Restore old path and clean up
    settings.chroma_persist_dir = old_path
    Embedder._client = None
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_retrieve_empty():
    """Retrieval on empty collection returns no results."""
    results = await retrieve("What is machine learning?")
    assert results == []


@pytest.mark.asyncio
@patch("src.retrieval.retriever.rewrite_query", side_effect=lambda q, h: q)
async def test_retrieve_returns_results(mock_rewrite):
    """Retrieval should return relevant chunks after indexing."""
    chunks = [
        Chunk(text="Machine learning is a subset of AI that learns from data.", source="ml.pdf", page=1, chunk_index=0, chunk_type="child", parent_id="p1"),
        Chunk(text="Machine learning is a subset of AI that learns from data. Parent context.", source="ml.pdf", page=1, chunk_index=0, chunk_type="parent", chunk_id="p1"),
        Chunk(text="Python is a programming language used in many domains.", source="python.pdf", page=1, chunk_index=0, chunk_type="child", parent_id="p2"),
        Chunk(text="Python is a programming language.", source="python.pdf", page=1, chunk_index=0, chunk_type="parent", chunk_id="p2"),
    ]
    Embedder.embed_chunks(chunks, document_id="test001")

    results = await retrieve("machine learning", top_k=2)
    assert len(results) >= 1


@pytest.mark.asyncio
@patch("src.retrieval.retriever.rewrite_query", side_effect=lambda q, h: q)
async def test_retrieve_metadata(mock_rewrite):
    """Retrieved results should have correct metadata."""
    chunks = [
        Chunk(text="The Raft consensus algorithm handles leader election.", source="raft.pdf", page=5, chunk_index=3, chunk_type="child", parent_id="p1"),
        Chunk(text="The Raft consensus algorithm handles leader election. Full parent.", source="raft.pdf", page=5, chunk_index=3, chunk_type="parent", chunk_id="p1"),
    ]
    Embedder.embed_chunks(chunks, document_id="test002")

    results = await retrieve("Raft consensus", top_k=1)
    assert len(results) == 1
    assert results[0].source == "raft.pdf"
    assert results[0].page == 5
    assert results[0].chunk_index == 3
    assert 0 <= results[0].score <= 1
