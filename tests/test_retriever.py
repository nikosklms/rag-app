"""Tests for the retriever module."""

import pytest

from src.ingestion.chunker import Chunk
from src.ingestion.embedder import Embedder
from src.retrieval.retriever import retrieve


@pytest.fixture(autouse=True)
def clean_collection():
    """Use a fresh collection for each test."""
    # Get collection and clear it
    collection = Embedder.get_collection()
    # Delete all items if any
    items = collection.get()
    if items["ids"]:
        collection.delete(ids=items["ids"])
    yield
    # Cleanup after test
    items = collection.get()
    if items["ids"]:
        collection.delete(ids=items["ids"])


def test_retrieve_empty():
    """Retrieval on empty collection returns no results."""
    results = retrieve("What is machine learning?")
    assert results == []


def test_retrieve_returns_results():
    """Retrieval should return relevant chunks after indexing."""
    chunks = [
        Chunk(text="Machine learning is a subset of AI that learns from data.", source="ml.pdf", page=1, chunk_index=0),
        Chunk(text="Python is a programming language used in many domains.", source="python.pdf", page=1, chunk_index=0),
        Chunk(text="Neural networks are inspired by biological neurons.", source="ml.pdf", page=2, chunk_index=1),
    ]
    Embedder.embed_chunks(chunks, document_id="test001")

    results = retrieve("What is machine learning?", top_k=2)
    assert len(results) == 2
    assert results[0].score >= results[1].score  # Sorted by relevance


def test_retrieve_metadata():
    """Retrieved results should have correct metadata."""
    chunks = [
        Chunk(text="The Raft consensus algorithm handles leader election.", source="raft.pdf", page=5, chunk_index=3),
    ]
    Embedder.embed_chunks(chunks, document_id="test002")

    results = retrieve("How does Raft work?", top_k=1)
    assert len(results) == 1
    assert results[0].source == "raft.pdf"
    assert results[0].page == 5
    assert results[0].chunk_index == 3
    assert 0 <= results[0].score <= 1
