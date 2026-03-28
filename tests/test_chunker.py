"""Tests for the text chunker."""

from src.ingestion.chunker import chunk_text


def test_short_text_single_chunk():
    """Short text should produce one parent and one child."""
    chunks = chunk_text("Hello world", source="test.txt", page=1, parent_chunk_size=100, child_chunk_size=50, child_chunk_overlap=0)
    assert len(chunks) == 2
    assert chunks[0].chunk_type == "parent"
    assert chunks[1].chunk_type == "child"
    assert chunks[1].parent_id == chunks[0].chunk_id
    assert chunks[0].text == "Hello world"
    assert chunks[1].text == "Hello world"
    assert chunks[0].source == "test.txt"
    assert chunks[0].page == 1
    assert chunks[0].chunk_index == 0


def test_empty_text():
    """Empty text should produce no chunks."""
    assert chunk_text("", source="test.txt", page=1) == []
    assert chunk_text("   ", source="test.txt", page=1) == []


def test_long_text_multiple_chunks():
    """Long text should be split into multiple parent and child chunks."""
    text = "word " * 200  # ~1000 characters
    chunks = chunk_text(text, source="test.txt", page=1, parent_chunk_size=400, child_chunk_size=100, child_chunk_overlap=0)
    assert len(chunks) > 3

    parents = [c for c in chunks if c.chunk_type == "parent"]
    children = [c for c in chunks if c.chunk_type == "child"]

    # Parents are generated correctly
    assert len(parents) > 1
    assert len(children) > len(parents)

    # Check that all chunks have correct metadata
    for i, chunk in enumerate(parents):
        assert chunk.source == "test.txt"
        assert chunk.page == 1


def test_chunk_overlap():
    """Chunks should have overlapping content."""
    text = " ".join([f"sentence{i}." for i in range(50)])
    chunks = chunk_text(text, source="test.txt", page=1, parent_chunk_size=300, child_chunk_size=100, child_chunk_overlap=30)

    children = [c for c in chunks if c.chunk_type == "child"]
    if len(children) >= 2:
        for i in range(len(children) - 1):
            assert len(children[i].text) > 0
            assert len(children[i + 1].text) > 0


def test_start_index():
    """Start index should offset chunk indices."""
    chunks = chunk_text("Hello world", source="test.txt", page=1, start_index=5)
    assert chunks[0].chunk_index == 5
    assert chunks[1].chunk_index == 5


def test_metadata_preservation():
    """Source and page metadata should be preserved in all chunks."""
    text = "word " * 200
    chunks = chunk_text(
        text, source="report.pdf", page=3
    )
    for chunk in chunks:
        assert chunk.source == "report.pdf"
        assert chunk.page == 3
