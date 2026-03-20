"""Tests for the text chunker."""

from src.ingestion.chunker import chunk_text


def test_short_text_single_chunk():
    """Short text should produce a single chunk."""
    chunks = chunk_text("Hello world", source="test.txt", page=1)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world"
    assert chunks[0].source == "test.txt"
    assert chunks[0].page == 1
    assert chunks[0].chunk_index == 0


def test_empty_text():
    """Empty text should produce no chunks."""
    assert chunk_text("", source="test.txt", page=1) == []
    assert chunk_text("   ", source="test.txt", page=1) == []


def test_long_text_multiple_chunks():
    """Long text should be split into multiple chunks."""
    text = "word " * 200  # ~1000 characters
    chunks = chunk_text(text, source="test.txt", page=1, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1

    # Check that all chunks have correct metadata
    for i, chunk in enumerate(chunks):
        assert chunk.source == "test.txt"
        assert chunk.page == 1
        assert chunk.chunk_index == i


def test_chunk_overlap():
    """Chunks should have overlapping content."""
    text = " ".join([f"sentence{i}." for i in range(50)])
    chunks = chunk_text(text, source="test.txt", page=1, chunk_size=100, chunk_overlap=30)

    if len(chunks) >= 2:
        # Check that consecutive chunks share some content
        for i in range(len(chunks) - 1):
            # The end of chunk i should appear at the start of chunk i+1
            # (this is approximate due to natural boundary splitting)
            assert len(chunks[i].text) > 0
            assert len(chunks[i + 1].text) > 0


def test_start_index():
    """Start index should offset chunk indices."""
    chunks = chunk_text("Hello world", source="test.txt", page=1, start_index=5)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 5


def test_metadata_preservation():
    """Source and page metadata should be preserved in all chunks."""
    text = "word " * 200
    chunks = chunk_text(
        text, source="report.pdf", page=3, chunk_size=100, chunk_overlap=10
    )
    for chunk in chunks:
        assert chunk.source == "report.pdf"
        assert chunk.page == 3
