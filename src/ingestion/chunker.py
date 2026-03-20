"""Text chunker — splits text into overlapping chunks for embedding."""

from dataclasses import dataclass


@dataclass
class Chunk:
    """A text chunk with metadata about its origin."""
    text: str
    source: str
    page: int
    chunk_index: int


def chunk_text(
    text: str,
    source: str,
    page: int,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    start_index: int = 0,
) -> list[Chunk]:
    """Split text into overlapping chunks using character-based splitting.

    Uses a recursive approach: tries to split on paragraph breaks first,
    then sentences, then words, falling back to character splitting.

    Args:
        text: The text to split.
        source: Source filename for metadata.
        page: Page number for metadata.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Number of overlapping characters between chunks.
        start_index: Starting chunk index (for numbering across pages).

    Returns:
        List of Chunk objects.
    """
    if not text or not text.strip():
        return []

    # Clean up whitespace
    text = text.strip()

    # If text fits in one chunk, return it directly
    if len(text) <= chunk_size:
        return [Chunk(text=text, source=source, page=page, chunk_index=start_index)]

    chunks: list[Chunk] = []
    start = 0
    idx = start_index

    while start < len(text):
        # Determine end position
        end = min(start + chunk_size, len(text))

        # If not at the end, try to break at a natural boundary
        if end < len(text):
            # Try paragraph break
            break_pos = text.rfind("\n\n", start, end)
            if break_pos == -1 or break_pos <= start:
                # Try sentence break
                for sep in (". ", "! ", "? ", "\n"):
                    break_pos = text.rfind(sep, start, end)
                    if break_pos > start:
                        end = break_pos + len(sep)
                        break
                else:
                    # Try word break
                    break_pos = text.rfind(" ", start, end)
                    if break_pos > start:
                        end = break_pos + 1
            else:
                end = break_pos + 2  # Include the double newline

        chunk_text_slice = text[start:end].strip()
        if chunk_text_slice:
            chunks.append(Chunk(
                text=chunk_text_slice,
                source=source,
                page=page,
                chunk_index=idx,
            ))
            idx += 1

        # Move start forward with overlap
        start = max(start + 1, end - chunk_overlap)

    return chunks
