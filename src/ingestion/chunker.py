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
    chunk_overlap: int = 0,
    start_index: int = 0,
) -> list[Chunk]:
    if not text or not text.strip():
        return []

    text = text.strip()
    chunks: list[Chunk] = []
    
    start = 0
    idx = start_index

    # Ensure we don't get stuck in an infinite loop if overlap >= size
    step = max(1, chunk_size - chunk_overlap)

    while start < len(text):
        end = start + chunk_size
        chunk_text_slice = text[start:end].strip()

        if chunk_text_slice:
            chunks.append(Chunk(
                text=chunk_text_slice,
                source=source,
                page=page,
                chunk_index=idx
            ))
            
            print(f"chunk idx={idx}\ntext={chunk_text_slice[:50]}...\n")
            
            idx += 1

        # Increment by the step (size minus overlap)
        start += step
        
        # Break if we've reached the end of the text
        if start >= len(text):
            break

    return chunks