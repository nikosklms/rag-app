import uuid
from dataclasses import dataclass, field

@dataclass
class Chunk:
    """A text chunk with metadata about its origin."""
    text: str
    source: str
    page: int
    chunk_index: int
    chunk_type: str = "child"  # 'parent' or 'child'
    parent_id: str | None = None
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))

def chunk_text(
    text: str,
    source: str,
    page: int,
    parent_chunk_size: int = 4096,
    parent_chunk_overlap: int = 0,
    child_chunk_size: int = 512,
    child_chunk_overlap: int = 128,
    start_index: int = 0,
) -> list[Chunk]:
    if not text or not text.strip():
        return []

    text = text.strip()
    chunks: list[Chunk] = []
    
    parent_start = 0
    idx = start_index

    parent_step = max(1, parent_chunk_size - parent_chunk_overlap)

    while parent_start < len(text):
        parent_end = parent_start + parent_chunk_size
        parent_text_slice = text[parent_start:parent_end].strip()

        if parent_text_slice:
            parent_chunk = Chunk(
                text=parent_text_slice,
                source=source,
                page=page,
                chunk_index=idx,
                chunk_type="parent"
            )
            chunks.append(parent_chunk)
            
            # Now, subdivide parent into children
            child_start = 0
            child_step = max(1, child_chunk_size - child_chunk_overlap)
            
            while child_start < len(parent_text_slice):
                child_end = child_start + child_chunk_size
                child_text_slice = parent_text_slice[child_start:child_end].strip()
                
                if child_text_slice:
                    chunks.append(Chunk(
                        text=child_text_slice,
                        source=source,
                        page=page,
                        chunk_index=idx,
                        chunk_type="child",
                        parent_id=parent_chunk.chunk_id
                    ))
                
                child_start += child_step
                if child_start >= len(parent_text_slice):
                    break
            
            print(f"chunk idx={idx} (Parent+Children created)\n")
            idx += 1

        parent_start += parent_step
        if parent_start >= len(text):
            break

    return chunks