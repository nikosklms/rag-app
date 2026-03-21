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
    if not text or not text.strip():
        return []

    text = text.strip()

    if len(text) <= chunk_size:
        return [Chunk(text=text, source=source, page=page, chunk_index=start_index)]

    chunks: list[Chunk] = []
    start = 0
    idx = start_index

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            break_pos = text.rfind("\n\n", start, end)
            if break_pos > start:
                end = break_pos + 2
            else:
                best_break_pos = -1
                best_sep_len = 0
                for sep in (". ", "! ", "? ", "\n"):
                    pos = text.rfind(sep, start, end)
                    if pos > best_break_pos:
                        best_break_pos = pos
                        best_sep_len = len(sep)
                
                if best_break_pos > start:
                    end = best_break_pos + best_sep_len
                else:
                    break_pos = text.rfind(" ", start, end)
                    if break_pos > start:
                        end = break_pos + 1

        chunk_text_slice = text[start:end].strip()
        if chunk_text_slice:
            chunks.append(Chunk(
                text=chunk_text_slice,
                source=source,
                page=page,
                chunk_index=idx,
            ))
            idx += 1

        if end == len(text):
            break

        raw_start = max(start + 1, end - chunk_overlap)
        
        if 0 < raw_start < len(text) and text[raw_start - 1] not in (" ", "\n"):
            next_space = text.find(" ", raw_start, end)
            if next_space != -1:
                start = next_space + 1
            else:
                start = raw_start
        else:
            start = raw_start

    return chunks