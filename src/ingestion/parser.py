"""PDF and TXT file parser — extracts text with page-level metadata."""

from pathlib import Path
from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class ParsedPage:
    """A single page of extracted text with metadata."""
    text: str
    source: str
    page: int


def parse_pdf(file_path: str | Path) -> list[ParsedPage]:
    """Extract text from a PDF file, one entry per page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of ParsedPage objects with text and metadata.
    """
    file_path = Path(file_path)
    pages: list[ParsedPage] = []

    doc = fitz.open(str(file_path))
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        if text:
            pages.append(ParsedPage(
                text=text,
                source=file_path.name,
                page=page_num + 1,  # 1-indexed
            ))
    doc.close()

    return pages


def parse_txt(file_path: str | Path) -> list[ParsedPage]:
    """Extract text from a plain text file.

    Args:
        file_path: Path to the text file.

    Returns:
        List containing a single ParsedPage (whole file = page 1).
    """
    file_path = Path(file_path)
    text = file_path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    return [ParsedPage(text=text, source=file_path.name, page=1)]


def parse_file(file_path: str | Path) -> list[ParsedPage]:
    """Parse a file based on its extension.

    Supports: .pdf, .txt

    Args:
        file_path: Path to the file.

    Returns:
        List of ParsedPage objects.

    Raises:
        ValueError: If the file type is not supported.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(file_path)
    elif suffix == ".txt":
        return parse_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .txt")
