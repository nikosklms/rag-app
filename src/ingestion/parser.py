"""Document parser — extracts text with page-level metadata.

Uses MarkItDown for OCR-enabled PDFs (per-page to save tokens),
Office docs with OCR, and other unsupported file types.
pdfplumber for text-only PDF pages, and native readers for TXT/MD.
"""

import tempfile
from pathlib import Path
from dataclasses import dataclass

import fitz  # PyMuPDF
import markdown
import pdfplumber
from bs4 import BeautifulSoup
from markitdown import MarkItDown
from openai import OpenAI

from src.config import settings




@dataclass
class ParsedPage:
    """A single page of extracted text with metadata."""
    text: str
    source: str
    page: int


# ── Helpers ─────────────────────────────────────────────────────────


def _get_ocr_markitdown() -> MarkItDown:
    """Create a MarkItDown instance with LLM-powered image description."""
    return MarkItDown(
        enable_plugins=True,
        llm_client=OpenAI(api_key=settings.openai_api_key),
        llm_model="gpt-4o-mini",
    )


def _pages_with_images(pdf_path: str | Path) -> set[int]:
    """Return set of 0-indexed page numbers that contain images."""
    doc = fitz.open(str(pdf_path))
    try:
        return {i for i, page in enumerate(doc) if page.get_images()}
    finally:
        doc.close()


def _extract_page_to_pdf(pdf_path: str | Path, page_num: int, output_path: str | Path) -> None:
    """Extract a single page from a PDF into a new PDF file."""
    doc = fitz.open(str(pdf_path))
    try:
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        new_doc.save(str(output_path))
        new_doc.close()
    finally:
        doc.close()


# ── Parsers ─────────────────────────────────────────────────────────


def parse_pdf(file_path: str | Path) -> list[ParsedPage]:
    """Extract text from a PDF file, one entry per page.

    Uses a hybrid approach:
    - Text-only pages → pdfplumber (fast, free)
    - Pages with images → extracted as individual temp PDFs and sent
      through MarkItDown + LLM OCR (only the pages that need it)

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of ParsedPage objects with text and metadata.
    """
    file_path = Path(file_path)
    image_pages = _pages_with_images(file_path)

    if not image_pages:
        print(f"📄 No images in {file_path.name}, using pdfplumber for all pages.")
        pages = []
        with pdfplumber.open(str(file_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages.append(ParsedPage(text=text.strip(), source=file_path.name, page=i + 1))
        return pages

    # Hybrid: pdfplumber for text pages, MarkItDown OCR for image pages
    print(f"🔍 Found images on {len(image_pages)} page(s) of {file_path.name} — using OCR only for those pages.")
    md = _get_ocr_markitdown()
    pages = []

    with pdfplumber.open(str(file_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            if i in image_pages:
                # Extract this single page as a temp PDF → send to OCR
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                    _extract_page_to_pdf(file_path, i, tmp.name)
                    result = md.convert(tmp.name)
                    text = (result.text_content or "").strip()
                    if text:
                        pages.append(ParsedPage(text=text, source=file_path.name, page=i + 1))
                    print(f"  🖼️  Page {i + 1} → OCR (MarkItDown)")
            else:
                text = page.extract_text()
                if text and text.strip():
                    pages.append(ParsedPage(text=text.strip(), source=file_path.name, page=i + 1))
                    print(f"  📄 Page {i + 1} → pdfplumber")

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


def parse_markdown(file_path: str | Path) -> list[ParsedPage]:
    """Parse markdown file."""
    file_path = Path(file_path)
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    html = markdown.markdown(text)
    clean = BeautifulSoup(html, "html.parser").get_text()

    return [ParsedPage(text=clean, source=file_path.name, page=1)]


# ── MarkItDown extensions ──────────────────────────────────────────

# Office formats that benefit from OCR (markitdown-ocr plugin)
OCR_EXTENSIONS = {".docx", ".pptx", ".xlsx"}

# All other formats MarkItDown handles (no OCR needed)
OTHER_MARKITDOWN_EXTENSIONS = {
    ".html", ".htm", ".csv", ".json", ".xml", ".zip", ".epub",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff",
    ".mp3", ".wav", ".m4a",
}

# Combined set for everything MarkItDown can handle
MARKITDOWN_EXTENSIONS = OCR_EXTENSIONS | OTHER_MARKITDOWN_EXTENSIONS

# All supported extensions (native + MarkItDown)
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"} | MARKITDOWN_EXTENSIONS


def parse_office(file_path: str | Path) -> list[ParsedPage]:
    """Parse Office docs (DOCX, PPTX, XLSX) with OCR support via MarkItDown."""
    file_path = Path(file_path)
    print(f"🔍 Using MarkItDown + OCR for {file_path.name}")
    md = _get_ocr_markitdown()
    result = md.convert(str(file_path))
    text = (result.text_content or "").strip()
    if not text:
        return []
    return [ParsedPage(text=text, source=file_path.name, page=1)]


def parse_any(file_path: str | Path) -> list[ParsedPage]:
    """Fallback parser using MarkItDown for other file types."""
    file_path = Path(file_path)
    print(f"📄 Using MarkItDown for {file_path.name}")
    md = MarkItDown(enable_plugins=True)
    result = md.convert(str(file_path))
    text = (result.text_content or "").strip()
    if not text:
        return []
    return [ParsedPage(text=text, source=file_path.name, page=1)]


# ── Router ──────────────────────────────────────────────────────────


def parse_file(file_path: str | Path) -> list[ParsedPage]:
    """Parse a file based on its extension.

    Supports: .pdf, .txt, .md, plus any format handled by MarkItDown
    (.docx, .pptx, .xlsx, .html, .csv, .json, .xml, images, audio, etc.)

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
    elif suffix == ".md":
        return parse_markdown(file_path)
    elif suffix in OCR_EXTENSIONS:
        return parse_office(file_path)
    elif suffix in OTHER_MARKITDOWN_EXTENSIONS:
        return parse_any(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
