import hashlib
from io import BytesIO
from uuid import uuid4

from pypdf import PdfReader

from app.core.config import get_settings
from app.schemas.state import ChunkRecord


class PdfValidationError(ValueError):
    pass


def validate_pdf_upload(filename: str, content_type: str, size_bytes: int) -> None:
    settings = get_settings()
    if size_bytes > settings.max_upload_bytes:
        raise PdfValidationError("PDF exceeds the 25MB upload limit")
    if not filename.lower().endswith(".pdf") or content_type not in {"application/pdf", "application/octet-stream"}:
        raise PdfValidationError("Only PDF files are supported")


def extract_pdf_pages(data: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(BytesIO(data))
    if reader.is_encrypted:
        raise PdfValidationError("Encrypted PDFs are not supported")
    pages = [(index + 1, (page.extract_text() or "").strip()) for index, page in enumerate(reader.pages)]
    pages = [(page_number, text) for page_number, text in pages if text]
    if not pages:
        raise PdfValidationError("PDF has no extractable text")
    return pages


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _window_words(words: list[str], size: int, overlap: int) -> list[list[str]]:
    windows: list[list[str]] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(words):
        window = words[start : start + size]
        if window:
            windows.append(window)
        start += step
    return windows


def chunk_pages(document_id: str, pages: list[tuple[int, str]]) -> list[ChunkRecord]:
    settings = get_settings()
    chunks: list[ChunkRecord] = []
    current_words: list[str] = []
    page_start = pages[0][0]
    page_end = pages[0][0]

    def flush_parent(words: list[str], start_page: int, end_page: int) -> None:
        if not words:
            return
        parent_id = str(uuid4())
        for child_index, child_words in enumerate(
            _window_words(words, settings.child_chunk_tokens, settings.child_overlap_tokens)
        ):
            text = " ".join(child_words)
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            chunks.append(
                ChunkRecord(
                    id=str(uuid4()),
                    document_id=document_id,
                    parent_id=parent_id,
                    page_start=start_page,
                    page_end=end_page,
                    text=text,
                    text_hash=text_hash,
                    token_count=estimate_tokens(text),
                    vector_id=f"{document_id}:{parent_id}:{child_index}",
                )
            )

    for page_number, text in pages:
        words = text.split()
        if current_words and len(current_words) + len(words) > settings.parent_chunk_tokens:
            flush_parent(current_words, page_start, page_end)
            current_words = []
            page_start = page_number
        current_words.extend(words)
        page_end = page_number
    flush_parent(current_words, page_start, page_end)
    return chunks
