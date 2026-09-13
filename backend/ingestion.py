import re
import unicodedata
from io import BytesIO
from pathlib import PurePath
from uuid import uuid4

from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError

from store import Chunk, Document

CHUNK_WORDS = 500
CHUNK_OVERLAP = 50
MAX_PDF_BYTES = 50 * 1024 * 1024


class PdfIngestionError(ValueError):
    pass


def safe_filename(filename: str | None) -> str:
    name = PurePath((filename or "document.pdf").replace("\\", "/")).name
    name = unicodedata.normalize("NFKC", name)
    name = re.sub(r"[^\w .()\[\]-]", "_", name, flags=re.UNICODE).strip(" .")
    return name[:200] or "document.pdf"


def validate_pdf(data: bytes) -> None:
    if not data:
        raise PdfIngestionError("The uploaded PDF is empty.")
    if len(data) > MAX_PDF_BYTES:
        raise PdfIngestionError("PDFs must be 50 MB or smaller.")
    if data[:1024].find(b"%PDF-") < 0:
        raise PdfIngestionError("The file content is not a PDF.")


def extract_pages(data: bytes) -> list[str]:
    validate_pdf(data)
    try:
        reader = PdfReader(BytesIO(data), strict=False)
        if reader.is_encrypted:
            try:
                unlocked = bool(reader.decrypt(""))
            except (FileNotDecryptedError, PdfReadError):
                unlocked = False
            if not unlocked:
                raise PdfIngestionError("Password-protected PDFs are not supported.")
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except PdfIngestionError:
        raise
    except (PdfReadError, ValueError, OSError) as exc:
        raise PdfIngestionError("The PDF is malformed and could not be read.") from exc
    if not pages:
        raise PdfIngestionError("The PDF contains no pages.")
    if sum(len(page) for page in pages) < 40:
        raise PdfIngestionError("No readable text was found. This PDF may need OCR.")
    return pages


def chunk_document(document: Document, pages: list[str]) -> list[Chunk]:
    if CHUNK_OVERLAP >= CHUNK_WORDS:
        raise ValueError("Chunk overlap must be smaller than chunk size.")
    step = CHUNK_WORDS - CHUNK_OVERLAP
    chunks: list[Chunk] = []
    for page_number, page_text in enumerate(pages, start=1):
        words = page_text.split()
        for start in range(0, len(words), step):
            text = " ".join(words[start : start + CHUNK_WORDS]).strip()
            if text:
                chunks.append(Chunk(uuid4(), document.id, document.name, page_number, text))
            if start + CHUNK_WORDS >= len(words):
                break
    return chunks
