from uuid import uuid4

import pytest

from ingestion import CHUNK_OVERLAP, CHUNK_WORDS, PdfIngestionError, chunk_document, extract_pages
from store import Document
from tests.fixtures import pdf_bytes


def test_extracts_text_with_real_page_numbers() -> None:
    pages = [
        "Page one describes retrieval and citations in enough detail for extraction.",
        "Page two explains reranking and page navigation in enough detail for extraction.",
    ]
    assert extract_pages(pdf_bytes(pages)) == pages


@pytest.mark.parametrize("data", [b"", b"not a pdf", b"%PDF-1.4 broken"])
def test_rejects_empty_non_pdf_and_malformed_files(data: bytes) -> None:
    with pytest.raises(PdfIngestionError):
        extract_pages(data)


def test_rejects_image_only_pdf() -> None:
    with pytest.raises(PdfIngestionError, match="OCR"):
        extract_pages(pdf_bytes([""]))


def test_chunks_each_page_with_overlap() -> None:
    words = [f"word{index}" for index in range(700)]
    document = Document(uuid4(), "paper.pdf", 10, 1, b"pdf")
    chunks = chunk_document(document, [" ".join(words)])
    assert [len(chunk.text.split()) for chunk in chunks] == [CHUNK_WORDS, 250]
    assert chunks[0].text.split()[-CHUNK_OVERLAP:] == chunks[1].text.split()[:CHUNK_OVERLAP]
    assert all(chunk.page_number == 1 for chunk in chunks)
