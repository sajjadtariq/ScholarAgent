from pathlib import Path
from typing import Annotated
from uuid import UUID

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from generation import generate_answer
from ingestion import MAX_PDF_BYTES, PdfIngestionError, chunk_document, extract_pages, safe_filename
from reranker import CrossEncoderReranker
from retrieval import RetrievalIndex
from store import Document, store

load_dotenv(Path(__file__).with_name(".env"))


class DocumentOut(BaseModel):
    id: UUID
    name: str
    size_bytes: int
    page_count: int
    chunk_count: int


class ChatIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class CitationOut(BaseModel):
    number: int
    chunk_id: UUID
    document_id: UUID
    document_name: str
    page_number: int
    text: str
    reranker_score: float


class ChatOut(BaseModel):
    answer: str
    citations: list[CitationOut]


retrieval_index = RetrievalIndex()
reranker = CrossEncoderReranker()

app = FastAPI(title="PDF Research Chat API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def document_out(document: Document) -> DocumentOut:
    return DocumentOut(
        id=document.id,
        name=document.name,
        size_bytes=document.size_bytes,
        page_count=document.page_count,
        chunk_count=len(document.chunks),
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents/upload", response_model=DocumentOut, status_code=201)
async def upload_document(file: Annotated[UploadFile, File()]) -> DocumentOut:
    data = await file.read(MAX_PDF_BYTES + 1)
    await file.close()
    try:
        pages = extract_pages(data)
    except PdfIngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    document = store.add(safe_filename(file.filename), len(data), len(pages), data)
    document.chunks = chunk_document(document, pages)
    try:
        await run_in_threadpool(retrieval_index.rebuild, store.all_chunks())
    except Exception as exc:
        store.remove(document.id)
        raise HTTPException(
            status_code=503, detail=f"Embedding model could not be loaded: {exc}"
        ) from exc
    return document_out(document)


@app.get("/documents", response_model=list[DocumentOut])
async def list_documents() -> list[DocumentOut]:
    return [document_out(document) for document in store.list_documents()]


@app.get("/documents/{document_id}/file")
async def document_file(document_id: UUID) -> Response:
    document = store.get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="PDF not found.")
    return Response(
        document.pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{document.name}"'},
    )


@app.post("/chat", response_model=ChatOut)
async def chat(request: ChatIn) -> ChatOut:
    if not store.list_documents():
        raise HTTPException(status_code=409, detail="Upload at least one PDF first.")
    try:
        candidates = retrieval_index.search(request.question)
        reranked = reranker.rerank(request.question, candidates)
        answer, citations = await generate_answer(request.question, reranked)
    except (RuntimeError, OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ChatOut(
        answer=answer, citations=[CitationOut.model_validate(item) for item in citations]
    )
