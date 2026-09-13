from dataclasses import dataclass, field
from threading import RLock
from uuid import UUID, uuid4

import numpy as np


@dataclass
class Chunk:
    id: UUID
    document_id: UUID
    document_name: str
    page_number: int
    text: str
    embedding: np.ndarray | None = None


@dataclass
class Document:
    id: UUID
    name: str
    size_bytes: int
    page_count: int
    pdf_bytes: bytes = field(repr=False)
    chunks: list[Chunk] = field(default_factory=list, repr=False)


class MemoryStore:
    def __init__(self) -> None:
        self._documents: dict[UUID, Document] = {}
        self._lock = RLock()

    def add(self, name: str, size_bytes: int, page_count: int, pdf_bytes: bytes) -> Document:
        document = Document(uuid4(), name, size_bytes, page_count, pdf_bytes)
        with self._lock:
            self._documents[document.id] = document
        return document

    def list_documents(self) -> list[Document]:
        with self._lock:
            return list(self._documents.values())

    def get(self, document_id: UUID) -> Document | None:
        with self._lock:
            return self._documents.get(document_id)

    def remove(self, document_id: UUID) -> None:
        with self._lock:
            self._documents.pop(document_id, None)

    def all_chunks(self) -> list[Chunk]:
        with self._lock:
            return [chunk for document in self._documents.values() for chunk in document.chunks]

    def clear(self) -> None:
        with self._lock:
            self._documents.clear()


store = MemoryStore()
