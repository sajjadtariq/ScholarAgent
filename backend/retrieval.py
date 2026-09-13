import hashlib
import re
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from store import Chunk

BM25_K = 15
DENSE_K = 15
FUSION_K = 20
RRF_CONSTANT = 60
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


class HashEncoder:
    """Deterministic encoder used by tests."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix: np.ndarray = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                index = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest())
                matrix[row, index % self.dimensions] += 1
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]]) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_index in enumerate(ranking, start=1):
            scores[chunk_index] = scores.get(chunk_index, 0.0) + 1 / (RRF_CONSTANT + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:FUSION_K]


class RetrievalIndex:
    def __init__(self, encoder: Encoder | None = None) -> None:
        self.encoder = encoder or SentenceTransformerEncoder()
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self.embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)

    def rebuild(self, chunks: list[Chunk]) -> None:
        if not chunks:
            self.chunks = []
            self.bm25 = None
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            return
        next_chunks = list(chunks)
        next_bm25 = BM25Okapi([tokenize(chunk.text) for chunk in next_chunks])
        next_embeddings = self.encoder.encode([chunk.text for chunk in next_chunks])
        for chunk, embedding in zip(next_chunks, next_embeddings, strict=True):
            chunk.embedding = embedding
        self.chunks = next_chunks
        self.bm25 = next_bm25
        self.embeddings = next_embeddings

    def search(self, question: str) -> list[tuple[Chunk, float]]:
        if not self.chunks or self.bm25 is None:
            return []
        bm25_scores = self.bm25.get_scores(tokenize(question))
        bm25_rank = np.argsort(-bm25_scores, kind="stable")[:BM25_K].tolist()
        query_vector = self.encoder.encode([question])[0]
        dense_scores = self.embeddings @ query_vector
        dense_rank = np.argsort(-dense_scores, kind="stable")[:DENSE_K].tolist()
        return [
            (self.chunks[index], score)
            for index, score in reciprocal_rank_fusion([bm25_rank, dense_rank])
        ]
