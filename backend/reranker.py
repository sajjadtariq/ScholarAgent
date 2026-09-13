from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from store import Chunk

RERANK_K = 5
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class PairScorer(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray: ...


class CrossEncoderScorer:
    def __init__(self, model_name: str = RERANK_MODEL) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        scores = self._model.predict(list(pairs))
        return np.asarray(scores, dtype=np.float32)


class CrossEncoderReranker:
    def __init__(self, scorer: PairScorer | None = None) -> None:
        self.scorer = scorer or CrossEncoderScorer()

    def rerank(
        self, question: str, candidates: list[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        scores = self.scorer.predict([(question, chunk.text) for chunk, _ in candidates])
        ranked = sorted(
            zip((chunk for chunk, _ in candidates), scores, strict=True),
            key=lambda item: (-float(item[1]), str(item[0].id)),
        )
        return [(chunk, float(score)) for chunk, score in ranked[:RERANK_K]]
