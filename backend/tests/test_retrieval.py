from uuid import uuid4

import numpy as np
import pytest

from retrieval import FUSION_K, HashEncoder, RetrievalIndex, reciprocal_rank_fusion
from store import Chunk


def chunk(text: str) -> Chunk:
    document_id = uuid4()
    return Chunk(uuid4(), document_id, "paper.pdf", 1, text)


def test_rrf_combines_rankings_deterministically() -> None:
    fused = reciprocal_rank_fusion([[0, 1, 2], [2, 1, 3]])
    assert [item for item, _ in fused] == [2, 1, 0, 3]


def test_hybrid_search_returns_fused_candidates() -> None:
    chunks = [chunk(f"topic {index} common evidence") for index in range(30)]
    chunks[17].text = "rareword exact relevant evidence"
    index = RetrievalIndex(HashEncoder())
    index.rebuild(chunks)
    results = index.search("rareword relevant")
    assert results
    assert len(results) <= FUSION_K
    assert chunks[17] in [item for item, _ in results]
    assert all(item.embedding is not None for item in chunks)


def test_failed_rebuild_preserves_the_working_index() -> None:
    class BrokenEncoder:
        def encode(self, texts: list[str]) -> np.ndarray:
            raise RuntimeError("model unavailable")

    original = [chunk("existing searchable evidence")]
    index = RetrievalIndex(HashEncoder())
    index.rebuild(original)
    index.encoder = BrokenEncoder()
    with pytest.raises(RuntimeError):
        index.rebuild([*original, chunk("new evidence")])
    assert index.chunks == original
