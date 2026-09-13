from collections.abc import Sequence
from typing import Self
from uuid import uuid4

import numpy as np
import pytest
from google.genai import errors

from generation import generate_answer
from reranker import CrossEncoderReranker
from store import Chunk


class KeywordScorer:
    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        return np.asarray([text.count("important") for _, text in pairs], dtype=np.float32)


class FakeResponse:
    text = "The reported accuracy was 92 percent [1]."


class FakeModels:
    async def generate_content(self, *, model: str, contents: str) -> FakeResponse:
        assert model == "gemini-3.5-flash-lite"
        assert "Evidence:" in contents
        return FakeResponse()


class FakeAsyncClient:
    models = FakeModels()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class FakeClient:
    def __init__(self, *, api_key: str) -> None:
        assert api_key == "test-key"
        self.aio = FakeAsyncClient()


class FailingModels:
    async def generate_content(self, *, model: str, contents: str) -> FakeResponse:
        del model, contents
        raise errors.APIError(429, {"error": {"message": "quota exceeded"}})


class FailingAsyncClient(FakeAsyncClient):
    models = FailingModels()


class FailingClient:
    def __init__(self, *, api_key: str) -> None:
        del api_key
        self.aio = FailingAsyncClient()


class RetryModels:
    calls = 0

    async def generate_content(self, *, model: str, contents: str) -> FakeResponse:
        del model
        self.calls += 1
        response = FakeResponse()
        if self.calls == 1:
            response.text = "The reported accuracy was 92 percent."
        else:
            assert "missing citations" in contents
            response.text = "The reported accuracy was 92 percent [1]."
        return response


class RetryAsyncClient(FakeAsyncClient):
    models = RetryModels()


class RetryClient:
    def __init__(self, *, api_key: str) -> None:
        del api_key
        self.aio = RetryAsyncClient()


def make_chunk(text: str, page: int) -> Chunk:
    return Chunk(uuid4(), uuid4(), "study.pdf", page, text)


def test_cross_encoder_reranks_and_keeps_five() -> None:
    candidates = [(make_chunk("important " * index, index + 1), 0.1) for index in range(8)]
    results = CrossEncoderReranker(KeywordScorer()).rerank("important", candidates)
    assert len(results) == 5
    assert results[0][0].page_number == 8


@pytest.mark.asyncio
async def test_llm_generation_maps_numeric_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("generation.genai.Client", FakeClient)
    candidates = [(make_chunk("The reported accuracy was 92 percent. More detail.", 4), 8.5)]
    answer, citations = await generate_answer("What was the accuracy?", candidates)
    assert "[1]" in answer
    assert citations[0]["page_number"] == 4
    assert citations[0]["reranker_score"] == 8.5


@pytest.mark.asyncio
async def test_llm_retries_when_gemini_omits_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    RetryAsyncClient.models.calls = 0
    monkeypatch.setattr("generation.genai.Client", RetryClient)
    candidates = [(make_chunk("The reported accuracy was 92 percent.", 4), 8.5)]

    answer, citations = await generate_answer("What was the accuracy?", candidates)

    assert answer.endswith("[1].")
    assert citations[0]["page_number"] == 4
    assert RetryAsyncClient.models.calls == 2


@pytest.mark.asyncio
async def test_gemini_rate_limit_has_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("generation.genai.Client", FailingClient)
    candidates = [(make_chunk("Evidence about the result.", 2), 1.0)]
    with pytest.raises(RuntimeError, match="rate limit"):
        await generate_answer("What is the result?", candidates)
