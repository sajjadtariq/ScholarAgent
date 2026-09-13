from collections.abc import Iterator, Sequence
from typing import Self

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
from reranker import CrossEncoderReranker
from retrieval import HashEncoder, RetrievalIndex
from tests.fixtures import pdf_bytes


class KeywordScorer:
    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        query = pairs[0][0].casefold()
        return np.asarray(
            [sum(text.casefold().count(word) for word in query.split()) for _, text in pairs],
            dtype=np.float32,
        )


class GeminiResponse:
    text = "Nuclear fusion produces energy by combining light nuclei [1]."


class GeminiModels:
    async def generate_content(self, *, model: str, contents: str) -> GeminiResponse:
        assert model == "gemini-3.5-flash-lite"
        assert "What produces nuclear fusion energy?" in contents
        return GeminiResponse()


class GeminiAsyncClient:
    models = GeminiModels()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class GeminiClient:
    def __init__(self, *, api_key: str) -> None:
        assert api_key == "test-key"
        self.aio = GeminiAsyncClient()


@pytest.fixture(autouse=True)
def isolated_app(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("generation.genai.Client", GeminiClient)
    main.store.clear()
    monkeypatch.setattr(main, "retrieval_index", RetrievalIndex(HashEncoder()))
    monkeypatch.setattr(main, "reranker", CrossEncoderReranker(KeywordScorer()))
    yield
    main.store.clear()


def test_upload_list_file_and_cited_chat() -> None:
    client = TestClient(main.app)
    payload = pdf_bytes(
        [
            "The introduction discusses general research methods and reliable source review.",
            "Nuclear fusion produces energy by combining light nuclei. Nuclear fusion powers stars.",
        ]
    )
    upload = client.post(
        "/documents/upload",
        files={"file": ("physics.pdf", payload, "application/pdf")},
    )
    assert upload.status_code == 201
    document = upload.json()
    assert document["page_count"] == 2
    assert client.get("/documents").json()[0]["name"] == "physics.pdf"
    assert client.get(f"/documents/{document['id']}/file").content == payload

    chat = client.post("/chat", json={"question": "What produces nuclear fusion energy?"})
    assert chat.status_code == 200
    result = chat.json()
    assert result["citations"]
    assert result["citations"][0]["page_number"] == 2
    assert "[1]" in result["answer"]


def test_chat_requires_a_document() -> None:
    response = TestClient(main.app).post("/chat", json={"question": "What is discussed?"})
    assert response.status_code == 409


def test_upload_rejects_non_pdf_content() -> None:
    response = TestClient(main.app).post(
        "/documents/upload",
        files={"file": ("fake.pdf", b"not a pdf", "application/pdf")},
    )
    assert response.status_code == 422
