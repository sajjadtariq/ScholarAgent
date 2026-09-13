import os
import re

from google import genai
from google.genai import errors

from store import Chunk

CITATION_RE = re.compile(r"\[(\d+)]")
GEMINI_MODEL = "gemini-3.5-flash-lite"


def _prompt(question: str, candidates: list[tuple[Chunk, float]]) -> str:
    evidence = "\n\n".join(
        f"[{index}] {chunk.document_name}, page {chunk.page_number}\n{chunk.text}"
        for index, (chunk, _) in enumerate(candidates, start=1)
    )
    return (
        "Answer the question using only the evidence below. Cite factual statements with the "
        "provided numeric IDs such as [1]. Do not invent citations. If the evidence is insufficient, "
        "say so. Write concise plain text paragraphs. Do not use Markdown headings, bold markers, "
        f"or bullet syntax.\n\nQuestion: {question}\n\nEvidence:\n{evidence}"
    )


def _citations(answer: str, candidates: list[tuple[Chunk, float]]) -> list[dict[str, object]]:
    citations: list[dict[str, object]] = []
    seen: set[int] = set()
    for match in CITATION_RE.finditer(answer):
        number = int(match.group(1))
        if number in seen or not 1 <= number <= len(candidates):
            continue
        seen.add(number)
        chunk, score = candidates[number - 1]
        citations.append(
            {
                "number": number,
                "chunk_id": str(chunk.id),
                "document_id": str(chunk.document_id),
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "text": chunk.text,
                "reranker_score": score,
            }
        )
    return citations


def _citation_retry_prompt(prompt: str, draft: str) -> str:
    return (
        f"{prompt}\n\n"
        "The draft below is missing citations in the required format. Rewrite it using only "
        "the supplied evidence. Include at least one valid evidence ID exactly as [1], [2], "
        "and so on. Do not use links, footnotes, or any other citation format.\n\n"
        f"Draft to rewrite:\n{draft}"
    )


async def generate_answer(
    question: str, candidates: list[tuple[Chunk, float]]
) -> tuple[str, list[dict[str, object]]]:
    if not candidates:
        return "I could not find relevant evidence in the uploaded PDFs.", []
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required to generate answers.")
    try:
        prompt = _prompt(question, candidates)
        async with genai.Client(api_key=api_key).aio as client:
            response = await client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            answer = response.text
            if not isinstance(answer, str):
                raise TypeError("Gemini returned an invalid answer.")
            citations = _citations(answer, candidates)
            if not citations:
                response = await client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=_citation_retry_prompt(prompt, answer),
                )
                answer = response.text
    except errors.APIError as exc:
        if exc.code == 429:
            message = "Gemini rate limit reached. Wait briefly and try again."
        elif exc.code in {400, 401, 403}:
            message = "Gemini rejected the API key or request. Check GEMINI_API_KEY."
        else:
            message = f"Gemini API request failed with status {exc.code}."
        raise RuntimeError(message) from exc
    if not isinstance(answer, str):
        raise TypeError("Gemini returned an invalid answer.")
    if not citations:
        citations = _citations(answer, candidates)
    if not citations:
        raise RuntimeError("The LLM answer did not contain a valid citation.")
    return answer, citations
