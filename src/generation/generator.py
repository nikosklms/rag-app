"""Generator — sends context + query to an LLM and returns a cited answer."""

import httpx
from openai import OpenAI

from src.config import settings
from src.models.schemas import RetrievalResult


SYSTEM_PROMPT = """You are a helpful assistant. Answer based ONLY on the provided context.
If the answer is not in the context, say "I couldn't find the answer in the provided documents."
Cite your sources using [Source: filename, page X] format.
Be concise and accurate."""


def _build_prompt(query: str, context_chunks: list[RetrievalResult]) -> str:
    """Build the user prompt with context chunks.

    Args:
        query: The user's question.
        context_chunks: Retrieved chunks to use as context.

    Returns:
        Formatted prompt string.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"[{i}] Source: {chunk.source}, Page {chunk.page}\n{chunk.text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    return f"Context:\n{context}\n\nUser question: {query}"


def _generate_openai(query: str, context_chunks: list[RetrievalResult]) -> str:
    """Generate answer using OpenAI API.

    Args:
        query: The user's question.
        context_chunks: Retrieved context chunks.

    Returns:
        Generated answer string.
    """
    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(query, context_chunks)},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    return response.choices[0].message.content or ""


def _generate_ollama(query: str, context_chunks: list[RetrievalResult]) -> str:
    """Generate answer using Ollama local API.

    Args:
        query: The user's question.
        context_chunks: Retrieved context chunks.

    Returns:
        Generated answer string.
    """
    url = f"{settings.ollama_base_url}/api/chat"

    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(query, context_chunks)},
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
        },
    }

    response = httpx.post(url, json=payload, timeout=120.0)
    response.raise_for_status()
    data = response.json()

    return data.get("message", {}).get("content", "")


def generate(query: str, context_chunks: list[RetrievalResult]) -> tuple[str, str]:
    """Generate an answer using the configured LLM provider.

    Args:
        query: The user's question.
        context_chunks: Retrieved context chunks.

    Returns:
        Tuple of (answer_text, model_name).
    """
    if settings.llm_provider == "openai":
        answer = _generate_openai(query, context_chunks)
        model = settings.openai_model
    elif settings.llm_provider == "ollama":
        answer = _generate_ollama(query, context_chunks)
        model = settings.ollama_model
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")

    return answer, model
