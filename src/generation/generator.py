"""Generator — sends context + query to an LLM and returns a cited answer."""

import json
from collections.abc import Iterator
import httpx
from openai import OpenAI

from src.config import settings
from src.models.schemas import RetrievalResult, ChatMessage


SYSTEM_PROMPT = """You are a highly capable and precise AI assistant. Your task is to answer the user's question based strictly on the provided document context.

Follow these rules unconditionally:
1. NO HALLUCINATIONS: Base your answer entirely on the provided context. Do not use outside knowledge. 
2. UNKNOWN ANSWERS: If the context does not contain the information needed to answer the question, do not guess. Reply exactly with: "I couldn't find the answer to that in the uploaded documents."
3. BE CONCISE: Give direct, clear answers without unnecessary filler. Use Markdown formatting (bullet points, bold text) to make your response easy to read.
4. NO SOURCES SECTION: DO NOT append a "Sources", "References", or "Citations" list at the end of your response. The UI already handles displaying the sources automatically.

"""


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


def _generate_openai(query: str, context_chunks: list[RetrievalResult], history: list[ChatMessage]) -> str:
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
        ] + [m.model_dump() for m in history] + [
            {"role": "user", "content": _build_prompt(query, context_chunks)},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    return response.choices[0].message.content or ""


def _generate_openai_stream(
    query: str, context_chunks: list[RetrievalResult], history: list[ChatMessage]
) -> Iterator[str]:
    """Stream answer deltas using OpenAI API."""
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
        ] + [m.model_dump() for m in history] + [
            {"role": "user", "content": _build_prompt(query, context_chunks)},
        ],
        temperature=0.3,
        max_tokens=1024,
        stream=True,
    )

    for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _generate_ollama(query: str, context_chunks: list[RetrievalResult], history: list[ChatMessage]) -> str:
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
        ] + [m.model_dump() for m in history] + [
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


def _generate_ollama_stream(
    query: str, context_chunks: list[RetrievalResult], history: list[ChatMessage]
) -> Iterator[str]:
    """Stream answer deltas using Ollama API."""
    url = f"{settings.ollama_base_url}/api/chat"
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
        ] + [m.model_dump() for m in history] + [
            {"role": "user", "content": _build_prompt(query, context_chunks)},
        ],
        "stream": True,
        "options": {
            "temperature": 0.3,
        },
    }

    with httpx.stream("POST", url, json=payload, timeout=120.0) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            message = data.get("message", {}).get("content", "")
            if message:
                yield message


def generate(query: str, context_chunks: list[RetrievalResult], history: list[ChatMessage]) -> tuple[str, str]:
    """Generate an answer using the configured LLM provider.

    Args:
        query: The user's question.
        context_chunks: Retrieved context chunks.

    Returns:
        Tuple of (answer_text, model_name).
    """
    if settings.llm_provider == "openai":
        answer = _generate_openai(query, context_chunks, history)
        model = settings.openai_model
    elif settings.llm_provider == "ollama":
        answer = _generate_ollama(query, context_chunks, history)
        model = settings.ollama_model
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")

    return answer, model


def generate_stream(
    query: str, context_chunks: list[RetrievalResult], history: list[ChatMessage]
) -> tuple[Iterator[str], str]:
    """Generate a streaming answer using the configured LLM provider."""
    if settings.llm_provider == "openai":
        return _generate_openai_stream(query, context_chunks, history), settings.openai_model
    if settings.llm_provider == "ollama":
        return _generate_ollama_stream(query, context_chunks, history), settings.ollama_model
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def generate_chat_title(query: str, answer: str) -> str:
    """Generate a short 3-5 word title for the chat based on the first interaction."""
    prompt = f"Based on the following query and answer, generate a very short, concise title (max 5 words) for this chat conversation. DO NOT add quotes around the title, just return the raw text.\n\nQuery: {query}\nAnswer: {answer}"
    
    if settings.llm_provider == "openai":
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=20,
        )
        return (response.choices[0].message.content or "New Chat").strip().replace('"', '')
        
    elif settings.llm_provider == "ollama":
        data = {
            "model": settings.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.3}
        }
        res = httpx.post(
            f"{settings.ollama_base_url}/api/chat",
            json=data,
            timeout=60.0
        )
        res.raise_for_status()
        return res.json().get("message", {}).get("content", "New Chat").strip().replace('"', '')
    else:
        return "New Chat"
