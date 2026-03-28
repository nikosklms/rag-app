"""Retriever — two-stage retrieval with dual vector search and LLM reranking."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.config import settings
from src.ingestion.embedder import Embedder
from src.models.schemas import RetrievalResult, ChatMessage
from src.generation.generator import rewrite_query

async def retrieve(query: str, top_k: int = 5, history: list[ChatMessage] | None = None) -> list[RetrievalResult]:
    """Retrieve the top-K most relevant chunks.

    Stage 0: Rewrite query for better search.
    Stage 1: Search the 'documents' collection using semantic similarity.

    Args:
        query: The user's question.
        top_k: Number of results to return.
        history: Optional chat history to resolve context.

    Returns:
        List of RetrievalResult objects sorted by relevance.
    """
    collection = Embedder.get_collection()

    # Check if collection has any documents
    if collection.count() == 0:
        return []

    # ── Stage 0: Query rewriting ────────────────────────────────
    rewritten = rewrite_query(query, history)
    print(f"\n🔄 Query rewrite: '{query}' → '{rewritten}'\n")

    # Embed the rewritten query
    query_embedding = Embedder.embed_query(rewritten)

    n_results = min(top_k, collection.count())
    if n_results == 0:
        return []

    # ── Stage 1: Vector search ─────────────────────────────
    # Search chunk texts
    chunk_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    results: list[RetrievalResult] = []

    if chunk_results["documents"] and chunk_results["documents"][0]:
        for i, doc in enumerate(chunk_results["documents"][0]):
            metadata = chunk_results["metadatas"][0][i]
            distance = chunk_results["distances"][0][i]
            score = 1.0 - distance  # cosine distance → similarity

            results.append(RetrievalResult(
                text=doc,
                source=metadata.get("source", "unknown"),
                page=metadata.get("page", 0),
                chunk_index=metadata.get("chunk_index", 0),
                score=round(score, 4),
            ))

    # Log results
    print(f"📊 Retrieved {len(results)} chunks")
    for idx, res in enumerate(results, 1):
        print(f"  [{idx}] Score: {res.score:.3f} | Source: {res.source}")

    return results
