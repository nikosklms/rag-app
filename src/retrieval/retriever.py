"""Retriever — queries ChromaDB for the most relevant chunks."""

from src.ingestion.embedder import Embedder
from src.models.schemas import RetrievalResult


def retrieve(query: str, top_k: int = 5) -> list[RetrievalResult]:
    """Retrieve the top-K most relevant chunks for a query.

    Args:
        query: The user's question.
        top_k: Number of results to return.

    Returns:
        List of RetrievalResult objects sorted by relevance.
    """
    collection = Embedder.get_collection()

    # Check if collection has any documents
    if collection.count() == 0:
        return []

    # Embed the query
    query_embedding = Embedder.embed_query(query)

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Convert to RetrievalResult objects
    # ChromaDB returns cosine distance, convert to similarity score
    retrieval_results: list[RetrievalResult] = []

    if not results["documents"] or not results["documents"][0]:
        return []

    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        score = 1.0 - distance  # cosine distance → similarity

        retrieval_results.append(RetrievalResult(
            text=doc,
            source=metadata.get("source", "unknown"),
            page=metadata.get("page", 0),
            chunk_index=metadata.get("chunk_index", 0),
            score=round(score, 4),
        ))

    return retrieval_results
