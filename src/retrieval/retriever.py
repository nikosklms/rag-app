"""Retriever — Hybrid search (Vector + BM25) and Small-to-Big Retrieval."""

from rank_bm25 import BM25Okapi

from src.ingestion.embedder import Embedder
from src.models.schemas import RetrievalResult, ChatMessage
from src.generation.generator import rewrite_query

class BM25Searcher:
    """Manages the BM25 keyword search index."""
    _index: BM25Okapi | None = None
    _corpus_ids: list[str] = []
    _corpus_texts: list[str] = []

    @classmethod
    def rebuild(cls):
        collection = Embedder.get_collection()
        results = collection.get(
            where={"chunk_type": "child"},
            include=["documents"]
        )
        if not results["ids"]:
            cls._index = None
            cls._corpus_ids = []
            cls._corpus_texts = []
            return

        cls._corpus_ids = results["ids"]
        cls._corpus_texts = results["documents"]

        # Simple tokenization by whitespace
        tokenized_corpus = [doc.lower().split(" ") for doc in cls._corpus_texts]
        cls._index = BM25Okapi(tokenized_corpus)
        print(f"✅ BM25 index rebuilt with {len(cls._corpus_ids)} child chunks.")

    @classmethod
    def search(cls, query: str, top_k: int) -> list[tuple[str, float]]:
        if not cls._index:
            return []
        tokenized_query = query.lower().split(" ")
        scores = cls._index.get_scores(tokenized_query)
        
        scored_docs = [(cls._corpus_ids[i], scores[i]) for i in range(len(scores)) if scores[i] > 0]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]

def rebuild_bm25_index():
    """Trigger a rebuild of the global BM25 index."""
    BM25Searcher.rebuild()


def compute_rrf(vector_ranked_ids: list[str], bm25_ranked_ids: list[str], k: int = 60) -> dict[str, float]:
    """Compute Reciprocal Rank Fusion scores.
    
    Formula: RRF_score = 1 / (k + rank)
    """
    rrf_scores: dict[str, float] = {}

    for rank, chunk_id in enumerate(vector_ranked_ids, 1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

    for rank, chunk_id in enumerate(bm25_ranked_ids, 1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

    return rrf_scores


async def retrieve(query: str, top_k: int = 5, history: list[ChatMessage] | None = None) -> list[RetrievalResult]:
    """Retrieve the context using Hybrid Search and Small-to-Big expansion.

    Stage 0: Rewrite query for better search.
    Stage 1: Search child chunks using Vector Similarity.
    Stage 2: Search child chunks using BM25.
    Stage 3: Merge results using Reciprocal Rank Fusion (RRF).
    Stage 4: Map winning child chunks to their Parent chunks and return them.
    """
    collection = Embedder.get_collection()

    if collection.count() == 0:
        return []

    # ── Stage 0: Query rewriting ────────────────────────────────
    rewritten = rewrite_query(query, history)
    print(f"\n🔄 Query rewrite: '{query}' → '{rewritten}'\n")

    # If the BM25 index is missing but we have items, build it
    if BM25Searcher._index is None:
        rebuild_bm25_index()

    # ── Stage 1: Vector search (only children) ───────────────────
    query_embedding = Embedder.embed_query(rewritten)
    
    n_results = min(top_k * 2, collection.count())
    if n_results == 0:
        return []

    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"chunk_type": "child"},
        include=[] # Only need ids for RRF
    )
    vector_ranked_ids = vector_results["ids"][0] if vector_results["ids"] and vector_results["ids"][0] else []

    # ── Stage 2: BM25 search (only children) ─────────────────────
    bm25_results = BM25Searcher.search(rewritten, top_k=n_results)
    bm25_ranked_ids = [res[0] for res in bm25_results]

    # ── Stage 3: RRF Fusion ──────────────────────────────────────
    rrf_scores = compute_rrf(vector_ranked_ids, bm25_ranked_ids)
    
    sorted_fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_child_ids = [chunk_id for chunk_id, score in sorted_fused[:top_k]]

    if not top_child_ids:
        return []

    # Fetch top child metadata to get parent_ids
    children_data = collection.get(
        ids=top_child_ids,
        include=["metadatas"]
    )

    # ── Stage 4: Small-to-Big (Parent) Expansion ──────────────────
    parent_ids_to_fetch = []
    # Keep track of mapping for source logging
    child_to_parent_meta = {}

    for i, meta in enumerate(children_data["metadatas"]):
        parent_id = meta.get("parent_id")
        if parent_id and parent_id not in parent_ids_to_fetch:
            parent_ids_to_fetch.append(parent_id)
            child_to_parent_meta[parent_id] = meta 
            
    # Fetch actual parent texts
    parents_data = collection.get(
        ids=parent_ids_to_fetch,
        include=["documents", "metadatas"]
    )

    results: list[RetrievalResult] = []

    if parents_data["documents"]:
        # Map ids to their returned index
        id_to_index = {pid: i for i, pid in enumerate(parents_data["ids"])}

        for i, parent_id in enumerate(parent_ids_to_fetch):
            if parent_id not in id_to_index:
                continue
                
            idx = id_to_index[parent_id]
            doc_text = parents_data["documents"][idx]
            parent_meta = parents_data["metadatas"][idx]
            
            # The RRF score is not an absolute similarity, so we just use the rank position for score
            pseudo_score = round(1.0 - (i * 0.1), 3)

            results.append(RetrievalResult(
                text=doc_text,
                source=parent_meta.get("source", "unknown"),
                page=parent_meta.get("page", 0),
                chunk_index=parent_meta.get("chunk_index", 0),
                score=pseudo_score, 
            ))

    print(f"📊 Hybrid RRF retrieved {len(top_child_ids)} children, expanded to {len(results)} Parent chunks.")
    for idx, res in enumerate(results, 1):
        snippet = res.text[:60].replace("\n", " ") + "..." if len(res.text) > 60 else res.text.replace("\n", " ")
        print(f"  [{idx}] RRF Rank | Source: {res.source} (Parent Chunk {res.chunk_index}) | Text: '{snippet}'")

    return results

