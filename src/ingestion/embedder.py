"""Embedder — generates vector embeddings and stores them in ChromaDB."""

import uuid

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.ingestion.chunker import Chunk


class Embedder:
    """Manages embedding generation and ChromaDB storage.

    Uses a singleton-like pattern for the model to avoid reloading.
    """

    _model: SentenceTransformer | None = None
    _client: chromadb.ClientAPI | None = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """Get or load the embedding model (cached)."""
        if cls._model is None:
            cls._model = SentenceTransformer(settings.embedding_model)
        return cls._model

    @classmethod
    def get_client(cls) -> chromadb.ClientAPI:
        """Get or create the ChromaDB client (cached)."""
        if cls._client is None:
            settings.ensure_dirs()
            cls._client = chromadb.PersistentClient(
                path=settings.chroma_persist_dir
            )
        return cls._client

    @classmethod
    def get_collection(cls) -> chromadb.Collection:
        """Get or create the documents collection."""
        client = cls.get_client()
        return client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )
    @classmethod
    def embed_chunks(cls, chunks: list[Chunk], document_id: str) -> int:
        """Embed a list of chunks and store in ChromaDB.

        Args:
            chunks: List of Chunk objects to embed.
            document_id: Unique ID for the parent document.

        Returns:
            Number of chunks stored.
        """
        if not chunks:
            return 0

        model = cls.get_model()
        collection = cls.get_collection()

        # Generate embeddings for chunk texts
        texts = [chunk.text for chunk in chunks]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # Prepare data for ChromaDB
        ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "document_id": document_id,
                "source": chunk.source,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]

        # Upsert chunk texts into documents collection
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        return len(chunks)

    @classmethod
    def embed_query(cls, query: str) -> list[float]:
        """Embed a single query string.

        Args:
            query: The query text.

        Returns:
            Embedding vector as a list of floats.
        """
        model = cls.get_model()
        return model.encode(query, show_progress_bar=False).tolist()

    @classmethod
    def delete_document(cls, document_id: str) -> bool:
        """Delete all chunks belonging to a document from the collection.

        Args:
            document_id: The document ID whose chunks to delete.

        Returns:
            True if deletion was performed.
        """
        collection = cls.get_collection()

        # Get all chunk IDs for this document
        results = collection.get(
            where={"document_id": document_id},
        )
        if results["ids"]:
            collection.delete(ids=results["ids"])
        return True

    @classmethod
    def get_document_count(cls) -> int:
        """Return the number of unique documents indexed."""
        collection = cls.get_collection()
        results = collection.get(include=["metadatas"])
        if not results["metadatas"]:
            return 0
        doc_ids = {m["document_id"] for m in results["metadatas"]}
        return len(doc_ids)

    @classmethod
    def list_documents(cls) -> list[dict]:
        """List all indexed documents with their chunk counts.

        Returns:
            List of dicts with document_id, filename, chunk_count.
        """
        collection = cls.get_collection()
        results = collection.get(include=["metadatas"])

        if not results["metadatas"]:
            return []

        # Group by document_id
        docs: dict[str, dict] = {}
        for meta in results["metadatas"]:
            doc_id = meta["document_id"]
            if doc_id not in docs:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "filename": meta["source"],
                    "chunk_count": 0,
                }
            docs[doc_id]["chunk_count"] += 1

        return list(docs.values())

    @classmethod
    def generate_document_id(cls) -> str:
        """Generate a unique document ID."""
        return str(uuid.uuid4())[:8]

    @classmethod
    def get_document_info(cls, document_id) -> dict:
        """Return document statistics."""

        collection = cls.get_collection()

        results = collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        
        if not results["ids"]:
            raise ValueError(f"Document {document_id} not found")

        chunks = results["documents"]
        chunk_count = len(chunks)
        metas = results["metadatas"]
        
        avg_chars_per_chunk = round(sum(len(c) for c in chunks) / chunk_count, 1) if chunk_count else 0

        pages = {m["page"] for m in metas if "page" in m and m["page"] is not None}

        pages_count = len(pages)

        return {
            "document_id": document_id,
            "chunk_count": chunk_count,
            "avg_chars_per_chunk": avg_chars_per_chunk,
            "pages_count": pages_count,
        }
        
        
