"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """RAG application settings."""

    # LLM Provider
    llm_provider: str = "ollama"  # "openai" or "ollama"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_data"

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    top_k: int = 5

    # Upload directory
    upload_dir: str = "./uploads"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def ensure_dirs(self) -> None:
        """Create necessary directories if they don't exist."""
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)


settings = Settings()
