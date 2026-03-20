# Architecture

## Overview

DocMind is a Retrieval-Augmented Generation (RAG) application that enables users to upload documents and ask questions about their content.

## Pipeline

1. **Ingestion**: PDF/TXT → Parse (PyMuPDF) → Chunk (overlapping 500-char) → Embed (sentence-transformers) → Store (ChromaDB)
2. **Retrieval**: Query → Embed → Cosine similarity search → Top-K chunks
3. **Generation**: Chunks + Query → System prompt → LLM (OpenAI/Ollama) → Cited answer

## Components

- **FastAPI Backend** (`src/main.py`): REST API with 6 endpoints
- **Ingestion Module** (`src/ingestion/`): `parser.py` → `chunker.py` → `embedder.py`
- **Retrieval Module** (`src/retrieval/`): Vector search in ChromaDB
- **Generation Module** (`src/generation/`): LLM prompt construction and API calls
- **Streamlit Frontend** (`frontend/app.py`): Upload + Chat UI

## Data Flow

```
User uploads PDF
       │
       ▼
   parser.py  ──→  Extract text per page
       │
       ▼
   chunker.py ──→  Split into overlapping chunks
       │
       ▼
   embedder.py ──→  Generate embeddings → Store in ChromaDB
       │
       ▼
   User asks question
       │
       ▼
   retriever.py ──→  Embed query → Find top-K similar chunks
       │
       ▼
   generator.py ──→  Build prompt with context → Call LLM → Return cited answer
```
