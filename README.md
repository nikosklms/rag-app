A **Retrieval-Augmented Generation (RAG)** application that lets you upload PDFs and ask questions about their content. The system finds relevant passages and generates answers with source citations.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Frontend   │────▶│   FastAPI    │────▶│  LLM (API)   │
│  (Streamlit) │◀────│   Backend    │◀────│  OpenAI /    │
└──────────────┘     │              │     │  Ollama      │
                     │  ┌────────┐  │     └──────────────┘
                     │  │ChromaDB│  │
                     │  │(vectors)│  │
                     │  └────────┘  │
                     └──────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend API | FastAPI (Python) |
| Vector DB | ChromaDB |
| LLM | OpenAI API / Ollama (local) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| PDF Parsing | PyMuPDF |
| Frontend | Streamlit |
| Containerization | Docker + docker-compose |

## Quick Start

### 1. Setup

```bash
# Clone and enter the project
cd rag-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings
```

### 2. Choose your LLM

**Option A: Ollama (free, local)**
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.2    # ~2GB download
# .env: LLM_PROVIDER=ollama
```

**Option B: OpenAI (paid, cloud)**
```bash
# .env: LLM_PROVIDER=openai
# .env: OPENAI_API_KEY=sk-your-key-here
```

### 3. Run

```bash
# Start the API backend
uvicorn src.main:app --reload --port 8000

# In another terminal — start the frontend
streamlit run frontend/app.py
```

Open http://localhost:8501 in your browser.

### 4. Docker (alternative)

```bash
docker-compose up --build
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/upload` | Upload & index a PDF/TXT |
| `GET` | `/documents/` | List indexed documents |
| `DELETE` | `/documents/{id}` | Delete document + vectors |
| `POST` | `/query/ask` | Full RAG pipeline |
| `POST` | `/query/search` | Retrieval only (no LLM) |
| `GET` | `/health` | Health check |

API docs available at: http://localhost:8000/docs

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
rag-app/
├── src/
│   ├── main.py              # FastAPI entrypoint
│   ├── config.py            # Settings (env vars)
│   ├── ingestion/           # PDF parsing → chunking → embedding
│   ├── retrieval/           # Vector similarity search
│   ├── generation/          # LLM prompt + response
│   └── models/              # Pydantic schemas
├── frontend/
│   └── app.py               # Streamlit UI
├── tests/                   # Unit tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
