"""Tests for the generator module."""

from unittest.mock import patch, MagicMock

from src.generation.generator import generate, _build_prompt
from src.models.schemas import RetrievalResult


def _make_chunks() -> list[RetrievalResult]:
    """Create sample retrieval results for testing."""
    return [
        RetrievalResult(
            text="Machine learning learns from data.",
            source="ml.pdf",
            page=1,
            chunk_index=0,
            score=0.92,
        ),
        RetrievalResult(
            text="Neural networks use layers of neurons.",
            source="ml.pdf",
            page=3,
            chunk_index=2,
            score=0.85,
        ),
    ]


def test_build_prompt():
    """Prompt should contain query and all context chunks."""
    chunks = _make_chunks()
    prompt = _build_prompt("What is ML?", chunks)

    assert "What is ML?" in prompt
    assert "Machine learning learns from data." in prompt
    assert "Neural networks use layers of neurons." in prompt
    assert "ml.pdf" in prompt
    assert "Page 1" in prompt
    assert "Page 3" in prompt


@patch("src.generation.generator.httpx.post")
def test_generate_ollama(mock_post):
    """Ollama generation should call the right endpoint and return answer."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": "ML is a subset of AI."}
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    chunks = _make_chunks()

    with patch("src.generation.generator.settings") as mock_settings:
        mock_settings.llm_provider = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.ollama_model = "llama3.2"

        answer, model = generate("What is ML?", chunks, [])

    assert answer == "ML is a subset of AI."
    assert model == "llama3.2"
    mock_post.assert_called_once()


@patch("src.generation.generator.OpenAI")
def test_generate_openai(mock_openai_class):
    """OpenAI generation should use the OpenAI client."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_choice = MagicMock()
    mock_choice.message.content = "ML is artificial intelligence."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    chunks = _make_chunks()

    with patch("src.generation.generator.settings") as mock_settings:
        mock_settings.llm_provider = "openai"
        mock_settings.openai_api_key = "sk-test"
        mock_settings.openai_model = "gpt-4o-mini"

        answer, model = generate("What is ML?", chunks, [])

    assert answer == "ML is artificial intelligence."
    assert model == "gpt-4o-mini"
