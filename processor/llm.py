import re

import requests


OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:4b"
DEFAULT_TIMEOUT = 120


class OllamaError(Exception):
    """Base exception for Ollama integration errors."""


class OllamaConnectionError(OllamaError):
    """Raised when DocuMind cannot connect to Ollama."""


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns an invalid or incomplete response."""


def build_prompt(
    question: str,
    retrieved_chunks: list[dict],
) -> list[dict]:
    """Build a grounded prompt using retrieved document chunks."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if not retrieved_chunks:
        raise ValueError("Retrieved chunks cannot be empty.")

    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        if not isinstance(chunk, dict):
            raise ValueError("Each retrieved chunk must be a dictionary.")

        text = chunk.get("text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Retrieved chunk text cannot be empty.")

        metadata = chunk.get("metadata", {})

        if not isinstance(metadata, dict):
            raise ValueError("Chunk metadata must be a dictionary.")

        source = metadata.get("source", "Unknown source")
        page = metadata.get("page_number")
        chunk_id = chunk.get("chunk_id", "Unknown")

        page_info = (
            str(page) if page is not None else "Not available"
        )

        context_parts.append(
            f"[S{index}]\n"
            f"Source: {source}\n"
            f"Page: {page_info}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Content:\n{text}"
        )

    if not context_parts:
        raise ValueError("No usable text found in retrieved chunks.")

    context = "\n\n".join(context_parts)

    system_prompt = """
You are DocuMind, an AI document assistant.

Answer the user's question using only the supplied document context.

Rules:
1. Give a direct, concise answer.
2. Do not display reasoning or planning.
3. Cite factual claims using [S1], [S2], etc.
4. Only cite sources provided in the context.
5. Never invent facts or citations.
6. If the context is insufficient, explicitly say:
   "I don't have enough information in the indexed documents
   to answer this question."

Return only the final answer.
"""

    user_prompt = f"""
Question:
{question}

Retrieved document context:
{context}

Answer the question using the supplied evidence.
Include source citations in your answer.
"""

    return [
        {
            "role": "system",
            "content": system_prompt.strip(),
        },
        {
            "role": "user",
            "content": user_prompt.strip(),
        },
    ]


def _extract_final_answer(content: str) -> str:
    """Extract the final answer from the model response."""

    if not isinstance(content, str):
        raise OllamaResponseError("Model response content must be text.")

    # Some Qwen responses include a preamble ending with </think>.
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[-1]

    return content.strip()


def generate_response(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Send chat messages to Ollama and return the final answer."""

    if not messages:
        raise ValueError("Messages cannot be empty.")

    if not model.strip():
        raise ValueError("Model name cannot be empty.")

    if timeout <= 0:
        raise ValueError("Timeout must be greater than zero.")

    payload = {
        "model": model,
        "messages": messages,
        "think": False,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 1536,
            "num_ctx": 4096,
        },
    }

    endpoint = f"{base_url.rstrip('/')}/api/chat"

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=timeout,
        )

    except requests.exceptions.Timeout as error:
        raise OllamaConnectionError(
            "Ollama request timed out."
        ) from error

    except requests.exceptions.ConnectionError as error:
        raise OllamaConnectionError(
            "Could not connect to Ollama. "
            "Make sure Ollama is running."
        ) from error

    except requests.exceptions.RequestException as error:
        raise OllamaConnectionError(
            f"Ollama request failed: {error}"
        ) from error

    try:
        response_data = response.json()

    except ValueError as error:
        raise OllamaResponseError(
            "Ollama returned invalid JSON."
        ) from error

    if not response.ok:
        error_message = response_data.get(
            "error",
            "Unknown Ollama API error.",
        )

        raise OllamaResponseError(error_message)

    if not isinstance(response_data, dict):
        raise OllamaResponseError(
            "Ollama returned an invalid response."
        )

    if response_data.get("done_reason") == "length":
        raise OllamaResponseError(
            "Model generation was incomplete because "
            "the output token limit was reached."
        )

    message = response_data.get("message")

    if not isinstance(message, dict):
        raise OllamaResponseError(
            "Ollama response does not contain a valid message."
        )

    content = _extract_final_answer(
        message.get("content", "")
    )

    if not content:
        raise OllamaResponseError(
            "Ollama returned an empty final answer."
        )

    return content

def extract_citations(answer: str) -> list[str]:
    """Extract unique citation labels in order of appearance."""

    if not isinstance(answer, str):
        raise ValueError("Answer must be a string.")

    citations = re.findall(r"\[(S\d+)\]", answer)

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(citations))


def validate_citations(
    answer: str,
    sources: list[dict],
) -> dict:
    """Separate valid citation labels from unknown references."""

    if not isinstance(sources, list):
        raise ValueError("Sources must be a list.")

    cited_labels = extract_citations(answer)

    available_labels = {
        source["citation"]
        for source in sources
        if isinstance(source, dict)
        and isinstance(source.get("citation"), str)
    }

    valid_citations = [
        citation
        for citation in cited_labels
        if citation in available_labels
    ]

    invalid_citations = [
        citation
        for citation in cited_labels
        if citation not in available_labels
    ]

    return {
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
    }



def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    model: str = DEFAULT_MODEL,
) -> dict:
    """Generate a grounded answer and return its source references."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if not retrieved_chunks:
        return {
            "answer": (
                "I don't have enough information in the "
                "indexed documents to answer this question."
            ),
            "sources": [],
        }

    messages = build_prompt(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    answer = generate_response(
        messages=messages,
        model=model,
    )

    sources = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {})

        sources.append({
            "citation": f"S{index}",
            "source": metadata.get("source", "Unknown source"),
            "page_number": metadata.get("page_number"),
            "chunk_id": chunk.get("chunk_id"),
        })

    citation_validation = validate_citations(
        answer=answer,
        sources=sources,
    )
    valid_labels = set(
        citation_validation["valid_citations"]
    )
    cited_sources = [
        source
        for source in sources
        if source["citation"] in valid_labels
    ]
    return {
        "answer": answer,
        "sources": sources,
        "cited_sources": cited_sources,
        "valid_citations": citation_validation["valid_citations"],
        "invalid_citations": citation_validation["invalid_citations"],
    }