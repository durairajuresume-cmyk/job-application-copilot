from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536


def embed_text(text: str, api_key: str) -> list[float]:
    """Return a 1536-dim embedding vector for the given text."""
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text[:8000],  # ~6 k tokens, within the 8191-token model limit
    )
    return resp.data[0].embedding
