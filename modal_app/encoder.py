import numpy as np
from sentence_transformers import SentenceTransformer

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model

def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of strings. Returns [N, 384] normalized float32 array."""
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.astype(np.float32)

def embed_context(history: list[dict], max_turns: int = 3) -> np.ndarray:
    """
    Embed the last `max_turns` turns of conversation history.
    history: list of {"role": "user"|"assistant", "text": str}
    Returns: [384] float32 array
    """
    recent = history[-max_turns:]
    text = " [SEP] ".join(f"{h['role']}: {h['text']}" for h in recent)
    return embed([text])[0]
