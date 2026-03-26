import numpy as np

def test_embed_returns_384_dim():
    from modal_app.encoder import embed
    result = embed(["hello world"])
    assert result.shape == (1, 384)

def test_embed_normalizes_output():
    from modal_app.encoder import embed
    result = embed(["test sentence"])
    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

def test_embed_context_window():
    from modal_app.encoder import embed_context
    history = [
        {"role": "user", "text": "what are you thinking about"},
        {"role": "assistant", "text": "nothing you'd understand"},
        {"role": "user", "text": "try me"},
    ]
    result = embed_context(history)
    assert result.shape == (384,)
