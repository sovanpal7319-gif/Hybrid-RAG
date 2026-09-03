"""Cross-encoder reranking.

Takes the top-N RRF-fused candidates and re-scores each (query, chunk)
pair with bge-reranker-base, a cross-encoder that reads the query and
chunk together (unlike the dense bi-encoder, which embeds them
separately) and so ranks relevance more precisely at the cost of being
too slow to run over the full corpus -- hence only reranking a small
candidate pool rather than replacing dense/sparse retrieval outright.
"""

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL_NAME

_model: CrossEncoder | None = None


def load_reranker() -> CrossEncoder:
    """Load once and reuse, same reasoning as embeddings.load_embedding_model."""
    global _model
    if _model is None:
        _model = CrossEncoder(RERANKER_MODEL_NAME)
    return _model


def rerank(query: str, chunks: list[dict], top_k: int, model: CrossEncoder | None = None) -> list[dict]:
    """Score each (query, chunk) pair, re-sort by cross-encoder score, keep top_k."""
    if not chunks:
        return []

    model = model or load_reranker()
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = model.predict(pairs)

    reranked = sorted(
        ({**chunk, "rerank_score": float(score)} for chunk, score in zip(chunks, scores)),
        key=lambda chunk: chunk["rerank_score"],
        reverse=True,
    )

    return reranked[:top_k]
