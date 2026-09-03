"""Retrieval.

Vector-only retrieval: takes a user query, embeds it with the same model
used to embed the chunks, and fetches the top-k nearest neighbors from the
Qdrant vector store. Hybrid retrieval layers dense + sparse retrieval,
RRF fusion, and cross-encoder reranking on top of that same dense path.
"""

from config import DEFAULT_TOP_K, FUSION_CANDIDATE_POOL, RERANK_TOP_N, RRF_K
from embeddings import load_embedding_model
import fusion
import reranker
import sparse
import vectorstore

# Loading the embedding model reads its weights off disk and is not cheap
# (see embeddings.load_embedding_model's docstring), so it's loaded once
# here and reused across every query rather than per-request.
_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_embedding_model()
    return _model


def retrieve_vector(query: str, top_k: int = DEFAULT_TOP_K, client=None) -> list[dict]:
    """query -> embed -> Qdrant top-k nearest neighbor search -> ranked chunks."""
    model = get_model()
    query_embedding = model.encode(query, convert_to_numpy=True)

    return vectorstore.search(query_embedding, top_k=top_k, client=client)


def retrieve_hybrid(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    fusion_pool: int = FUSION_CANDIDATE_POOL,
    rerank_top_n: int = RERANK_TOP_N,
    rrf_k: int = RRF_K,
    client=None,
) -> list[dict]:
    """Dense retrieval + sparse retrieval -> RRF fusion -> cross-encoder rerank.

    fusion_pool controls how many candidates each of the dense/sparse
    legs contributes to RRF; rerank_top_n controls how many of the fused
    candidates the (slower) cross-encoder actually scores; top_k is the
    final number of chunks returned after reranking.
    """
    dense_results = retrieve_vector(query, top_k=fusion_pool, client=client)
    sparse_results = sparse.search(query, top_k=fusion_pool)

    fused = fusion.reciprocal_rank_fusion([dense_results, sparse_results], k=rrf_k)

    return reranker.rerank(query, fused[:rerank_top_n], top_k=top_k)
