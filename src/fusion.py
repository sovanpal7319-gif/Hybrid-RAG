"""Reciprocal Rank Fusion.

Combines a dense-ranked list and a sparse-ranked list of chunks (as
returned by retrieval.retrieve_vector / sparse.search) into one fused
ranking, keyed by chunk_id. Implemented directly per the README -- no
fusion library needed, the formula is a few lines.
"""

from config import RRF_K


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = RRF_K,
) -> list[dict]:
    """score(chunk) = sum over lists containing it of 1 / (k + rank_in_that_list).

    `rank_in_that_list` is 1-based. Each input list is a ranked list of
    chunk dicts (dense results, sparse results, ...); chunks are matched
    across lists by chunk_id. Returns chunks sorted by descending fused
    score, each carrying a "fused_score" field and the chunk metadata from
    wherever it was first seen.
    """
    fused_scores: dict[str, float] = {}
    chunk_by_id: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            chunk_id = chunk["chunk_id"]
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            # Drop the per-list "score" (dense cosine similarity vs. BM25
            # raw score live on incomparable scales) -- fused_score is the
            # only score that means the same thing regardless of which
            # list(s) a chunk came from.
            chunk_by_id.setdefault(chunk_id, {field: value for field, value in chunk.items() if field != "score"})

    ranked_chunk_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)

    return [
        {**chunk_by_id[chunk_id], "fused_score": fused_scores[chunk_id]}
        for chunk_id in ranked_chunk_ids
    ]
