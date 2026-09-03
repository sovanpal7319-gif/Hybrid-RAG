"""Sparse retrieval (BM25).

Builds a rank_bm25 index over the same chunks persisted by store.py (the
shared source of truth from Step 1), keyed by the same chunk_id the dense
index uses, so fusion.py can combine both ranked lists by that id.
"""

import re

from rank_bm25 import BM25Okapi

import store

_bm25: BM25Okapi | None = None
_chunk_ids: list[str] = []
_chunk_lookup: dict[str, dict] = {}


def _tokenize(text: str) -> list[str]:
    # Simple lowercase word-boundary tokenizer. BM25 just needs consistent
    # term matching between corpus and query, not linguistic precision, so
    # a regex split is enough here without pulling in a heavier tokenizer.
    return re.findall(r"[a-z0-9]+", text.lower())


def build_index() -> BM25Okapi | None:
    """(Re)build the in-memory BM25 index from the current chunk store.

    Called once at startup and again after every /ingest so the sparse
    index never drifts out of sync with the chunk store it's built from.
    """
    global _bm25, _chunk_ids, _chunk_lookup

    chunks = store.load_chunks()
    _chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    _chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}

    _bm25 = BM25Okapi([_tokenize(chunk["text"]) for chunk in chunks]) if chunks else None
    return _bm25


def get_index() -> BM25Okapi | None:
    if _bm25 is None:
        build_index()
    return _bm25


def search(query: str, top_k: int) -> list[dict]:
    """query -> BM25 scoring -> ranked list of chunk dicts, each carrying a "score"."""
    bm25 = get_index()
    if bm25 is None:
        return []

    scores = bm25.get_scores(_tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [{**_chunk_lookup[_chunk_ids[i]], "score": float(scores[i])} for i in ranked_indices]
