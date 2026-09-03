"""Chunk store.

Persists the plain-text chunk records produced by ingestion.py as JSON, so
they act as the single source of truth that both the dense (Qdrant) and
sparse (BM25) indexes are built from. Embeddings are deliberately not
persisted here — this file holds pre-embedding chunk records only.
"""

import json

from config import CHUNKS_PATH, PROCESSED_DIR


def save_chunks(chunks: list[dict], path=CHUNKS_PATH) -> None:
    """Write chunk records to disk as a JSON array, replacing any existing file."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Strip embeddings if present — this store is the pre-embedding source of
    # truth; vectors belong in the vector store, not duplicated on disk here.
    plain_chunks = [{k: v for k, v in chunk.items() if k != "embedding"} for chunk in chunks]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(plain_chunks, f, ensure_ascii=False, indent=2)


def load_chunks(path=CHUNKS_PATH) -> list[dict]:
    """Read chunk records back from disk. Returns [] if the store doesn't exist yet."""
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def upsert_chunks(new_chunks: list[dict], path=CHUNKS_PATH) -> list[dict]:
    """Merge new_chunks into the existing store, keyed by chunk_id.

    Re-ingesting the same source document produces the same chunk_ids
    (source name + page + index are all deterministic), so this replaces
    stale chunks for that id rather than duplicating them, while chunks
    from other documents are left untouched.
    """
    existing = {chunk["chunk_id"]: chunk for chunk in load_chunks(path)}

    for chunk in new_chunks:
        existing[chunk["chunk_id"]] = {k: v for k, v in chunk.items() if k != "embedding"}

    merged = list(existing.values())
    save_chunks(merged, path)
    return merged
