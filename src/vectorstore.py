"""Vector store interface (Qdrant).

Handles storing chunk embeddings with their metadata and running dense
nearest-neighbor search against them. This is the dense half of retrieval;
the sparse (BM25) half lives in sparse.py.
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import EMBEDDING_DIM, QDRANT_COLLECTION, QDRANT_URL


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def _point_id(chunk_id: str) -> str:
    """Derive a stable UUID from a chunk_id string.

    Qdrant point IDs must be an unsigned int or a UUID, not an arbitrary
    string, but chunk_id (e.g. "guide.pdf_12_0") is the join key shared
    with the BM25 index and the chunk store. uuid5 deterministically maps
    the same chunk_id to the same UUID every time, so re-ingesting a
    document upserts (overwrites) the same points instead of duplicating
    them.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def ensure_collection(client: QdrantClient, collection: str = QDRANT_COLLECTION) -> None:
    """Create the collection if it doesn't already exist. Safe to call repeatedly."""
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(size=EMBEDDING_DIM, distance=qmodels.Distance.COSINE),
        )


def upsert_chunks(
    chunks: list[dict],
    client: QdrantClient | None = None,
    collection: str = QDRANT_COLLECTION,
) -> None:
    """Upsert embedded chunks (each with an "embedding" key) into Qdrant.

    Full chunk metadata (source, page_number, chunk_id, text) is stored as
    the point payload so search results carry everything needed to explain
    *why* a chunk was returned, without a second lookup.
    """
    client = client or get_client()
    ensure_collection(client, collection)

    points = [
        qmodels.PointStruct(
            id=_point_id(chunk["chunk_id"]),
            vector=chunk["embedding"].tolist() if hasattr(chunk["embedding"], "tolist") else chunk["embedding"],
            payload={
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "page_number": chunk["page_number"],
            },
        )
        for chunk in chunks
    ]

    client.upsert(collection_name=collection, points=points)


def search(
    query_embedding,
    top_k: int,
    client: QdrantClient | None = None,
    collection: str = QDRANT_COLLECTION,
) -> list[dict]:
    """Dense top-k nearest-neighbor search. Returns ranked chunk dicts with a "score" field."""
    client = client or get_client()
    vector = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding

    results = client.query_points(
        collection_name=collection,
        query=vector,
        limit=top_k,
    ).points

    return [
        {
            "chunk_id": point.payload["chunk_id"],
            "text": point.payload["text"],
            "source": point.payload["source"],
            "page_number": point.payload["page_number"],
            "score": point.score,
        }
        for point in results
    ]
