"""Embedding generation.

Converts text chunks (as produced by ingestion.chunk_all_pages()) into
vector embeddings using a local sentence-transformers model. Vector
database / storage logic is deliberately out of scope here — this
module's only job is text-in, vectors-out.
"""

from sentence_transformers import SentenceTransformer

# bge-small-en-v1.5 is a small (~130MB), fast, CPU-friendly model that still
# ranks well on retrieval benchmarks, so it's a good default for a local RAG
# pipeline where we don't want to depend on a paid embedding API.
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Batch size for model.encode() calls. Exposed as a module constant so both
# embed_chunks() and any callers/tests can reference the same default.
BATCH_SIZE = 32


def load_embedding_model() -> SentenceTransformer:
    """Load the sentence-transformers model.

    Loading a SentenceTransformer means reading its weights (tens to
    hundreds of MB) off disk/network and moving them onto the compute
    device (CPU/GPU). That cost is fixed per load and independent of how
    much text you plan to embed, so callers should call this once and
    reuse the returned model across every embed_chunks() call rather than
    loading it fresh per call or per chunk — reloading per call would pay
    that fixed cost repeatedly for no benefit.
    """
    return SentenceTransformer(MODEL_NAME)


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> list[dict]:
    """Embed each chunk's text and attach the resulting vector.

    Takes the chunk dicts produced by ingestion.chunk_all_pages() (each
    with at least a "text" field) and returns the same list of dicts, each
    now carrying an "embedding" key with that chunk's vector.
    """
    texts = [chunk["text"] for chunk in chunks]

    # Batching instead of calling model.encode() once per chunk matters
    # because the model runs as tensor ops on the CPU/GPU: encoding 32
    # texts in one batched matrix multiply is far faster per-text than 32
    # separate single-text forward passes, since each individual call pays
    # fixed overhead (Python-to-C dispatch, tensor allocation, and — on a
    # GPU — a device sync/transfer) that batching amortizes across many
    # texts at once. A batch_size of 32 (rather than embedding everything
    # in one giant batch) caps peak memory usage so embedding a
    # large document set doesn't try to hold every chunk's activations in
    # memory simultaneously.
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    return chunks


if __name__ == "__main__":
    from ingestion import chunk_all_pages, parse_all_pdfs

    pages = parse_all_pdfs()
    chunks = chunk_all_pages(pages)

    # Load once here, then reuse for the single embed_chunks() call below —
    # exactly the pattern callers of this module should follow.
    model = load_embedding_model()
    embedded_chunks = embed_chunks(chunks, model)

    print(f"Total chunks embedded: {len(embedded_chunks)}")

    if embedded_chunks:
        sample_embedding = embedded_chunks[0]["embedding"]
        print(f"Embedding dimension: {len(sample_embedding)}")
        print(f"First 5 values of sample embedding: {sample_embedding[:5]}")
