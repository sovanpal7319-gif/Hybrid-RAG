"""Ingestion pipeline glue.

Wires ingestion.py (parse + chunk) -> store.py (persist source-of-truth
chunks) -> embeddings.py (dense vectors) -> vectorstore.py (Qdrant upsert)
into the single flow the /ingest endpoint (and CLI bulk-ingest below) both
call. Kept separate from ingestion.py so that module can stay
parsing/chunking-only.
"""

from pathlib import Path

import embeddings
import sparse
import store
import vectorstore
from config import DATA_DIR
from ingestion import chunk_page, parse_pdf


def ingest_document(pdf_path: Path, model=None) -> dict:
    """Run one PDF through parse -> chunk -> persist -> embed -> upsert.

    Returns a small summary dict (pages/chunks counts) so callers (the API
    endpoint, the bulk CLI below) can report what happened.
    """
    pages = parse_pdf(pdf_path)

    chunks = []
    for page in pages:
        chunks.extend(chunk_page(page))

    store.upsert_chunks(chunks)
    sparse.build_index()  # keep BM25 in sync with the chunk store it's built from

    model = model or embeddings.load_embedding_model()
    embedded_chunks = embeddings.embed_chunks(chunks, model)
    vectorstore.upsert_chunks(embedded_chunks)

    return {"source": pdf_path.name, "pages": len(pages), "chunks": len(chunks)}


def ingest_all(data_dir: Path = DATA_DIR) -> list[dict]:
    """Bulk-ingest every PDF already sitting in data_dir.

    Used to seed the index from the repo's sample PDFs (for eval-set
    building, etc.) without uploading each one through the API individually.
    """
    model = embeddings.load_embedding_model()
    return [ingest_document(pdf_path, model=model) for pdf_path in sorted(data_dir.glob("*.pdf"))]


if __name__ == "__main__":
    summaries = ingest_all()
    for summary in summaries:
        print(f"{summary['source']}: {summary['pages']} pages, {summary['chunks']} chunks")
