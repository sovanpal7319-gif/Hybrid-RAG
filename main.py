"""FastAPI application entrypoint for the RAG service.

Wires up the ingestion and ask endpoints. Route handlers stay thin —
all retrieval/generation logic lives in src/, this file just adapts it to
HTTP.
"""

import sys
from pathlib import Path
from typing import Literal

# src/ modules import each other with bare names (e.g. `from config import
# ...`), matching the convention already used in embeddings.py's __main__
# block. Adding src/ to sys.path once here lets every src module be
# imported the same way regardless of whether it's the top-level script or
# imported as a dependency of another src module.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
import requests

import generation
import ingest_pipeline
import retrieval
from config import DATA_DIR, DEFAULT_TOP_K

app = FastAPI(title="RAG Service")


@app.get("/health")
def health() -> dict[str, str]:
    """Basic liveness check placeholder."""
    return {"status": "ok"}


class SourceChunk(BaseModel):
    chunk_id: str
    source: str
    page_number: int
    text: str
    score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None


class AskRequest(BaseModel):
    question: str
    mode: Literal["vector", "hybrid"] = "vector"
    top_k: int = DEFAULT_TOP_K


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    retrieval_mode: str


class IngestResponse(BaseModel):
    source: str
    pages: int
    chunks: int


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = DATA_DIR / file.filename
    dest_path.write_bytes(await file.read())

    summary = ingest_pipeline.ingest_document(dest_path)
    return IngestResponse(**summary)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if request.mode == "vector":
        chunks = retrieval.retrieve_vector(request.question, top_k=request.top_k)
    else:
        chunks = retrieval.retrieve_hybrid(request.question, top_k=request.top_k)

    try:
        answer = generation.generate_answer(request.question, chunks)
    except (requests.RequestException, KeyError) as exc:
        # Retrieval already succeeded at this point, so we still return the
        # sources that would have grounded an answer (needed by the eval
        # script's hit-rate check, and useful for debugging) rather than
        # failing the whole request over a generation-side outage.
        answer = f"[generation unavailable: {exc}]"

    return AskResponse(
        answer=answer,
        sources=[SourceChunk(**chunk) for chunk in chunks],
        retrieval_mode=request.mode,
    )
