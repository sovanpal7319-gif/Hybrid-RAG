# Build Prompt: Hybrid RAG Document Search Pipeline

## Objective
Build a complete, working hybrid Retrieval-Augmented Generation (RAG) system for document search, exposed via a FastAPI service. The system must support two retrieval modes that can be benchmarked against each other:
1. **Vector-only retrieval** (dense embeddings, baseline)
2. **Hybrid retrieval** (dense + sparse, fused with Reciprocal Rank Fusion, reranked with a cross-encoder)

Build and wire up in this order — vector-only must run and pass its own eval before the hybrid layers are added — so the codebase preserves a working baseline for a before/after comparison, not just a final combined system.

---

## Tech stack
- **API/serving:** FastAPI
- **Parsing:** `unstructured`, `pypdf`
- **Embeddings:** OpenAI embeddings API or `bge` (open-source) embeddings
- **Vector store:** Qdrant or Weaviate
- **Sparse retrieval:** `rank_bm25` (or Elasticsearch if a heavier setup is preferred)
- **Fusion:** Reciprocal Rank Fusion (implemented directly, no library needed)
- **Reranking:** `bge-reranker-base` cross-encoder
- **Deployment:** Docker
- **Evaluation:** simple hit-rate/precision script now; RAGAS-ready structure for later

---

## Step 1 — Ingestion & chunking
- Accept PDF/text documents as input
- Parse with `unstructured`/`pypdf` into clean text
- Chunk text with a configurable chunk size and overlap (expose as parameters, don't hardcode)
- Attach metadata to every chunk: source document, chunk index, character offsets
- Persist chunks (e.g., as JSON or a lightweight table) so both the dense and sparse indexes can be built from the same source of truth

## Step 2 — Dense index (vector-only baseline)
- Embed every chunk (OpenAI or `bge`)
- Upsert into Qdrant/Weaviate with chunk metadata attached
- Implement a retrieval function: query → embed → top-k nearest neighbor search → return ranked chunks

## Step 3 — Generation
- Prompt template that inserts retrieved chunks + the user question
- Call the LLM, return the answer along with the source chunks used (for traceability)

## Step 4 — FastAPI service (baseline)
- `POST /ask` — accepts a query, runs vector-only retrieval + generation, returns `{answer, sources, retrieval_mode: "vector"}`
- `POST /ingest` — accepts a document, runs it through Step 1–2

## Step 5 — Baseline eval
- Build a small eval set: 10–20 questions with known correct source chunks
- Script that runs each question through `/ask`, checks whether the correct chunk appears in top-k, and records a hit-rate score
- Save this score — it's the "before" number

**Checkpoint: do not proceed until Step 5 runs and produces a baseline score.**

## Step 6 — Sparse index
- Build a BM25 index (`rank_bm25`) over the same chunks from Step 1, keyed by the same chunk IDs as the dense index
- Retrieval function: query → BM25 scoring → ranked list of chunk IDs

## Step 7 — Reciprocal Rank Fusion
- Given the dense ranked list and the sparse ranked list for a query, compute fused scores:
  `score(chunk) = sum over lists containing chunk of 1 / (k + rank_in_that_list)`, with `k = 60` as default (expose as a parameter)
- Return chunks sorted by fused score

## Step 8 — Cross-encoder reranking
- Take the top-N (e.g., 20–30) fused candidates
- Score each (query, chunk) pair with `bge-reranker-base`
- Re-sort by cross-encoder score, keep top-k

## Step 9 — Hybrid endpoint
- `POST /ask` should accept a `mode` parameter (`"vector"` or `"hybrid"`), or add a separate `/ask/hybrid` endpoint — either way, both modes must remain independently callable
- Hybrid path: dense retrieval + sparse retrieval → RRF fusion → cross-encoder rerank → generation
- Response should include `retrieval_mode` and, ideally, intermediate scores (fused score, rerank score) for debugging/demo purposes

## Step 10 — Hybrid eval
- Re-run the exact same eval set from Step 5 against the hybrid endpoint
- Produce a comparison table: vector-only hit-rate vs. hybrid hit-rate
- This comparison is a first-class deliverable — store it (e.g., `eval_results.json` or a markdown table) so it can be shown directly

## Step 11 — Dockerize
- Dockerfile for the FastAPI app + all dependencies
- `docker-compose.yml` if Qdrant/Weaviate/Elasticsearch are run as separate services

---

## Non-negotiable structural requirements
- Vector-only and hybrid retrieval must both remain callable and testable independently after the build is done — don't let hybrid replace/delete the baseline path
- All configuration (chunk size/overlap, top-k, RRF constant `k`, rerank top-N) should be parameters, not hardcoded magic numbers
- Every retrieval response should carry enough metadata (source, scores) to explain *why* a chunk was returned — this matters for both debugging and demoing
- Keep the eval script and eval set reusable across both modes; it's the artifact that proves the hybrid upgrade helped

## Out of scope for this build (Phase 2, later)
Query rewriting, multi-turn conversation memory, RAGAS metrics, observability/tracing. Structure the code so these can be added later without a rewrite, but do not implement them now.

---

## Frontend

A small Flask UI lives in `frontend/` — upload a PDF, ask questions, toggle between vector-only and hybrid mode, and inspect per-source scores. It's a thin proxy: all it does is forward `/api/ask` and `/api/ingest` calls server-side to the FastAPI backend's `/ask` and `/ingest`, so the browser never talks to the backend directly. It has its own lightweight `requirements.txt` (Flask + requests only) so its Docker image doesn't pull in the backend's ML stack.

Run locally (with the backend already running on `:8000` and Qdrant up):

```
cd frontend
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000`. Point it at a different backend with `BACKEND_URL` (see `.env.example`).

With `docker-compose up`, the frontend is available at `http://localhost:5000` alongside the API (`:8000`) and Qdrant (`:6333`).