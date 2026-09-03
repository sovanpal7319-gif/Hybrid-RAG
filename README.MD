# Hybrid RAG Document Search System

A high-performance, containerized hybrid Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **Qdrant**, **BM25**, **Reciprocal Rank Fusion (RRF)**, and **BGE Cross-Encoder Reranking**. 

This system provides a full end-to-end pipeline for document ingestion, multi-stage retrieval, LLM-powered generation, and interactive search via a Flask frontend web interface.

---

## 🌟 Key Features

- **Multi-Stage Retrieval Pipeline**:
  - **Dense Search**: Semantic embeddings using `BAAI/bge-small-en-v1.5` stored in **Qdrant**.
  - **Sparse Search**: Exact keyword matching powered by `rank_bm25`.
  - **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse result lists dynamically using constant $k=60$.
  - **Cross-Encoder Reranking**: Fine-grained relevance scoring of top candidates using `BAAI/bge-reranker-base`.
- **Evaluated & Benchmarked**: Includes an evaluation framework to compare vector-only vs. hybrid retrieval performance on target datasets.
- **FastAPI REST Service**: Clean, typed endpoints for document ingestion (`/ingest`) and question answering (`/ask`).
- **Interactive Flask Frontend**: Clean web interface to upload PDFs, run queries, toggle retrieval modes, and view chunk-level metadata and scores.
- **Dockerized Architecture**: Pre-configured `docker-compose` setup linking FastAPI, Qdrant, and Flask.

---

## 📊 Benchmark Results

Evaluated on test query benchmarks to measure retrieval hit-rate:

| Retrieval Mode | Top-K | Questions Tested | Hit Rate |
| :--- | :---: | :---: | :---: |
| **Vector-Only** (Baseline) | 5 | 15 | **86.67%** |
| **Hybrid (Dense + BM25 + RRF + Reranker)** | 5 | 15 | **100.00%** |

*Hybrid retrieval achieved a 100% hit rate by resolving keyword miss issues inherent to dense-only embeddings.*

---

## 📐 Architecture Overview

```text
               +-----------------------+
               |  PDF / Text Document  |
               +-----------+-----------+
                           |
                           v
              +-------------------------+
              | Ingestion & Chunking    |
              | (metadata & JSON store) |
              +------------+------------+
                           |
           +---------------+---------------+
           |                               |
           v                               v
 +-------------------+           +-------------------+
 |  Dense Embeddings |           |   Sparse BM25     |
 | (BGE + Qdrant)    |           |   Index           |
 +---------+---------+           +---------+---------+
           |                               |
           +---------------+---------------+
                           |
                           v
              +-------------------------+
              | Reciprocal Rank Fusion  |
              | (RRF) Candidate Merger  |
              +------------+------------+
                           |
                           v
              +-------------------------+
              | Cross-Encoder Reranker  |
              | (bge-reranker-base)     |
              +------------+------------+
                           |
                           v
              +-------------------------+
              |  LLM Generation         |
              |  (Ollama / Llama 3.2)   |
              +-------------------------+
```

---

## 🛠️ Quickstart Guide

### Option 1: Docker Compose (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shrivastava0820/Hybrid-Rag.git
   cd Hybrid-Rag
   ```

2. **Copy `.env.example` to `.env`**:
   ```bash
   cp .env.example .env
   ```

3. **Start the containers**:
   ```bash
   docker-compose up --build
   ```

4. **Access the services**:
   - **Frontend UI**: [http://localhost:5000](http://localhost:5000)
   - **FastAPI API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Qdrant Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

---

### Option 2: Local Setup

#### Prerequisites
- Python 3.10+
- Running instance of Qdrant (e.g. `docker run -p 6333:6333 qdrant/qdrant`)
- Ollama running locally (for generation)

#### 1. Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn main:app --reload --port 8000
```

#### 2. Frontend Setup
```bash
cd frontend
pip install -r requirements.txt
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📡 API Reference

### `POST /ingest`
Uploads and processes a PDF file into chunks, populating both Qdrant and the BM25 index.

- **Request**: `multipart/form-data` with `file=@path/to/doc.pdf`
- **Response**:
  ```json
  {
    "source": "document.pdf",
    "pages": 12,
    "chunks": 48
  }
  ```

### `POST /ask`
Queries the system using either vector-only or hybrid mode.

- **Request Body**:
  ```json
  {
    "question": "What is Reciprocal Rank Fusion?",
    "mode": "hybrid",
    "top_k": 5
  }
  ```
- **Response**:
  ```json
  {
    "answer": "Reciprocal Rank Fusion (RRF) is a method to combine multiple ranked result lists...",
    "sources": [
      {
        "chunk_id": "doc_chunk_12",
        "source": "document.pdf",
        "page_number": 4,
        "text": "...chunk snippet text...",
        "rerank_score": 0.942
      }
    ],
    "retrieval_mode": "hybrid"
  }
  ```

---

## ⚙️ Configuration Settings

Key parameters can be configured via environment variables or `.env`:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `QDRANT_URL` | `http://localhost:6333` | Connection URL for vector store |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama service endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Model name used for text generation |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Dense embedding model |
| `CHUNK_SIZE_TOKENS` | `400` | Token length per document chunk |
| `CHUNK_OVERLAP_TOKENS` | `50` | Overlap length between chunks |
| `RRF_K` | `60` | Constant $k$ for Reciprocal Rank Fusion |
| `FUSION_CANDIDATE_POOL` | `30` | Top candidates selected before reranking |

---

## 🧪 Running Evaluations

To run the retrieval benchmark across both vector and hybrid modes:

```bash
python eval/run_eval.py
```
Results will be output to `eval/eval_results.json` and summarized in `eval/eval_results.md`.

---

## 📄 License
MIT License. Free for open use and contribution.