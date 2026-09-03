"""Flask frontend for the RAG service.

Thin UI layer over the FastAPI backend's /ask and /ingest endpoints -- all
retrieval/generation logic stays in the FastAPI service; this app only
renders a browser UI and proxies requests to it server-side (so the browser
never needs to know the backend's address, and CORS never comes up).
"""

import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
import requests

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/ask")
def ask():
    payload = request.get_json(force=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    try:
        response = requests.post(
            f"{BACKEND_URL}/ask",
            json={
                "question": question,
                "mode": payload.get("mode", "vector"),
                "top_k": payload.get("top_k", 5),
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return jsonify({"error": f"Backend unreachable: {exc}"}), 502

    return jsonify(response.json())


@app.post("/api/ingest")
def ingest():
    if "file" not in request.files or not request.files["file"].filename:
        return jsonify({"error": "No file selected"}), 400

    upload = request.files["file"]
    try:
        response = requests.post(
            f"{BACKEND_URL}/ingest",
            files={"file": (upload.filename, upload.stream, upload.mimetype)},
            timeout=300,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return jsonify({"error": f"Backend unreachable: {exc}"}), 502

    return jsonify(response.json())


if __name__ == "__main__":
    debug = os.getenv("FRONTEND_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=int(os.getenv("FRONTEND_PORT", "5000")), debug=debug)
