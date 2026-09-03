"""Answer generation.

Combines a user query with retrieved chunks into a prompt and calls a
local Ollama model to produce the final answer. Retrieval (vector or
hybrid) is deliberately out of scope here — this module's only job is
context + question -> answer.
"""

import requests

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS

PROMPT_TEMPLATE = """Answer the question using only the context below. \
If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i + 1}] (source: {chunk['source']}, page {chunk['page_number']})\n{chunk['text']}"
        for i, chunk in enumerate(chunks)
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Call the local Ollama model and return its answer text.

    Raises requests.RequestException if Ollama isn't reachable (e.g. not
    installed/running) or ValueError if the model name isn't pulled yet —
    callers decide how to surface that (see main.py's /ask handler, which
    keeps retrieval results usable even when generation fails).
    """
    prompt = build_prompt(question, chunks)

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    return response.json()["response"].strip()
