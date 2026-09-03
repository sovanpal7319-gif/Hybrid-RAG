"""Hit-rate eval script.

Runs every question in eval_set.json through the running /ask endpoint,
checks whether the known correct chunk shows up in the returned sources,
and records a hit-rate score. Reusable across both retrieval modes: run
once with --mode vector for the Step 5 baseline, then again with --mode
hybrid (once that path exists) for the Step 10 comparison — both runs
write into the same eval_results.json/.md so they can be compared
side by side.
"""

import argparse
import json
from pathlib import Path

import requests

EVAL_DIR = Path(__file__).resolve().parent
EVAL_SET_PATH = EVAL_DIR / "eval_set.json"
RESULTS_JSON_PATH = EVAL_DIR / "eval_results.json"
RESULTS_MD_PATH = EVAL_DIR / "eval_results.md"


def run_eval(mode: str, base_url: str, top_k: int) -> dict:
    eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))

    per_question = []
    for item in eval_set:
        response = requests.post(
            f"{base_url}/ask",
            json={"question": item["question"], "mode": mode, "top_k": top_k},
            timeout=120,
        )
        response.raise_for_status()
        sources = response.json()["sources"]
        retrieved_ids = [s["chunk_id"] for s in sources]
        hit = item["expected_chunk_id"] in retrieved_ids

        per_question.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_chunk_id": item["expected_chunk_id"],
                "retrieved_chunk_ids": retrieved_ids,
                "hit": hit,
            }
        )

    hit_rate = sum(q["hit"] for q in per_question) / len(per_question)

    return {
        "mode": mode,
        "top_k": top_k,
        "num_questions": len(per_question),
        "hit_rate": hit_rate,
        "results": per_question,
    }


def save_results(run_result: dict) -> dict:
    """Merge this run into eval_results.json, keyed by mode, then rewrite the .md summary."""
    all_results = {}
    if RESULTS_JSON_PATH.exists():
        all_results = json.loads(RESULTS_JSON_PATH.read_text(encoding="utf-8"))

    all_results[run_result["mode"]] = run_result
    RESULTS_JSON_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    lines = ["# Eval results: vector-only vs. hybrid", "", "| mode | top_k | questions | hit rate |", "| --- | --- | --- | --- |"]
    for mode in ("vector", "hybrid"):
        if mode in all_results:
            r = all_results[mode]
            lines.append(f"| {mode} | {r['top_k']} | {r['num_questions']} | {r['hit_rate']:.2%} |")
    RESULTS_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vector", "hybrid"], default="vector")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    result = run_eval(args.mode, args.base_url, args.top_k)
    save_results(result)

    print(f"mode={result['mode']} top_k={result['top_k']} hit_rate={result['hit_rate']:.2%} "
          f"({sum(q['hit'] for q in result['results'])}/{result['num_questions']})")
