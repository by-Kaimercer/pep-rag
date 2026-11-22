"""Render results/*.json into results/REPORT.md."""
import json

from .config import RESULTS_DIR

LABELS = {
    "bm25": "BM25 (keyword)",
    "dense_plain": "Dense, raw chunks (naive baseline)",
    "dense_ctx": "Dense, contextual chunks",
    "hybrid": "Hybrid BM25 + dense (RRF)",
    "hybrid_route": "Hybrid + PEP-number routing",
    "hybrid_route_rerank": "Hybrid + routing + LLM rerank",
}


def table(summary, cols, higher_better=True):
    best = {}
    for c in cols:
        vals = [s[c] for s in summary.values() if s.get(c) is not None]
        if vals:
            best[c] = max(vals) if higher_better(c) else min(vals)
    lines = ["| Pipeline | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for mode, s in summary.items():
        cells = []
        for c in cols:
            v = s.get(c)
            cell = "–" if v is None else f"{v:.3f}" if isinstance(v, float) and v <= 1 and "latency" not in c else f"{v}"
            cells.append(f"**{cell}**" if v is not None and v == best.get(c) else cell)
        lines.append(f"| {LABELS.get(mode, mode)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    out = ["# PEP-RAG evaluation report\n"]
    ret_path = RESULTS_DIR / "retrieval.json"
    if ret_path.exists():
        summary = json.loads(ret_path.read_text())["summary"]
        cols = ["hit@5", "mrr@5", "pep_recall@5", "pep_recall@5[factual]", "pep_recall@5[named]", "pep_recall@5[multi]"]
        out += ["## Retrieval (52 answerable questions, k = 5)\n", table(summary, cols, lambda c: True), ""]
    gen_path = RESULTS_DIR / "generation.json"
    if gen_path.exists():
        gen = {m: v["summary"] for m, v in json.loads(gen_path.read_text()).items()}
        cols = [
            "accuracy", "accuracy[factual]", "accuracy[named]", "accuracy[multi]",
            "refusal_accuracy[unanswerable]", "false_refusal_rate", "citation_precision",
            "hallucinated_citation_rate", "latency_p50_s",
        ]
        lower = {"false_refusal_rate", "hallucinated_citation_rate", "latency_p50_s", "uncited_answer_rate"}
        out += ["## End-to-end answers (60 questions, judged by LLM)\n", table(gen, cols, lambda c: c not in lower), ""]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
