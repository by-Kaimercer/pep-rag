"""Evaluation harness.

Retrieval (no LLM except for the rerank mode): Hit@5, MRR@5 at chunk level against gold evidence, and PEP-Recall@5
(fraction of gold PEPs present in the top 5; this is what matters for multi-PEP questions).

Generation (LLM answer + LLM judge): answer accuracy, citation precision, hallucinated-citation rate,
refusal accuracy on unanswerable questions, false-refusal rate, latency.

Usage:
    python -m pep_rag.evaluate retrieval
    python -m pep_rag.evaluate generation dense_plain hybrid_route_rerank
"""
import json
import re
import statistics
import sys
import time

from .answer import ask
from .config import EVAL_DIR, RESULTS_DIR
from .index import tokenize
from .llm import chat
from .retrieve import MODES, Retriever

K = 5


def norm(s):
    s = re.sub(r"[`*]", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def load_questions():
    with open(EVAL_DIR / "questions.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def attach_gold_chunks(questions, retriever):
    """Map each evidence quote to the chunk(s) that contain it. Falls back to best token overlap within the PEP
    (a quote can straddle a chunk boundary)."""
    for q in questions:
        gold = set()
        for ev in q.get("evidence", []):
            cand = retriever.by_pep.get(ev["pep"], [])
            quote = norm(ev["quote"])
            hits = [i for i in cand if quote in norm(retriever.chunks[i]["text"])]
            if not hits and cand:
                qt = set(tokenize(quote))
                hits = [max(cand, key=lambda i: len(qt & set(tokenize(retriever.chunks[i]["text"]))))]
            gold.update(hits)
        q["gold_chunks"] = sorted(gold)
    return questions


def retrieval_metrics(rows):
    def avg(key, subset=None):
        vals = [r[key] for r in rows if subset is None or r["type"] == subset]
        return round(sum(vals) / len(vals), 3) if vals else None

    out = {"hit@5": avg("hit"), "mrr@5": avg("rr"), "pep_recall@5": avg("pep_recall")}
    for t in ("factual", "named", "multi"):
        out[f"pep_recall@5[{t}]"] = avg("pep_recall", t)
        out[f"hit@5[{t}]"] = avg("hit", t)
    return out


def eval_retrieval(retriever, questions, modes=MODES):
    answerable = [q for q in questions if q["type"] != "unanswerable"]
    summary, per_question = {}, {}
    for mode in modes:
        rows, t0 = [], time.perf_counter()
        for q in answerable:
            got = retriever.search(q["question"], mode=mode, k=K)
            ids = [c["idx"] for c in got]
            peps = {c["pep"] for c in got}
            rank = next((n + 1 for n, i in enumerate(ids) if i in q["gold_chunks"]), None)
            rows.append(
                {
                    "id": q["id"],
                    "type": q["type"],
                    "hit": int(rank is not None),
                    "rr": 1 / rank if rank else 0.0,
                    "pep_recall": len(peps & set(q["gold_peps"])) / len(q["gold_peps"]),
                    "retrieved_peps": [c["pep"] for c in got],
                }
            )
        summary[mode] = retrieval_metrics(rows) | {"secs_per_query": round((time.perf_counter() - t0) / len(rows), 3)}
        per_question[mode] = rows
        print(f"{mode:22s} {summary[mode]}")
    return summary, per_question


JUDGE = """You are grading an answer to a question about Python Enhancement Proposals.
Question: {question}
Reference answer: {gold}
Candidate answer: {answer}

Grade the candidate ONLY on whether it conveys the key facts of the reference answer (wording may differ; extra correct
detail is fine; a refusal or "I don't know" is incorrect; contradicting the reference is incorrect).
Reply with JSON only: {{"verdict": "correct" | "partial" | "incorrect", "reason": "<one short sentence>"}}"""

VERDICT_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def judge(q, answer):
    text, _ = chat(
        [{"role": "user", "content": JUDGE.format(question=q["question"], gold=q["answer"], answer=answer)}],
        json_mode=True,
    )
    try:
        j = json.loads(text)
        verdict = str(j.get("verdict", "incorrect")).lower()
        return (verdict if verdict in VERDICT_SCORE else "incorrect"), j.get("reason", "")
    except json.JSONDecodeError:
        return "incorrect", "judge returned invalid JSON"


def eval_generation(retriever, questions, mode):
    rows = []
    for n, q in enumerate(questions, 1):
        r = ask(q["question"], retriever, mode=mode, k=K)
        retrieved = {c["pep"] for c in r["sources"]}
        row = {
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "gold": q["answer"],
            "gold_peps": q["gold_peps"],
            "answer": r["answer"],
            "citations": r["citations"],
            "retrieved_peps": sorted(retrieved),
            "refused": r["refused"],
            "latency": r["retrieval_secs"] + r["gen_secs"],
        }
        if q["type"] == "unanswerable":
            row["verdict"], row["reason"] = ("correct", "refused") if r["refused"] else ("incorrect", "answered an unanswerable question")
        elif r["refused"]:  # deterministic: the LLM judge was found to grade some refusals as correct
            row["verdict"], row["reason"] = "incorrect", "refused an answerable question"
        else:
            row["verdict"], row["reason"] = judge(q, r["answer"])
        rows.append(row)
        print(f"  [{mode}] {n}/{len(questions)} {q['id']} {row['verdict']}", flush=True)
    return rows


def generation_metrics(rows):
    ans = [r for r in rows if r["type"] != "unanswerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    cited = [r for r in ans if r["citations"] and not r["refused"]]
    all_cites = [(c, r) for r in rows for c in r["citations"]]
    by_type = {}
    for t in ("factual", "named", "multi"):
        sub = [r for r in ans if r["type"] == t]
        by_type[f"accuracy[{t}]"] = round(statistics.mean(VERDICT_SCORE[r["verdict"]] for r in sub), 3) if sub else None
    return {
        "accuracy": round(statistics.mean(VERDICT_SCORE[r["verdict"]] for r in ans), 3),
        **by_type,
        "refusal_accuracy[unanswerable]": round(sum(r["refused"] for r in una) / len(una), 3) if una else None,
        "false_refusal_rate": round(sum(r["refused"] for r in ans) / len(ans), 3),
        "citation_precision": round(
            statistics.mean(len(set(r["citations"]) & set(r["gold_peps"])) / len(r["citations"]) for r in cited), 3
        )
        if cited
        else None,
        "hallucinated_citation_rate": round(
            sum(c not in r["retrieved_peps"] for c, r in all_cites) / len(all_cites), 3
        )
        if all_cites
        else 0.0,
        "uncited_answer_rate": round(sum(1 for r in ans if not r["citations"] and not r["refused"]) / len(ans), 3),
        "latency_p50_s": round(statistics.median(r["latency"] for r in rows), 2),
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
    retriever = Retriever()
    questions = attach_gold_chunks(load_questions(), retriever)
    if what == "retrieval":
        summary, per_q = eval_retrieval(retriever, questions)
        (RESULTS_DIR / "retrieval.json").write_text(json.dumps({"summary": summary, "per_question": per_q}, indent=1))
    elif what == "generation":
        modes = sys.argv[2:] or ["dense_plain", "hybrid_route_rerank"]
        path = RESULTS_DIR / "generation.json"
        results = json.loads(path.read_text()) if path.exists() else {}
        for mode in modes:
            rows = eval_generation(retriever, questions, mode)
            results[mode] = {"summary": generation_metrics(rows), "rows": rows}
            print(mode, results[mode]["summary"])
            path.write_text(json.dumps(results, indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
