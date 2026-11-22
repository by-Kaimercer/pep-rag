import json, re, sys
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
QFILE = HERE / "questions.jsonl"
PEPS = HERE.parent / "data_raw_peps" / "peps"
norm = lambda s: re.sub(r"\s+", " ", s).strip()

fails = []
rows = [json.loads(l) for l in QFILE.read_text(encoding="utf-8").splitlines() if l.strip()]
if len(rows) != 60: fails.append(f"row count {len(rows)} != 60")
want = {"factual": 30, "named": 12, "multi": 10, "unanswerable": 8}
got = Counter(r["type"] for r in rows)
if dict(got) != want: fails.append(f"type counts {dict(got)} != {want}")
ids = [r["id"] for r in rows]
if len(set(ids)) != len(ids) or ids != [f"q{i:02d}" for i in range(1, 61)]: fails.append("ids not unique/sequential q01..q60")
cache = {}
for r in rows:
    for k in ("id", "type", "question", "answer", "gold_peps", "evidence"):
        if k not in r: fails.append(f"{r.get('id')} missing {k}")
    if r["type"] == "unanswerable":
        if r["gold_peps"] or r["evidence"]: fails.append(f"{r['id']} unanswerable must be empty")
        continue
    if r["type"] == "factual" and re.search(r"\bPEP\s*\d", r["question"], re.I): fails.append(f"{r['id']} factual mentions PEP number")
    if r["type"] == "named" and not re.search(r"\bPEP\s*\d", r["question"]): fails.append(f"{r['id']} named lacks PEP number")
    if r["type"] == "multi":
        if len(set(r["gold_peps"])) < 2 or {e["pep"] for e in r["evidence"]} != set(r["gold_peps"]): fails.append(f"{r['id']} multi needs evidence from each of 2+ PEPs")
    for e in r["evidence"]:
        p = e["pep"]
        if p not in r["gold_peps"]: fails.append(f"{r['id']} evidence pep {p} not in gold_peps")
        if p not in cache: cache[p] = norm((PEPS / f"pep-{p:04d}.rst").read_text(encoding="utf-8"))
        n = len(e["quote"].split())
        if not 8 <= n <= 40: fails.append(f"{r['id']} quote word count {n} (pep {p})")
        if norm(e["quote"]) not in cache[p]: fails.append(f"{r['id']} quote not found in PEP {p}: {e['quote'][:60]}")
distinct = {p for r in rows for p in r["gold_peps"]}
if fails:
    print("FAILED:"); [print(" -", f) for f in fails]; sys.exit(1)
print(f"all checks passed: 60 rows, types {dict(got)}, unique ids, all evidence quotes found; distinct PEPs = {len(distinct)}")
