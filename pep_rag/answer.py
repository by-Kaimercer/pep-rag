"""Grounded answer generation with inline citations and an explicit refusal path."""
import re
import sys
import time

from .llm import chat

REFUSAL = "I don't know based on the PEPs."
CITE = re.compile(r"\[PEP\s*(\d{1,4})[^\]]*\]", re.IGNORECASE)
SOURCE_REF = re.compile(r"\[Source\s*(\d{1,2})\]", re.IGNORECASE)

SYSTEM = f"""You answer questions about Python using ONLY the numbered source excerpts from Python Enhancement Proposals (PEPs).
Rules:
- Every factual sentence must end with a citation like [PEP 572]. Only cite PEPs that appear in the sources.
- Be concise: 1-4 sentences. Prefer concrete details (versions, rules, names) from the sources.
- If the sources do not contain the answer, reply exactly: {REFUSAL}
- Never use outside knowledge, even if you are confident."""


def format_sources(chunks):
    return "\n\n".join(
        f"[Source {n + 1}] PEP {c['pep']}: {c['title']} (status: {c['status']}"
        f"{', Python-Version: ' + c['python_version'] if c['python_version'] else ''}) / section: {c['section']}\n{c['text']}"
        for n, c in enumerate(chunks)
    )


def generate(question, chunks):
    user = f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"
    text, secs = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    # small models often cite "[Source 2]" despite instructions; resolve those to the PEP they point at
    text = SOURCE_REF.sub(
        lambda m: f"[PEP {chunks[int(m.group(1)) - 1]['pep']}]" if 0 < int(m.group(1)) <= len(chunks) else m.group(0),
        text.strip(),
    )
    return {
        "answer": text,
        "citations": sorted({int(m) for m in CITE.findall(text)}),
        "refused": "don't know based on the peps" in text.lower(),
        "gen_secs": secs,
    }


def ask(question, retriever, mode="hybrid_route", k=5):
    t0 = time.perf_counter()
    chunks = retriever.search(question, mode=mode, k=k)
    t_ret = time.perf_counter() - t0
    out = generate(question, chunks)
    return out | {"sources": chunks, "retrieval_secs": t_ret}


def main():
    from .retrieve import Retriever

    q = " ".join(sys.argv[1:]) or "Why is the walrus operator not allowed as a top-level statement?"
    r = ask(q, Retriever())
    print(r["answer"], "\n")
    for c in r["sources"]:
        print(f"  - PEP {c['pep']}: {c['title']} / {c['section']}")


if __name__ == "__main__":
    main()
