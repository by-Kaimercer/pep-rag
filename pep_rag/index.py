"""Build the dense (two variants) indexes. BM25 is built in memory at load time (it takes ~1s)."""
import math
import re
import time
from collections import Counter, defaultdict

import numpy as np

from .config import DATA_DIR
from .ingest import contextual_text, load_chunks
from .llm import embed

STOP = set(
    "a an and are as at be but by for from has have how i if in into is it its of on or that the their then there these "
    "this to was were what when where which who why will with does do did can should would could about which".split()
)


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean())
        self.postings = defaultdict(list)  # term -> [(doc, tf)]
        for i, toks in enumerate(docs):
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        self.idf = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query_tokens):
        s = np.zeros(self.N, dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        for t in set(query_tokens):
            if t not in self.postings:
                continue
            idx, tf = zip(*self.postings[t])
            idx, tf = np.array(idx), np.array(tf, dtype=np.float32)
            s[idx] += self.idf[t] * tf * (self.k1 + 1) / (tf + norm[idx])
        return s


def emb_path(variant):
    return DATA_DIR / f"emb_{variant}.npy"


def main():
    chunks = load_chunks()
    variants = {
        "plain": [c["text"] for c in chunks],
        "contextual": [contextual_text(c) for c in chunks],
    }
    for name, texts in variants.items():
        if emb_path(name).exists():
            print(f"{name}: exists, skipping")
            continue
        t0 = time.perf_counter()
        m = embed(texts, "document")
        np.save(emb_path(name), m)
        print(f"{name}: {m.shape} in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
