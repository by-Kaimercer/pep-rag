"""Retrieval strategies, from a naive dense baseline up to hybrid + query routing + LLM reranking."""
import json
import re
from collections import defaultdict

import numpy as np

from .index import BM25, emb_path, tokenize
from .ingest import contextual_text, load_chunks
from .llm import chat, embed

MODES = ["bm25", "dense_plain", "dense_ctx", "hybrid", "hybrid_route", "hybrid_route_rerank"]
PEP_REF = re.compile(r"\bPEP[\s\-]*(\d{1,4})\b", re.IGNORECASE)
RRF_K = 60


def rrf(rankings, k=RRF_K):
    """Reciprocal Rank Fusion: score(d) = sum over rankers of 1 / (k + rank)."""
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[int(idx)] += 1.0 / (k + rank + 1)
    return sorted(fused, key=fused.get, reverse=True)


class Retriever:
    def __init__(self):
        self.chunks = load_chunks()
        self.emb = {"plain": np.load(emb_path("plain")), "ctx": np.load(emb_path("contextual"))}
        self.bm25 = BM25([tokenize(contextual_text(c)) for c in self.chunks])
        self.by_pep = defaultdict(list)
        for i, c in enumerate(self.chunks):
            self.by_pep[c["pep"]].append(i)

    # --- base scorers -------------------------------------------------------
    def _scores(self, query, qvec):
        return {
            "bm25": self.bm25.scores(tokenize(query)),
            "dense_plain": self.emb["plain"] @ qvec,
            "dense_ctx": self.emb["ctx"] @ qvec,
        }

    @staticmethod
    def _top(scores, n, restrict=None):
        if restrict is not None:
            sub = np.asarray(restrict)
            return sub[np.argsort(-scores[sub])[:n]].tolist()
        return np.argsort(-scores)[:n].tolist()

    def _hybrid(self, s, n, restrict=None, depth=100):
        return rrf([self._top(s["bm25"], depth, restrict), self._top(s["dense_ctx"], depth, restrict)])[:n]

    # --- public API -----------------------------------------------------------
    def search(self, query, mode="hybrid_route", k=5):
        qvec = embed([query], "query")[0]
        s = self._scores(query, qvec)
        if mode in ("bm25", "dense_plain", "dense_ctx"):
            ids = self._top(s[mode], k)
        elif mode == "hybrid":
            ids = self._hybrid(s, k)
        else:
            pool = 20 if mode.endswith("rerank") else k
            ids = self._routed(query, s, pool)
            if mode.endswith("rerank"):
                ids = self._rerank(query, ids, k)
        return [self.chunks[i] | {"idx": i} for i in ids[:k]]

    def _routed(self, query, s, n):
        """If the user names PEPs explicitly, search inside those PEPs first (interleaved), then fill globally."""
        named = [int(m) for m in PEP_REF.findall(query) if int(m) in self.by_pep]
        named = list(dict.fromkeys(named))
        if not named:
            return self._hybrid(s, n)
        per_pep = [self._hybrid(s, n, restrict=self.by_pep[p]) for p in named]
        out = []
        for row in zip(*[r + [None] * (n - len(r)) for r in per_pep]):
            out.extend(i for i in row if i is not None)
        reserve = max(1, n // 4)  # leave room for globally-relevant chunks from other PEPs
        out = out[: n - reserve]
        for i in self._hybrid(s, n * 2):
            if len(out) >= n:
                break
            if i not in out:
                out.append(i)
        return out

    def _rerank(self, query, ids, k):
        """Listwise LLM rerank: show the candidates, ask for the ids of the most useful passages in order."""
        lines = []
        for n, i in enumerate(ids):
            c = self.chunks[i]
            words = c["text"].split()
            snippet = " ".join(words[:110]) + (" ..." if len(words) > 110 else "")
            lines.append(f"[{n}] PEP {c['pep']} ({c['title']}) / {c['section']}\n{snippet}")
        prompt = (
            f"Question: {query}\n\nCandidate passages:\n\n" + "\n\n".join(lines) + "\n\n"
            f"Select the passages that contain information needed to answer the question, most useful first. "
            f'Reply with JSON only: {{"ranking": [ids]}} with at most {k} ids.'
        )
        text, _ = chat([{"role": "user", "content": prompt}], json_mode=True)
        try:
            picked = [int(x) for x in json.loads(text).get("ranking", [])]
        except (ValueError, TypeError, AttributeError):
            picked = []
        order = [ids[p] for p in dict.fromkeys(picked) if 0 <= p < len(ids)]
        order += [i for i in ids if i not in order]  # backfill so we always return k
        return order[:k]
