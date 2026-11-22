"""Thin Ollama client with an on-disk cache so eval reruns are free and deterministic."""
import hashlib
import json
import sqlite3
import threading
import time

import numpy as np
import requests

from .config import CACHE_PATH, CHAT_MODEL, DATA_DIR, EMBED_MODEL, OLLAMA_URL

_lock = threading.Lock()
_db = None


def _cache():
    global _db
    if _db is None:
        DATA_DIR.mkdir(exist_ok=True)
        _db = sqlite3.connect(CACHE_PATH, check_same_thread=False)
        _db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
    return _db


def _cached(payload, fn):
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    with _lock:
        row = _cache().execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
    if row:
        return json.loads(row[0])
    value = fn()
    with _lock:
        _cache().execute("INSERT OR REPLACE INTO c VALUES (?, ?)", (key, json.dumps(value)))
        _cache().commit()
    return value


def embed(texts, kind, batch=64, model=EMBED_MODEL):
    """kind is 'document' or 'query' (nomic-embed-text expects task prefixes). Returns L2-normalised float32 matrix."""
    prefix = "search_document: " if kind == "document" else "search_query: "
    def post(inputs):
        return requests.post(f"{OLLAMA_URL}/api/embed", json={"model": model, "input": inputs, "truncate": True}, timeout=600)

    out = []
    for i in range(0, len(texts), batch):
        inputs = [prefix + t for t in texts[i : i + batch]]
        r = post(inputs)
        if r.status_code == 400:  # an input overflowed the model context; embed one by one, shortening offenders
            for t in inputs:
                while (r := post([t])).status_code == 400 and len(t) > 200:
                    t = t[: int(len(t) * 0.7)]
                r.raise_for_status()
                out.extend(r.json()["embeddings"])
            continue
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
    m = np.asarray(out, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def chat(messages, json_mode=False, model=CHAT_MODEL, num_ctx=8192, use_cache=True):
    """Deterministic (temperature 0) chat call. Returns (text, seconds); cached calls report the original latency."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "seed": 7, "num_ctx": num_ctx},
    }
    if json_mode:
        payload["format"] = "json"

    def call():
        t0 = time.perf_counter()
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
        r.raise_for_status()
        return {"text": r.json()["message"]["content"], "secs": time.perf_counter() - t0}

    res = _cached(payload, call) if use_cache else call()
    return res["text"], res["secs"]
