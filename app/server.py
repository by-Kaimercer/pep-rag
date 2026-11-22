"""FastAPI demo server for the local PEP RAG pipeline."""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pep_rag.answer import ask
from pep_rag.retrieve import MODES, Retriever

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="PEP RAG demo")
_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


@app.on_event("startup")
def _load():
    try:
        get_retriever()
    except Exception as e:  # indexes may still be building; /api/ask retries lazily
        print(f"[startup] retriever not loaded: {e}")


class AskBody(BaseModel):
    question: str
    mode: str = "hybrid_route"


def _read_json(name):
    p = ROOT / "results" / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/modes")
def modes():
    return {"modes": MODES}


@app.post("/api/ask")
def api_ask(body: AskBody):
    if body.mode not in MODES:
        raise HTTPException(400, f"unknown mode '{body.mode}'; expected one of {MODES}")
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "empty question")
    try:
        out = ask(q, get_retriever(), mode=body.mode, k=5)
    except Exception as e:
        raise HTTPException(503, f"pipeline unavailable: {e}")
    out["sources"] = [{k: v for k, v in c.items() if k != "idx"} for c in out["sources"]]
    return out


@app.get("/api/results")
def results():
    ret = _read_json("retrieval.json")
    gen = _read_json("generation.json")
    return {
        "retrieval": ret.get("summary") if isinstance(ret, dict) else None,
        "generation": (
            {m: v.get("summary") for m, v in gen.items() if isinstance(v, dict)}
            if isinstance(gen, dict)
            else None
        ),
    }
