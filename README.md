# PEP-RAG

Grounded question answering over all 747 Python Enhancement Proposals, running fully locally on an 8 GB laptop GPU, with an evaluation harness that measures every design decision instead of guessing.

**Headline result:** moving from a naive dense-retrieval baseline to hybrid retrieval with query routing raised end-to-end answer accuracy from **69% to 83%**, cut false refusals from **19% to 8%**, and raised retrieval Hit@5 from **0.56 to 0.85**, while refusing **100%** of unanswerable questions in every configuration.

## What it does

Ask a question about Python's design ("Why can't the walrus operator be a top-level statement?") and get a short answer where every claim cites the PEP it came from. If the PEPs don't cover the question, it says so instead of inventing an answer.

```
python -m pep_rag.answer "What does PEP 8 say about line length for comments?"
python -m uvicorn app.server:app --port 8765     # web UI with an Ask tab and an Evals dashboard
```

## Pipeline

1. **Ingestion** (`pep_rag/ingest.py`): parses each PEP's RFC-822 header (number, title, status, type, Python version) and splits the reStructuredText body along its section hierarchy, packing paragraphs into ~220-word chunks. 747 PEPs become 16,594 chunks.
2. **Indexes** (`pep_rag/index.py`): a BM25 index (implemented from scratch) and two dense indexes using `nomic-embed-text`. One embeds raw chunk text, the other embeds *contextual* chunks prefixed with their PEP number, title, status and section path.
3. **Retrieval** (`pep_rag/retrieve.py`), six strategies:
   - `bm25`: keyword search
   - `dense_plain`: naive embedding search
   - `dense_ctx`: embedding search on contextual chunks
   - `hybrid`: BM25 and dense rankings fused with Reciprocal Rank Fusion
   - `hybrid_route`: when the question names a PEP ("PEP 604"), search inside that PEP first, then fill from the global ranking
   - `hybrid_route_rerank`: listwise reranking of 20 candidates by the local LLM
4. **Generation** (`pep_rag/answer.py`): `qwen2.5:7b` at temperature 0, instructed to cite `[PEP N]` after every claim and to refuse with a fixed phrase when the sources are insufficient. Stray `[Source N]` citations are resolved back to PEP numbers.

## Evaluation

The eval set (`evals/questions.jsonl`) has 60 hand-checked questions over 38 PEPs:

- 30 paraphrased factual questions that never mention a PEP number
- 12 questions that name a PEP
- 10 multi-PEP questions that need two documents
- 8 plausible but unanswerable questions

Every gold answer has a verbatim evidence quote, and `evals/verify_questions.py` confirms all 62 quotes exist in the corpus.

### Retrieval (52 answerable questions, k = 5)

| Pipeline | Hit@5 | MRR@5 | PEP-Recall@5 | Recall (multi-PEP) |
|---|---:|---:|---:|---:|
| BM25 | 0.769 | 0.625 | 0.875 | 0.750 |
| Dense, raw chunks (baseline) | 0.558 | 0.410 | 0.837 | 0.850 |
| Dense, contextual chunks | 0.712 | 0.455 | 0.913 | 0.750 |
| Hybrid (RRF) | 0.827 | 0.656 | 0.942 | 0.800 |
| **Hybrid + routing** | **0.846** | 0.663 | **0.942** | 0.800 |
| Hybrid + routing + LLM rerank | 0.769 | **0.671** | 0.904 | 0.700 |

### End-to-end answers (60 questions)

| Pipeline | Accuracy | Multi-PEP acc. | Unanswerable refused | False refusals | Citation precision |
|---|---:|---:|---:|---:|---:|
| Dense, raw chunks (baseline) | 0.692 | 0.500 | 1.000 | 0.192 | 0.881 |
| **Hybrid + routing** | **0.827** | 0.650 | 1.000 | **0.077** | 0.938 |
| Hybrid + routing + LLM rerank | 0.817 | **0.800** | 1.000 | 0.096 | **0.946** |

The full tables, including per-question-type breakdowns, are in `results/REPORT.md`.

### What the numbers taught

- **Contextual chunk headers are the cheapest big win.** Prefixing chunks with their PEP title and section raised dense Hit@5 from 0.56 to 0.71 with no other change, because a chunk like "This was rejected because..." is meaningless without knowing which PEP it is from.
- **Keyword search beats naive embeddings on this corpus.** Python questions are full of exact identifiers (`typing.Union`, `__init_subclass__`, PEP numbers) that embeddings blur. Hybrid fusion gets the best of both.
- **The LLM reranker is a trade-off.** A 7B model reranking 20 candidates improved the top-1 ordering (best MRR) and multi-PEP answers (0.65 to 0.80), but dropped relevant passages often enough to lower Hit@5 and single-PEP accuracy. Overall it was slightly worse, so it is not the default. A proper cross-encoder reranker is the obvious next experiment.
- **Fixing the measurement mattered as much as fixing the model.** The first eval run scored 23% of answers as "uncited" because the model wrote `[Source 2]` instead of `[PEP 572]`. That was a measurement artifact, fixed by resolving source references.

### Auditing the judge

Answer accuracy is graded by the same local 7B model, so the judge itself was audited. A stronger model (Claude Haiku) blind-graded all 52 baseline answers. Agreement was **71% (Cohen's κ = 0.48)**, and the 7B judge was **more lenient** in 11 of 15 disagreements. It also graded some refusals as correct, so refusals on answerable questions are now scored deterministically instead of by the judge (`results/judge_audit.json`).

Treat absolute accuracy numbers as optimistic. The *relative* comparison between pipelines is more trustworthy, because every pipeline is graded by the same judge.

## Limitations

- 60 questions is enough to see large effects but not to separate pipelines within ~5 points of each other.
- The eval questions were written with knowledge of the corpus, so they may be easier than real user questions.
- Latency figures were measured on a shared laptop GPU and are indicative only.

## Running it

Requirements: Python 3.11+, [Ollama](https://ollama.com), and about 6 GB of free VRAM.

```
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
git clone --depth 1 https://github.com/python/peps data_raw_peps   # on Windows add: -c core.longpaths=true
python -m pep_rag.ingest                   # parse and chunk
python -m pep_rag.index                    # embed (~20 min on an RTX 5050)
python -m pep_rag.evaluate retrieval       # retrieval ablation
python -m pep_rag.evaluate generation dense_plain hybrid_route hybrid_route_rerank
python -m pep_rag.report                   # writes results/REPORT.md
```

All LLM calls are cached in SQLite (`data/llm_cache.sqlite`), so rerunning an evaluation is nearly instant and fully reproducible.
