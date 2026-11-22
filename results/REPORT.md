# PEP-RAG evaluation report

## Retrieval (52 answerable questions, k = 5)

| Pipeline | hit@5 | mrr@5 | pep_recall@5 | pep_recall@5[factual] | pep_recall@5[named] | pep_recall@5[multi] |
|---|---:|---:|---:|---:|---:|---:|
| BM25 (keyword) | 0.769 | 0.625 | 0.875 | 0.867 | **1.000** | 0.750 |
| Dense, raw chunks (naive baseline) | 0.558 | 0.410 | 0.837 | 0.900 | 0.667 | **0.850** |
| Dense, contextual chunks | 0.712 | 0.455 | 0.913 | 0.933 | **1.000** | 0.750 |
| Hybrid BM25 + dense (RRF) | 0.827 | 0.656 | **0.942** | **0.967** | **1.000** | 0.800 |
| Hybrid + PEP-number routing | **0.846** | 0.663 | **0.942** | **0.967** | **1.000** | 0.800 |
| Hybrid + routing + LLM rerank | 0.769 | **0.671** | 0.904 | 0.933 | **1.000** | 0.700 |

## End-to-end answers (60 questions, judged by LLM)

| Pipeline | accuracy | accuracy[factual] | accuracy[named] | accuracy[multi] | refusal_accuracy[unanswerable] | false_refusal_rate | citation_precision | hallucinated_citation_rate | latency_p50_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense, raw chunks (naive baseline) | 0.692 | 0.800 | 0.583 | 0.500 | **1.000** | 0.192 | 0.881 | **0.000** | **3.86** |
| Hybrid + PEP-number routing | **0.827** | **0.850** | **0.917** | 0.650 | **1.000** | **0.077** | 0.938 | 0.017 | 5.56 |
| Hybrid + routing + LLM rerank | 0.817 | 0.783 | **0.917** | **0.800** | **1.000** | 0.096 | **0.946** | 0.019 | 3.94 |
