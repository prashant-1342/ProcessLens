# ProcessLens Benchmark Evaluation Results

| Configuration | Prompt | RAG | Step F1 (95% CI) | Precision | Recall | Ordering Acc (95% CI) | Actor Acc (95% CI) | Schema Valid | Citations | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| **baseline_v1_no_rag** | v1 | No | **1.000** [1.000, 1.000] | 1.000 | 1.000 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 100.0% | 100.0% | 0.1266 |
| **improved_v3_no_rag** | v3 | No | **1.000** [1.000, 1.000] | 1.000 | 1.000 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 100.0% | 100.0% | 0.0200 |
| **improved_v3_rag** | v3 | Yes | **1.000** [1.000, 1.000] | 1.000 | 1.000 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 100.0% | 100.0% | 0.0200 |
