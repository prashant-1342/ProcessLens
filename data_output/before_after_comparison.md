# Before vs. After Benchmark Comparison

Evaluation results comparing baseline prompt (v1) versus improved prompt (v3) targeting top failure categories:

| Configuration | Prompt | RAG | Step F1 (95% CI) | Ordering Acc | Actor Acc | Schema Valid | Citations | ECE |
|---|---|---|---|---|---|---|---|---|
| **baseline_v1_no_rag** | v1 | No | **1.000** [1.000, 1.000] | 1.000 | 1.000 | 100.0% | 100.0% | 0.1266 |
| **improved_v3_no_rag** | v3 | No | **1.000** [1.000, 1.000] | 1.000 | 1.000 | 100.0% | 100.0% | 0.0200 |
| **improved_v3_rag** | v3 | Yes | **1.000** [1.000, 1.000] | 1.000 | 1.000 | 100.0% | 100.0% | 0.0200 |
