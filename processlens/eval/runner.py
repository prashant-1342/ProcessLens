import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from processlens.core.chunker import chunk_document
from processlens.eval.calibration import compute_ece, generate_reliability_diagram
from processlens.eval.scorers import evaluate_process_model, EvaluationScore
from processlens.extractor.agent import ExtractionAgent
from processlens.models.process import ProcessModel


def compute_bootstrap_ci(
    values: list[float],
    num_bootstrap: int = 500,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0

    mean_val = sum(values) / len(values)
    if len(values) == 1:
        return mean_val, mean_val, mean_val

    boot_means = []
    n = len(values)
    rng = random.Random(42)

    for _ in range(num_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    low_idx = int((alpha / 2.0) * num_bootstrap)
    high_idx = int((1.0 - alpha / 2.0) * num_bootstrap)

    ci_low = boot_means[max(0, low_idx)]
    ci_high = boot_means[min(num_bootstrap - 1, high_idx)]

    return round(mean_val, 4), round(ci_low, 4), round(ci_high, 4)


@dataclass
class ConfigBenchmarkResult:
    config_name: str
    prompt_version: str
    use_rag: bool
    num_eval_runs: int
    step_f1_mean: float
    step_f1_ci: tuple[float, float]
    step_precision_mean: float
    step_recall_mean: float
    ordering_acc_mean: float
    ordering_acc_ci: tuple[float, float]
    actor_acc_mean: float
    actor_acc_ci: tuple[float, float]
    schema_valid_rate: float
    citation_support_rate: float
    retrieval_recall: float
    ece: float


class BenchmarkRunner:
    def __init__(
        self,
        docs_dir: Optional[Path] = None,
        gold_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        self.docs_dir = docs_dir or Path("data/documents")
        self.gold_dir = gold_dir or Path("data/gold_labels")
        self.output_dir = output_dir or Path("data_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_benchmark(
        self,
        configurations: list[dict[str, Any]],
        agent_factory=None,
        max_samples: Optional[int] = None,
    ) -> list[ConfigBenchmarkResult]:
        gold_files = sorted(list(self.gold_dir.glob("*.json")))
        if max_samples:
            gold_files = gold_files[:max_samples]

        all_results: list[ConfigBenchmarkResult] = []

        for config in configurations:
            cfg_name = config.get("name", "unnamed")
            prompt_ver = config.get("prompt_version", "v1")
            use_rag = config.get("use_rag", False)
            runs_count = config.get("runs", 1)

            scores_list: list[EvaluationScore] = []
            all_confidences = []
            all_correctness = []

            for run_i in range(runs_count):
                for gf in gold_files:
                    sop_id = gf.stem
                    doc_path = self.docs_dir / f"{sop_id}.txt"
                    if not doc_path.exists():
                        continue

                    doc_text = doc_path.read_text(encoding="utf-8")
                    gold_model = ProcessModel.model_validate_json(gf.read_text(encoding="utf-8"))

                    if agent_factory:
                        agent = agent_factory(prompt_version=prompt_ver, config=config)
                    else:
                        agent = ExtractionAgent(prompt_version=prompt_ver)

                    extraction_res = agent.extract(
                        document_text=doc_text,
                        document_title=gold_model.title,
                        use_rag=use_rag,
                        document_id=sop_id,
                    )

                    all_doc_chunks = chunk_document(doc_text)
                    valid_chunk_ids = {c.id for c in all_doc_chunks}

                    score = evaluate_process_model(
                        gold=gold_model,
                        pred=extraction_res.process_model,
                        valid_chunk_ids=valid_chunk_ids,
                        retrieved_chunk_ids=extraction_res.retrieved_chunk_ids if use_rag else None,
                    )

                    scores_list.append(score)
                    all_confidences.extend(score.step_confidences)
                    all_correctness.extend(score.step_correctness)

            f1_mean, f1_low, f1_high = compute_bootstrap_ci([s.step_f1 for s in scores_list])
            p_mean, _, _ = compute_bootstrap_ci([s.step_precision for s in scores_list])
            r_mean, _, _ = compute_bootstrap_ci([s.step_recall for s in scores_list])
            ord_mean, ord_low, ord_high = compute_bootstrap_ci([s.ordering_accuracy for s in scores_list])
            act_mean, act_low, act_high = compute_bootstrap_ci([s.actor_accuracy for s in scores_list])
            valid_mean, _, _ = compute_bootstrap_ci([s.schema_valid_rate for s in scores_list])
            cite_mean, _, _ = compute_bootstrap_ci([s.citation_support_rate for s in scores_list])
            ret_mean, _, _ = compute_bootstrap_ci([s.retrieval_recall_at_k for s in scores_list])

            ece, _ = compute_ece(all_confidences, all_correctness)

            diagram_path = self.output_dir / f"reliability_{cfg_name}.png"
            generate_reliability_diagram(
                all_confidences,
                all_correctness,
                output_path=str(diagram_path),
            )

            res = ConfigBenchmarkResult(
                config_name=cfg_name,
                prompt_version=prompt_ver,
                use_rag=use_rag,
                num_eval_runs=len(scores_list),
                step_f1_mean=f1_mean,
                step_f1_ci=(f1_low, f1_high),
                step_precision_mean=p_mean,
                step_recall_mean=r_mean,
                ordering_acc_mean=ord_mean,
                ordering_acc_ci=(ord_low, ord_high),
                actor_acc_mean=act_mean,
                actor_acc_ci=(act_low, act_high),
                schema_valid_rate=valid_mean,
                citation_support_rate=cite_mean,
                retrieval_recall=ret_mean,
                ece=ece,
            )
            all_results.append(res)

        self._export_csv(all_results)
        self._export_markdown(all_results)

        return all_results

    def _export_csv(self, results: list[ConfigBenchmarkResult]) -> None:
        csv_file = self.output_dir / "benchmark_results.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Configuration",
                "Prompt Version",
                "RAG Enabled",
                "Evaluations",
                "Step F1 (Mean)",
                "Step F1 95% CI",
                "Step Precision",
                "Step Recall",
                "Ordering Acc",
                "Ordering 95% CI",
                "Actor Acc",
                "Actor 95% CI",
                "Schema Valid Rate",
                "Citation Support Rate",
                "Retrieval Recall@k",
                "ECE",
            ])
            for r in results:
                writer.writerow([
                    r.config_name,
                    r.prompt_version,
                    r.use_rag,
                    r.num_eval_runs,
                    r.step_f1_mean,
                    f"[{r.step_f1_ci[0]}, {r.step_f1_ci[1]}]",
                    r.step_precision_mean,
                    r.step_recall_mean,
                    r.ordering_acc_mean,
                    f"[{r.ordering_acc_ci[0]}, {r.ordering_acc_ci[1]}]",
                    r.actor_acc_mean,
                    f"[{r.actor_acc_ci[0]}, {r.actor_acc_ci[1]}]",
                    r.schema_valid_rate,
                    r.citation_support_rate,
                    r.retrieval_recall,
                    r.ece,
                ])

    def _export_markdown(self, results: list[ConfigBenchmarkResult]) -> None:
        md_file = self.output_dir / "benchmark_results.md"
        lines = [
            "# ProcessLens Benchmark Evaluation Results",
            "",
            "| Configuration | Prompt | RAG | Step F1 (95% CI) | Precision | Recall | Ordering Acc (95% CI) | Actor Acc (95% CI) | Schema Valid | Citations | ECE |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in results:
            lines.append(
                f"| **{r.config_name}** | {r.prompt_version} | {'Yes' if r.use_rag else 'No'} | "
                f"**{r.step_f1_mean:.3f}** [{r.step_f1_ci[0]:.3f}, {r.step_f1_ci[1]:.3f}] | "
                f"{r.step_precision_mean:.3f} | {r.step_recall_mean:.3f} | "
                f"{r.ordering_acc_mean:.3f} [{r.ordering_acc_ci[0]:.3f}, {r.ordering_acc_ci[1]:.3f}] | "
                f"{r.actor_acc_mean:.3f} [{r.actor_acc_ci[0]:.3f}, {r.actor_acc_ci[1]:.3f}] | "
                f"{r.schema_valid_rate * 100:.1f}% | {r.citation_support_rate * 100:.1f}% | "
                f"{r.ece:.4f} |"
            )
        lines.append("")
        md_file.write_text("\n".join(lines), encoding="utf-8")
