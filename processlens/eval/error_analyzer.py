import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from processlens.eval.scorers import compute_string_similarity, match_steps
from processlens.models.process import ProcessModel, Step


@dataclass
class FailureInstance:
    category: str
    sop_title: str
    description: str
    gold_detail: Optional[str] = None
    pred_detail: Optional[str] = None


@dataclass
class ErrorAnalysisReport:
    total_documents: int
    total_gold_steps: int
    total_pred_steps: int
    missed_step_count: int
    hallucinated_step_count: int
    wrong_order_count: int
    wrong_actor_count: int
    failure_breakdown_percentages: dict[str, float] = field(default_factory=dict)
    sample_failures: list[FailureInstance] = field(default_factory=list)
    top_failure_category: str = "none"
    recommended_prompt_adjustments: list[str] = field(default_factory=list)


class ErrorAnalyzer:
    def __init__(self, step_similarity_threshold: float = 0.55):
        self.step_similarity_threshold = step_similarity_threshold

    def analyze_pair(
        self,
        gold: ProcessModel,
        pred: Optional[ProcessModel],
        sop_title: str,
    ) -> list[FailureInstance]:
        failures: list[FailureInstance] = []

        if pred is None:
            for g in gold.steps:
                failures.append(
                    FailureInstance(
                        category="missed_step",
                        sop_title=sop_title,
                        description=f"Extraction failed entirely; missed step: '{g.name}'",
                        gold_detail=f"[{g.id}] {g.name} (Actor: {g.actor})",
                    )
                )
            return failures

        gold_to_pred, _ = match_steps(
            gold.steps, pred.steps, threshold=self.step_similarity_threshold
        )
        pred_to_gold = {v: k for k, v in gold_to_pred.items()}
        gold_map = {g.id: g for g in gold.steps}
        pred_map = {p.id: p for p in pred.steps}

        for g in gold.steps:
            if g.id not in gold_to_pred:
                failures.append(
                    FailureInstance(
                        category="missed_step",
                        sop_title=sop_title,
                        description=f"Gold step '{g.name}' not found in predictions",
                        gold_detail=f"[{g.id}] {g.name} (Actor: {g.actor})",
                    )
                )

        for p in pred.steps:
            if p.id not in pred_to_gold:
                failures.append(
                    FailureInstance(
                        category="hallucinated_step",
                        sop_title=sop_title,
                        description=f"Predicted step '{p.name}' has no basis in gold model",
                        pred_detail=f"[{p.id}] {p.name} (Actor: {p.actor})",
                    )
                )

        for g_id, p_id in gold_to_pred.items():
            g_step = gold_map[g_id]
            p_step = pred_map[p_id]
            g_act = g_step.actor.lower().strip()
            p_act = p_step.actor.lower().strip()
            if g_act != p_act and compute_string_similarity(g_act, p_act) <= 0.7:
                failures.append(
                    FailureInstance(
                        category="wrong_actor",
                        sop_title=sop_title,
                        description=f"Mismatched actor on step '{g_step.name}'",
                        gold_detail=f"Expected: '{g_step.actor}'",
                        pred_detail=f"Predicted: '{p_step.actor}'",
                    )
                )

        gold_edges = set()
        for g in gold.steps:
            for nxt in g.next_step_ids:
                gold_edges.add((g.id, nxt))

        pred_edges = set()
        for p in pred.steps:
            for nxt in p.next_step_ids:
                pred_edges.add((p.id, nxt))

        for g_u, g_v in gold_edges:
            if g_u in gold_to_pred and g_v in gold_to_pred:
                expected_pred_edge = (gold_to_pred[g_u], gold_to_pred[g_v])
                if expected_pred_edge not in pred_edges:
                    failures.append(
                        FailureInstance(
                            category="wrong_order",
                            sop_title=sop_title,
                            description=f"Missing dependency edge from '{gold_map[g_u].name}' -> '{gold_map[g_v].name}'",
                            gold_detail=f"{g_u} -> {g_v}",
                            pred_detail=f"Edge {expected_pred_edge} missing in prediction",
                        )
                    )

        return failures

    def analyze_dataset(
        self,
        pairs: list[tuple[ProcessModel, Optional[ProcessModel], str]],
    ) -> ErrorAnalysisReport:
        all_failures: list[FailureInstance] = []
        total_gold_steps = 0
        total_pred_steps = 0

        for gold, pred, title in pairs:
            total_gold_steps += len(gold.steps)
            if pred is not None:
                total_pred_steps += len(pred.steps)
            fails = self.analyze_pair(gold, pred, title)
            all_failures.extend(fails)

        missed = sum(1 for f in all_failures if f.category == "missed_step")
        hallucinated = sum(1 for f in all_failures if f.category == "hallucinated_step")
        wrong_order = sum(1 for f in all_failures if f.category == "wrong_order")
        wrong_actor = sum(1 for f in all_failures if f.category == "wrong_actor")
        total_errors = len(all_failures)

        breakdown = {}
        if total_errors > 0:
            breakdown = {
                "missed_step": round(missed / total_errors * 100, 1),
                "hallucinated_step": round(hallucinated / total_errors * 100, 1),
                "wrong_order": round(wrong_order / total_errors * 100, 1),
                "wrong_actor": round(wrong_actor / total_errors * 100, 1),
            }

        counts = {
            "missed_step": missed,
            "hallucinated_step": hallucinated,
            "wrong_order": wrong_order,
            "wrong_actor": wrong_actor,
        }
        top_cat = max(counts, key=counts.get) if total_errors > 0 else "none"

        recommendations = []
        if top_cat == "missed_step":
            recommendations.append("Enhance prompt instructions to enforce capturing every granular handoff and decision step.")
            recommendations.append("Add chunk-level step density verification in extraction agent.")
        elif top_cat == "hallucinated_step":
            recommendations.append("Strengthen negative constraints against extrapolating unmentioned systems or actors.")
        elif top_cat == "wrong_order":
            recommendations.append("Prompt the LLM to trace explicit temporal signals (after, upon receipt, subsequently, then).")
        elif top_cat == "wrong_actor":
            recommendations.append("Instruct model to map actors strictly to named titles in Section 2 (Scope & Roles).")

        return ErrorAnalysisReport(
            total_documents=len(pairs),
            total_gold_steps=total_gold_steps,
            total_pred_steps=total_pred_steps,
            missed_step_count=missed,
            hallucinated_step_count=hallucinated,
            wrong_order_count=wrong_order,
            wrong_actor_count=wrong_actor,
            failure_breakdown_percentages=breakdown,
            sample_failures=all_failures[:15],
            top_failure_category=top_cat,
            recommended_prompt_adjustments=recommendations,
        )

    def write_markdown_report(
        self,
        report: ErrorAnalysisReport,
        output_file: Path,
    ) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# ProcessLens Error Analysis & Failure Categorization Report",
            "",
            "## Summary of Evaluation Failures",
            f"- **Evaluated Documents**: {report.total_documents}",
            f"- **Total Gold Steps**: {report.total_gold_steps}",
            f"- **Total Predicted Steps**: {report.total_pred_steps}",
            f"- **Primary Failure Category**: `{report.top_failure_category.upper()}`",
            "",
            "| Failure Category | Error Count | Percentage |",
            "|---|---|---|",
            f"| Missed Steps | {report.missed_step_count} | {report.failure_breakdown_percentages.get('missed_step', 0)}% |",
            f"| Hallucinated Steps | {report.hallucinated_step_count} | {report.failure_breakdown_percentages.get('hallucinated_step', 0)}% |",
            f"| Wrong Ordering / Transitions | {report.wrong_order_count} | {report.failure_breakdown_percentages.get('wrong_order', 0)}% |",
            f"| Wrong Responsible Actor | {report.wrong_actor_count} | {report.failure_breakdown_percentages.get('wrong_actor', 0)}% |",
            "",
            "## Targeted Prompt & Schema Improvements",
        ]

        for rec in report.recommended_prompt_adjustments:
            lines.append(f"- {rec}")

        lines.extend([
            "",
            "## Concrete Failure Examples",
        ])

        for idx, fail in enumerate(report.sample_failures, 1):
            lines.append(f"### {idx}. [{fail.category.upper()}] in {fail.sop_title}")
            lines.append(f"- **Description**: {fail.description}")
            if fail.gold_detail:
                lines.append(f"- **Gold Reference**: {fail.gold_detail}")
            if fail.pred_detail:
                lines.append(f"- **Prediction Detail**: {fail.pred_detail}")
            lines.append("")

        output_file.write_text("\n".join(lines), encoding="utf-8")
