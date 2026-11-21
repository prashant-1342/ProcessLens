import json
from pathlib import Path
import pytest
from processlens.eval.calibration import compute_ece, generate_reliability_diagram
from processlens.eval.runner import BenchmarkRunner, compute_bootstrap_ci
from processlens.eval.scorers import (
    evaluate_actors,
    evaluate_citations,
    evaluate_ordering,
    evaluate_process_model,
    match_steps,
)
from processlens.extractor.agent import ExtractionAgent
from processlens.models.process import ProcessModel, Step


@pytest.fixture
def sample_gold_model():
    s1 = Step(
        id="g1",
        name="Submit travel claim",
        actor="Employee",
        next_step_ids=["g2"],
        source_chunk_ids=["chunk_0"],
        confidence=1.0,
    )
    s2 = Step(
        id="g2",
        name="Manager approval",
        actor="Line Manager",
        next_step_ids=["g3"],
        source_chunk_ids=["chunk_1"],
        confidence=0.95,
    )
    s3 = Step(
        id="g3",
        name="Disburse reimbursement",
        actor="Finance Clerk",
        next_step_ids=[],
        source_chunk_ids=["chunk_2"],
        confidence=0.98,
    )
    return ProcessModel(
        title="Travel Reimbursement",
        actors=["Employee", "Line Manager", "Finance Clerk"],
        steps=[s1, s2, s3],
    )


@pytest.fixture
def sample_pred_model():
    p1 = Step(
        id="p1",
        name="Submit travel claim request",
        actor="Employee",
        next_step_ids=["p2"],
        source_chunk_ids=["chunk_0"],
        confidence=0.92,
    )
    p2 = Step(
        id="p2",
        name="Manager approval gate",
        actor="Line Manager",
        next_step_ids=["p3"],
        source_chunk_ids=["chunk_1"],
        confidence=0.88,
    )
    p3 = Step(
        id="p3",
        name="Disburse reimbursement payment",
        actor="Finance Clerk",
        next_step_ids=[],
        source_chunk_ids=["chunk_2"],
        confidence=0.95,
    )
    return ProcessModel(
        title="Travel Reimbursement",
        actors=["Employee", "Line Manager", "Finance Clerk"],
        steps=[p1, p2, p3],
    )


def test_step_matching_and_f1(sample_gold_model, sample_pred_model):
    gold_to_pred, matches = match_steps(sample_gold_model.steps, sample_pred_model.steps)
    assert len(gold_to_pred) == 3
    assert gold_to_pred["g1"] == "p1"
    assert gold_to_pred["g2"] == "p2"
    assert gold_to_pred["g3"] == "p3"

    score = evaluate_process_model(sample_gold_model, sample_pred_model)
    assert score.step_precision == 1.0
    assert score.step_recall == 1.0
    assert score.step_f1 == 1.0
    assert score.ordering_accuracy == 1.0
    assert score.actor_accuracy == 1.0
    assert score.schema_valid_rate == 1.0
    assert score.citation_support_rate == 1.0


def test_partial_step_matching(sample_gold_model):
    p1 = Step(
        id="p1",
        name="Submit travel claim request",
        actor="Employee",
        next_step_ids=[],
        source_chunk_ids=["chunk_0"],
    )
    partial_pred = ProcessModel(title="Partial", actors=["Employee"], steps=[p1])

    score = evaluate_process_model(sample_gold_model, partial_pred)
    assert score.step_precision == 1.0
    assert score.step_recall == round(1.0 / 3.0, 4)
    assert score.matched_step_count == 1


def test_ordering_mismatch(sample_gold_model):
    p1 = Step(
        id="p1",
        name="Submit travel claim",
        actor="Employee",
        next_step_ids=["p3"],
        source_chunk_ids=["chunk_0"],
    )
    p2 = Step(
        id="p2",
        name="Manager approval",
        actor="Line Manager",
        next_step_ids=[],
        source_chunk_ids=["chunk_1"],
    )
    p3 = Step(
        id="p3",
        name="Disburse reimbursement",
        actor="Finance Clerk",
        next_step_ids=["p2"],
        source_chunk_ids=["chunk_2"],
    )
    mismatched_pred = ProcessModel(
        title="Mismatched Order",
        actors=["Employee", "Line Manager", "Finance Clerk"],
        steps=[p1, p3, p2],
    )

    gold_to_pred, _ = match_steps(sample_gold_model.steps, mismatched_pred.steps)
    ordering_acc = evaluate_ordering(sample_gold_model.steps, mismatched_pred.steps, gold_to_pred)
    assert ordering_acc < 1.0


def test_actor_mismatch(sample_gold_model):
    p1 = Step(
        id="p1",
        name="Submit travel claim",
        actor="Customer",
        next_step_ids=["p2"],
        source_chunk_ids=["chunk_0"],
    )
    p2 = Step(
        id="p2",
        name="Manager approval",
        actor="Line Manager",
        next_step_ids=[],
        source_chunk_ids=["chunk_1"],
    )
    pred = ProcessModel(
        title="Actor Mismatch",
        actors=["Customer", "Line Manager"],
        steps=[p1, p2],
    )
    gold_to_pred, _ = match_steps(sample_gold_model.steps, pred.steps)
    actor_acc = evaluate_actors(sample_gold_model.steps, pred.steps, gold_to_pred)
    assert actor_acc == 0.5


def test_calibration_ece_computation():
    confidences = [0.95, 0.90, 0.85, 0.80, 0.20, 0.10]
    accuracies = [1, 1, 1, 1, 0, 0]
    ece, bin_data = compute_ece(confidences, accuracies, num_bins=5)
    assert ece < 0.20
    assert len(bin_data) == 5


def test_bootstrap_ci():
    values = [0.80, 0.82, 0.85, 0.84, 0.86, 0.83, 0.81]
    mean, ci_low, ci_high = compute_bootstrap_ci(values, num_bootstrap=200)
    assert ci_low <= mean <= ci_high
    assert 0.79 <= ci_low
    assert ci_high <= 0.88


def test_benchmark_runner_flow(tmp_path):
    runner = BenchmarkRunner(output_dir=tmp_path)

    gold_model = ProcessModel(
        title="Mock Process",
        actors=["Agent"],
        steps=[
            Step(
                id="s1",
                name="Action 1",
                actor="Agent",
                next_step_ids=[],
                source_chunk_ids=["chunk_0"],
                confidence=0.99,
            )
        ],
    )

    class MockAgent:
        def __init__(self, **kwargs):
            pass

        def extract(self, **kwargs):
            from processlens.extractor.agent import ExtractionResult
            return ExtractionResult(
                process_model=gold_model,
                attempts=1,
                success=True,
                chunks=[],
            )

    results = runner.run_benchmark(
        configurations=[{"name": "test_cfg", "prompt_version": "v1", "use_rag": False, "runs": 1}],
        agent_factory=lambda **kw: MockAgent(),
        max_samples=2,
    )

    assert len(results) == 1
    assert results[0].step_f1_mean >= 0.0
    assert (tmp_path / "benchmark_results.csv").exists()
    assert (tmp_path / "benchmark_results.md").exists()
