from pathlib import Path
import pytest
from processlens.eval.error_analyzer import ErrorAnalyzer
from processlens.models.process import ProcessModel, Step


@pytest.fixture
def base_gold_model():
    return ProcessModel(
        title="Sample Process",
        actors=["Clerk", "Manager"],
        steps=[
            Step(
                id="s1",
                name="Submit Request",
                actor="Clerk",
                next_step_ids=["s2"],
                source_chunk_ids=["chunk_0"],
            ),
            Step(
                id="s2",
                name="Approve Request",
                actor="Manager",
                next_step_ids=[],
                source_chunk_ids=["chunk_1"],
            ),
        ],
    )


def test_categorize_missed_step(base_gold_model):
    pred = ProcessModel(
        title="Pred Process",
        actors=["Clerk"],
        steps=[
            Step(
                id="p1",
                name="Submit Request",
                actor="Clerk",
                next_step_ids=[],
                source_chunk_ids=["chunk_0"],
            )
        ],
    )
    analyzer = ErrorAnalyzer()
    failures = analyzer.analyze_pair(base_gold_model, pred, "Test SOP")
    categories = [f.category for f in failures]
    assert "missed_step" in categories


def test_categorize_hallucinated_step(base_gold_model):
    pred = ProcessModel(
        title="Pred Process",
        actors=["Clerk", "Manager", "Auditor"],
        steps=[
            Step(
                id="p1",
                name="Submit Request",
                actor="Clerk",
                next_step_ids=["p2"],
                source_chunk_ids=["chunk_0"],
            ),
            Step(
                id="p2",
                name="Approve Request",
                actor="Manager",
                next_step_ids=["p3"],
                source_chunk_ids=["chunk_1"],
            ),
            Step(
                id="p3",
                name="Invented Audit Step",
                actor="Auditor",
                next_step_ids=[],
                source_chunk_ids=["chunk_1"],
            ),
        ],
    )
    analyzer = ErrorAnalyzer()
    failures = analyzer.analyze_pair(base_gold_model, pred, "Test SOP")
    categories = [f.category for f in failures]
    assert "hallucinated_step" in categories


def test_categorize_wrong_actor(base_gold_model):
    pred = ProcessModel(
        title="Pred Process",
        actors=["Contractor", "Manager"],
        steps=[
            Step(
                id="p1",
                name="Submit Request",
                actor="Contractor",
                next_step_ids=["p2"],
                source_chunk_ids=["chunk_0"],
            ),
            Step(
                id="p2",
                name="Approve Request",
                actor="Manager",
                next_step_ids=[],
                source_chunk_ids=["chunk_1"],
            ),
        ],
    )
    analyzer = ErrorAnalyzer()
    failures = analyzer.analyze_pair(base_gold_model, pred, "Test SOP")
    categories = [f.category for f in failures]
    assert "wrong_actor" in categories


def test_write_markdown_report(tmp_path, base_gold_model):
    analyzer = ErrorAnalyzer()
    report = analyzer.analyze_dataset([(base_gold_model, None, "Fail SOP")])
    out_file = tmp_path / "report.md"
    analyzer.write_markdown_report(report, out_file)

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "ProcessLens Error Analysis" in content
    assert "Missed Steps" in content
