import json
import subprocess
import sys
from pathlib import Path
import pytest
from processlens.models.process import ProcessModel

DOCS_DIR = Path("data/documents")
GOLD_DIR = Path("data/gold_labels")


def test_documents_and_gold_labels_count():
    doc_files = sorted(list(DOCS_DIR.glob("*.txt")))
    gold_files = sorted(list(GOLD_DIR.glob("*.json")))

    assert len(doc_files) == 25
    assert len(gold_files) == 25

    doc_stems = {f.stem for f in doc_files}
    gold_stems = {f.stem for f in gold_files}
    assert doc_stems == gold_stems


def test_documents_content_integrity():
    for doc_file in DOCS_DIR.glob("*.txt"):
        text = doc_file.read_text(encoding="utf-8")
        assert len(text.strip()) > 300
        assert "STANDARD OPERATING PROCEDURE" in text
        assert "Chunk 0" in text


def test_gold_labels_schema_and_graph_semantics():
    for gold_file in GOLD_DIR.glob("*.json"):
        with open(gold_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        model = ProcessModel.model_validate(data)
        assert model.title
        assert len(model.steps) >= 3
        assert len(model.actors) >= 2

        known_step_ids = {s.id for s in model.steps}
        for step in model.steps:
            assert step.actor in model.actors
            for nxt in step.next_step_ids:
                assert nxt in known_step_ids


def test_gold_labels_manual_review_status():
    for gold_file in GOLD_DIR.glob("*.json"):
        with open(gold_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert data["metadata"].get("review_status") == "NEEDS_MANUAL_REVIEW"
        assert data["metadata"].get("reviewer") is None


def test_gold_label_citation_chunks():
    for gold_file in GOLD_DIR.glob("*.json"):
        sop_id = gold_file.stem
        doc_file = DOCS_DIR / f"{sop_id}.txt"
        doc_text = doc_file.read_text(encoding="utf-8").lower()

        with open(gold_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for step in data.get("steps", []):
            assert len(step.get("source_chunk_ids", [])) > 0
            for chunk_id in step["source_chunk_ids"]:
                chunk_num = chunk_id.replace("chunk_", "")
                assert f"chunk {chunk_num}" in doc_text


def test_review_tool_execution():
    cmd = [
        sys.executable,
        "scripts/review_dataset.py",
        "--id",
        "sop_01_invoice_approval",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "."},
    )
    assert result.returncode == 0
    assert "SOP REVIEW: SOP_01_INVOICE_APPROVAL" in result.stdout
    assert "NEEDS_MANUAL_REVIEW" in result.stdout
    assert "Accounts Payable Clerk" in result.stdout
