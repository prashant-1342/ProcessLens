import argparse
import json
import sys
from pathlib import Path
from processlens.models.process import ProcessModel


def print_separator(char="=", length=90):
    print(char * length)


def review_single(sop_id: str, docs_dir: Path, gold_dir: Path):
    doc_path = docs_dir / f"{sop_id}.txt"
    gold_path = gold_dir / f"{sop_id}.json"

    if not doc_path.exists():
        print(f"Error: Document not found: {doc_path}")
        return
    if not gold_path.exists():
        print(f"Error: Gold label not found: {gold_path}")
        return

    with open(doc_path, "r", encoding="utf-8") as f:
        doc_text = f.read()

    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    model = ProcessModel.model_validate(gold_data)

    print_separator("=")
    print(f"SOP REVIEW: {sop_id.upper()}")
    print(f"TITLE:      {model.title}")
    print(f"STATUS:     {model.metadata.get('review_status', 'UNKNOWN')} [Awaiting Human Verification]")
    print_separator("=")

    print("\n--- SOURCE DOCUMENT CONTENT ---")
    for line in doc_text.strip().split("\n"):
        print(f"  | {line}")

    print("\n--- EXTRACTED GOLD LABELS (VALIDATED PROCESS MODEL) ---")
    print(f"Process Title:   {model.title}")
    print(f"Version:         {model.version}")
    print(f"Actors:          {', '.join(model.actors)}")
    print(f"Number of Steps: {len(model.steps)}")
    print("\nSteps & Execution Logic:")

    for idx, step in enumerate(model.steps, start=1):
        decision_str = " [DECISION]" if step.is_decision else ""
        loop_str = " [LOOP MARKER]" if step.is_loop else ""
        print(f"\n  {idx}. [{step.id}] {step.name}{decision_str}{loop_str}")
        print(f"     Actor:       {step.actor}")
        print(f"     Description: {step.description}")
        print(f"     Inputs:      {step.inputs}")
        print(f"     Outputs:     {step.outputs}")
        print(f"     Next Steps:  {step.next_step_ids}")
        if step.conditions:
            print(f"     Conditions:  {step.conditions}")
        print(f"     Citations:   {step.source_chunk_ids}")
        print(f"     Confidence:  {step.confidence}")

    print_separator("-")
    print("MANUAL VERIFICATION CHECKLIST:")
    print("  [ ] All expected steps present and in correct order")
    print("  [ ] Responsible actors accurately assigned")
    print("  [ ] Decision points and branches verified")
    print("  [ ] Source chunk citations accurately reference document text")
    print_separator("=")


def list_all(docs_dir: Path, gold_dir: Path):
    gold_files = sorted(list(gold_dir.glob("*.json")))
    print_separator("=")
    print(f"{'SOP ID':<35} | {'STEPS':<6} | {'ACTORS':<6} | {'STATUS':<22}")
    print_separator("-")
    for gf in gold_files:
        sop_id = gf.stem
        with open(gf, "r", encoding="utf-8") as f:
            data = json.load(f)
        status = data.get("metadata", {}).get("review_status", "UNKNOWN")
        steps_count = len(data.get("steps", []))
        actors_count = len(data.get("actors", []))
        print(f"{sop_id:<35} | {steps_count:<6} | {actors_count:<6} | {status:<22}")
    print_separator("=")
    print(f"Total: {len(gold_files)} documents configured for review.\n")


def main():
    parser = argparse.ArgumentParser(description="ProcessLens SOP Dataset Review Tool")
    parser.add_argument("--id", type=str, help="Specific SOP ID to review (e.g. sop_01_invoice_approval)")
    parser.add_argument("--all", action="store_true", help="Print review details for all 25 documents")
    parser.add_argument("--list", action="store_true", help="List all documents and their review status")

    args = parser.parse_args()

    docs_dir = Path("data/documents")
    gold_dir = Path("data/gold_labels")

    if args.list or (not args.id and not args.all):
        list_all(docs_dir, gold_dir)
        if not args.id and not args.all:
            print("Tip: Use '--id <sop_id>' to inspect a single SOP, or '--all' to inspect all.\n")
            return

    if args.id:
        review_single(args.id, docs_dir, gold_dir)
    elif args.all:
        gold_files = sorted(list(gold_dir.glob("*.json")))
        for gf in gold_files:
            review_single(gf.stem, docs_dir, gold_dir)
            print("\n")


if __name__ == "__main__":
    main()
