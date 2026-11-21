from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional
from processlens.models.process import ProcessModel, Step


@dataclass
class EvaluationScore:
    step_precision: float
    step_recall: float
    step_f1: float
    ordering_accuracy: float
    actor_accuracy: float
    schema_valid_rate: float
    citation_support_rate: float
    retrieval_recall_at_k: float
    matched_step_count: int
    gold_step_count: int
    pred_step_count: int
    step_confidences: list[float] = field(default_factory=list)
    step_correctness: list[int] = field(default_factory=list)


def compute_string_similarity(str1: str, str2: str) -> float:
    return SequenceMatcher(None, str1.lower().strip(), str2.lower().strip()).ratio()


def match_steps(
    gold_steps: list[Step],
    pred_steps: list[Step],
    threshold: float = 0.55,
) -> tuple[dict[str, str], list[tuple[str, str, float]]]:
    candidates = []
    for g in gold_steps:
        for p in pred_steps:
            sim = compute_string_similarity(g.name, p.name)
            if sim >= threshold:
                candidates.append((sim, g.id, p.id))

    candidates.sort(key=lambda x: x[0], reverse=True)

    gold_to_pred: dict[str, str] = {}
    matched_pred = set()
    matches_info = []

    for sim, g_id, p_id in candidates:
        if g_id not in gold_to_pred and p_id not in matched_pred:
            gold_to_pred[g_id] = p_id
            matched_pred.add(p_id)
            matches_info.append((g_id, p_id, sim))

    return gold_to_pred, matches_info


def evaluate_ordering(
    gold_steps: list[Step],
    pred_steps: list[Step],
    gold_to_pred: dict[str, str],
) -> float:
    gold_edges = set()
    for g in gold_steps:
        for nxt in g.next_step_ids:
            gold_edges.add((g.id, nxt))

    if not gold_edges:
        return 1.0

    pred_edges = set()
    for p in pred_steps:
        for nxt in p.next_step_ids:
            pred_edges.add((p.id, nxt))

    matched_edges = 0
    evaluable_edges = 0

    for g_u, g_v in gold_edges:
        if g_u in gold_to_pred and g_v in gold_to_pred:
            evaluable_edges += 1
            expected_pred_edge = (gold_to_pred[g_u], gold_to_pred[g_v])
            if expected_pred_edge in pred_edges:
                matched_edges += 1

    if evaluable_edges == 0:
        return 0.0

    return matched_edges / evaluable_edges


def evaluate_actors(
    gold_steps: list[Step],
    pred_steps: list[Step],
    gold_to_pred: dict[str, str],
) -> float:
    if not gold_to_pred:
        return 0.0

    gold_map = {g.id: g for g in gold_steps}
    pred_map = {p.id: p for p in pred_steps}

    correct = 0
    for g_id, p_id in gold_to_pred.items():
        g_actor = gold_map[g_id].actor.lower().strip()
        p_actor = pred_map[p_id].actor.lower().strip()
        if g_actor == p_actor or compute_string_similarity(g_actor, p_actor) > 0.7:
            correct += 1

    return correct / len(gold_to_pred)


def evaluate_citations(
    pred_steps: list[Step],
    valid_chunk_ids: set[str],
) -> float:
    if not pred_steps:
        return 0.0

    supported = 0
    clean_valid = {cid.lower().strip() for cid in valid_chunk_ids}
    for p in pred_steps:
        p_chunks = {cid.lower().strip() for cid in p.source_chunk_ids}
        if p_chunks and (p_chunks.issubset(clean_valid) or any(c in clean_valid for c in p_chunks)):
            supported += 1

    return supported / len(pred_steps)


def evaluate_retrieval(
    gold_steps: list[Step],
    retrieved_chunk_ids: list[str],
) -> float:
    gold_chunks = set()
    for g in gold_steps:
        for cid in g.source_chunk_ids:
            gold_chunks.add(cid.lower().strip())

    if not gold_chunks:
        return 1.0

    retrieved_set = {cid.lower().strip() for cid in retrieved_chunk_ids}
    found = gold_chunks.intersection(retrieved_set)
    return len(found) / len(gold_chunks)


def evaluate_process_model(
    gold: ProcessModel,
    pred: Optional[ProcessModel],
    valid_chunk_ids: Optional[set[str]] = None,
    retrieved_chunk_ids: Optional[list[str]] = None,
) -> EvaluationScore:
    if pred is None:
        return EvaluationScore(
            step_precision=0.0,
            step_recall=0.0,
            step_f1=0.0,
            ordering_accuracy=0.0,
            actor_accuracy=0.0,
            schema_valid_rate=0.0,
            citation_support_rate=0.0,
            retrieval_recall_at_k=0.0,
            matched_step_count=0,
            gold_step_count=len(gold.steps),
            pred_step_count=0,
        )

    gold_to_pred, _ = match_steps(gold.steps, pred.steps)
    matched_count = len(gold_to_pred)
    gold_count = len(gold.steps)
    pred_count = len(pred.steps)

    prec = matched_count / pred_count if pred_count > 0 else 0.0
    rec = matched_count / gold_count if gold_count > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0.0 else 0.0

    ordering_acc = evaluate_ordering(gold.steps, pred.steps, gold_to_pred)
    actor_acc = evaluate_actors(gold.steps, pred.steps, gold_to_pred)

    if valid_chunk_ids is None:
        valid_chunk_ids = set()
        for g in gold.steps:
            valid_chunk_ids.update(g.source_chunk_ids)

    citation_rate = evaluate_citations(pred.steps, valid_chunk_ids)

    ret_recall = 1.0
    if retrieved_chunk_ids is not None:
        ret_recall = evaluate_retrieval(gold.steps, retrieved_chunk_ids)

    gold_map = {g.id: g for g in gold.steps}
    pred_map = {p.id: p for p in pred.steps}
    confidences = []
    correctness = []

    for p in pred.steps:
        confidences.append(float(p.confidence))
        matched_gold_id = next((g_id for g_id, pid in gold_to_pred.items() if pid == p.id), None)
        if matched_gold_id is not None:
            g_act = gold_map[matched_gold_id].actor.lower().strip()
            p_act = p.actor.lower().strip()
            is_actor_correct = (g_act == p_act or compute_string_similarity(g_act, p_act) > 0.7)
            correctness.append(1 if is_actor_correct else 0)
        else:
            correctness.append(0)

    return EvaluationScore(
        step_precision=round(prec, 4),
        step_recall=round(rec, 4),
        step_f1=round(f1, 4),
        ordering_accuracy=round(ordering_acc, 4),
        actor_accuracy=round(actor_acc, 4),
        schema_valid_rate=1.0,
        citation_support_rate=round(citation_rate, 4),
        retrieval_recall_at_k=round(ret_recall, 4),
        matched_step_count=matched_count,
        gold_step_count=gold_count,
        pred_step_count=pred_count,
        step_confidences=confidences,
        step_correctness=correctness,
    )
