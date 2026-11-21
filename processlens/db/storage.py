import json
import uuid
from typing import Any, Optional
from processlens.db.models import ProcessModelRecord
from processlens.db.session import get_db_session
from processlens.models.process import ProcessModel

_memory_models: dict[str, dict[str, Any]] = {}


def save_model(model: ProcessModel, document_id: Optional[str] = None) -> str:
    model_id = model.id or f"pm_{uuid.uuid4().hex[:12]}"
    model.id = model_id
    model_json = model.model_dump_json(indent=2)

    try:
        with get_db_session() as session:
            record = ProcessModelRecord(
                id=model_id,
                document_id=document_id,
                title=model.title,
                version=model.version,
                model_json=model_json,
            )
            session.merge(record)
    except Exception:
        pass

    _memory_models[model_id] = {
        "id": model_id,
        "document_id": document_id,
        "title": model.title,
        "version": model.version,
        "model_json": model_json,
    }
    return model_id


def get_model(model_id: str) -> Optional[ProcessModel]:
    try:
        with get_db_session() as session:
            record = session.query(ProcessModelRecord).filter(ProcessModelRecord.id == model_id).first()
            if record:
                return ProcessModel.model_validate_json(record.model_json)
    except Exception:
        pass

    entry = _memory_models.get(model_id)
    if entry:
        return ProcessModel.model_validate_json(entry["model_json"])

    return None


def list_models() -> list[dict[str, Any]]:
    results = []
    try:
        with get_db_session() as session:
            records = session.query(ProcessModelRecord).all()
            for r in records:
                results.append({
                    "id": r.id,
                    "title": r.title,
                    "version": r.version,
                    "document_id": r.document_id,
                })
            if results:
                return results
    except Exception:
        pass

    return [
        {
            "id": v["id"],
            "title": v["title"],
            "version": v["version"],
            "document_id": v["document_id"],
        }
        for v in _memory_models.values()
    ]
