import json
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from processlens.api.app import app
from processlens.core.llm import LLMClient
from processlens.models.process import ProcessModel, Step

client = TestClient(app)

MOCK_EXTRACTED_JSON = json.dumps({
    "title": "Onboarding Workflow",
    "version": "1.0",
    "actors": ["HR", "IT"],
    "steps": [
        {
            "id": "step_1",
            "name": "Prepare Contract",
            "actor": "HR",
            "next_step_ids": ["step_2"],
            "source_chunk_ids": ["chunk_0"],
            "confidence": 0.99,
            "is_loop": False,
        },
        {
            "id": "step_2",
            "name": "Configure Workstation",
            "actor": "IT",
            "next_step_ids": [],
            "source_chunk_ids": ["chunk_1"],
            "confidence": 0.96,
            "is_loop": False,
        },
    ],
})


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ProcessLens"


def test_extract_endpoint_success():
    mock_llm = LLMClient(mock_response=MOCK_EXTRACTED_JSON)

    with patch("processlens.extractor.agent.get_llm_client", return_value=mock_llm):
        payload = {
            "document_text": "STANDARD OPERATING PROCEDURE: ONBOARDING\n[Chunk 0: HR]\nHR prepares contract.\n[Chunk 1: IT]\nIT configures computer.",
            "document_title": "Onboarding Workflow",
            "use_rag": False,
            "prompt_version": "v3",
        }
        response = client.post("/extract", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model_id"] is not None
        assert data["process_model"]["title"] == "Onboarding Workflow"
        assert len(data["process_model"]["steps"]) == 2

        model_id = data["model_id"]
        get_res = client.get(f"/models/{model_id}")
        assert get_res.status_code == 200
        get_data = get_res.json()
        assert get_data["id"] == model_id
        assert get_data["title"] == "Onboarding Workflow"


def test_get_non_existent_model():
    response = client.get("/models/non_existent_id_12345")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_list_models_endpoint():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
