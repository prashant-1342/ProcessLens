import json
from typing import Type
from pydantic import BaseModel
import pytest
from processlens.core.chunker import chunk_document
from processlens.core.llm import LLMClient
from processlens.extractor.agent import ExtractionAgent
from processlens.models.process import ProcessModel


class SequenceMockLLM(LLMClient):
    def __init__(self, responses: list[str]):
        super().__init__(mock_response=None)
        self.responses = list(responses)
        self.call_count = 0
        self.prompts_received = []

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        system_prompt: str = None,
        model: str = None,
        temperature: float = 0.0,
    ) -> BaseModel:
        self.prompts_received.append(prompt)
        if self.call_count < len(self.responses):
            raw = self.responses[self.call_count]
            self.call_count += 1
        else:
            raw = self.responses[-1]

        data = json.loads(raw)
        return response_model.model_validate(data)


SAMPLE_DOC = """STANDARD OPERATING PROCEDURE: REFUND
[Chunk 0: Intake]
Customer requests a refund. Support agent reviews ticket.
[Chunk 1: Decision]
If purchase under 30 days, billing dispatches refund; otherwise reject."""

VALID_MODEL_JSON = json.dumps({
    "title": "Refund Process",
    "version": "1.0",
    "actors": ["Customer", "Support Agent", "Billing Specialist"],
    "steps": [
        {
            "id": "step_1",
            "name": "Customer requests refund",
            "description": "Customer submits ticket",
            "actor": "Customer",
            "inputs": [],
            "outputs": ["Ticket"],
            "next_step_ids": ["step_2"],
            "is_decision": False,
            "conditions": [],
            "source_chunk_ids": ["chunk_0"],
            "confidence": 0.99,
            "is_loop": False,
        },
        {
            "id": "step_2",
            "name": "Review eligibility",
            "description": "Agent evaluates purchase date",
            "actor": "Support Agent",
            "inputs": ["Ticket"],
            "outputs": ["Decision"],
            "next_step_ids": ["step_3"],
            "is_decision": True,
            "conditions": ["Under 30 days"],
            "source_chunk_ids": ["chunk_1"],
            "confidence": 0.95,
            "is_loop": False,
        },
        {
            "id": "step_3",
            "name": "Disburse refund",
            "description": "Billing reverses charge",
            "actor": "Billing Specialist",
            "inputs": ["Decision"],
            "outputs": ["Receipt"],
            "next_step_ids": [],
            "is_decision": False,
            "conditions": [],
            "source_chunk_ids": ["chunk_1"],
            "confidence": 0.98,
            "is_loop": False,
        },
    ],
})

INVALID_UNMARKED_CYCLE_JSON = json.dumps({
    "title": "Invalid Cycle Process",
    "version": "1.0",
    "actors": ["Author", "Reviewer"],
    "steps": [
        {
            "id": "s1",
            "name": "Draft",
            "actor": "Author",
            "next_step_ids": ["s2"],
            "is_decision": False,
            "conditions": [],
            "source_chunk_ids": ["chunk_0"],
            "confidence": 0.9,
            "is_loop": False,
        },
        {
            "id": "s2",
            "name": "Review",
            "actor": "Reviewer",
            "next_step_ids": ["s1"],
            "is_decision": False,
            "conditions": [],
            "source_chunk_ids": ["chunk_1"],
            "confidence": 0.9,
            "is_loop": False,
        },
    ],
})

CORRECTED_CYCLE_JSON = json.dumps({
    "title": "Corrected Cycle Process",
    "version": "1.0",
    "actors": ["Author", "Reviewer"],
    "steps": [
        {
            "id": "s1",
            "name": "Draft",
            "actor": "Author",
            "next_step_ids": ["s2"],
            "is_decision": False,
            "conditions": [],
            "source_chunk_ids": ["chunk_0"],
            "confidence": 0.9,
            "is_loop": False,
        },
        {
            "id": "s2",
            "name": "Review",
            "actor": "Reviewer",
            "next_step_ids": ["s1"],
            "is_decision": True,
            "conditions": ["Needs revision"],
            "source_chunk_ids": ["chunk_1"],
            "confidence": 0.95,
            "is_loop": True,
        },
    ],
})


def test_extraction_success_first_attempt():
    mock_llm = SequenceMockLLM([VALID_MODEL_JSON])
    agent = ExtractionAgent(llm_client=mock_llm, prompt_version="v1", max_retries=2)
    result = agent.extract(SAMPLE_DOC)

    assert result.success is True
    assert result.attempts == 1
    assert len(result.retry_logs) == 0
    assert result.process_model is not None
    assert result.process_model.title == "Refund Process"
    assert len(result.process_model.steps) == 3
    assert result.process_model.steps[0].source_chunk_ids == ["chunk_0"]


def test_extraction_retry_on_validation_failure():
    mock_llm = SequenceMockLLM([INVALID_UNMARKED_CYCLE_JSON, CORRECTED_CYCLE_JSON])
    agent = ExtractionAgent(llm_client=mock_llm, prompt_version="v1", max_retries=2)
    result = agent.extract(SAMPLE_DOC)

    assert result.success is True
    assert result.attempts == 2
    assert len(result.retry_logs) == 1
    assert "Cycle detected without explicit loop marker" in result.retry_logs[0]["error"]
    assert "CRITICAL VALIDATION FEEDBACK" in mock_llm.prompts_received[1]
    assert result.process_model.steps[1].is_loop is True


def test_extraction_exhaustion_on_persistent_failure():
    mock_llm = SequenceMockLLM([
        INVALID_UNMARKED_CYCLE_JSON,
        INVALID_UNMARKED_CYCLE_JSON,
        INVALID_UNMARKED_CYCLE_JSON,
    ])
    agent = ExtractionAgent(llm_client=mock_llm, prompt_version="v1", max_retries=2)
    result = agent.extract(SAMPLE_DOC)

    assert result.success is False
    assert result.attempts == 3
    assert len(result.retry_logs) == 3
    assert result.process_model is None
    assert "Cycle detected without explicit loop marker" in result.error


def test_prompt_version_switching():
    mock_llm = SequenceMockLLM([VALID_MODEL_JSON])
    agent_v2 = ExtractionAgent(llm_client=mock_llm, prompt_version="v2")
    result = agent_v2.extract(SAMPLE_DOC)
    assert result.success is True
    assert "enterprise process modeling engine" in mock_llm.prompts_received[0]


def test_chunker_utility():
    chunks = chunk_document(SAMPLE_DOC)
    assert len(chunks) == 2
    assert chunks[0].id == "chunk_0"
    assert "Customer requests a refund" in chunks[0].text
    assert chunks[1].id == "chunk_1"
    assert "billing dispatches refund" in chunks[1].text
