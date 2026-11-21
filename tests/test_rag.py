import json
import pytest
from processlens.core.chunker import DocumentChunk, chunk_document
from processlens.core.llm import LLMClient
from processlens.extractor.agent import ExtractionAgent
from processlens.models.process import ProcessModel
from processlens.rag.embedder import Embedder
from processlens.rag.vector_store import VectorStore, cosine_similarity
from processlens.db.models import Base, DocumentRecord, ChunkRecord, ProcessModelRecord
from sqlalchemy import create_engine


SAMPLE_LONG_SOP = """STANDARD OPERATING PROCEDURE: DATA CENTER FAILOVER
[Chunk 0: Intake and Anomaly Trigger]
Monitoring system flags primary cluster outage. Systems Engineer logs critical incident ticket.
[Chunk 1: Verification and Impact Analysis]
Infrastructure Lead verifies failure of primary power grid and calculates RTO impact.
[Chunk 2: Approval Gate]
Operations Director authorizes DNS routing cutover to secondary disaster recovery site.
[Chunk 3: Traffic Migration]
DevOps Engineer updates BGP routing tables and synchronizes database replication streams.
[Chunk 4: Health Check and Verification]
QA Validator executes synthetic transaction tests to confirm 100% operational readiness.
[Chunk 5: Incident Closure]
Communications Lead publishes post-failover stakeholder bulletin and closes ticket."""

MOCK_EXTRACTED_JSON = json.dumps({
    "title": "Data Center Failover",
    "version": "1.0",
    "actors": ["Systems Engineer", "Operations Director", "DevOps Engineer"],
    "steps": [
        {
            "id": "step_1",
            "name": "Log critical incident",
            "actor": "Systems Engineer",
            "next_step_ids": ["step_2"],
            "source_chunk_ids": ["chunk_0"],
            "confidence": 0.99,
            "is_loop": False,
        },
        {
            "id": "step_2",
            "name": "Authorize DNS cutover",
            "actor": "Operations Director",
            "next_step_ids": ["step_3"],
            "source_chunk_ids": ["chunk_2"],
            "confidence": 0.97,
            "is_loop": False,
        },
        {
            "id": "step_3",
            "name": "Migrate traffic",
            "actor": "DevOps Engineer",
            "next_step_ids": [],
            "source_chunk_ids": ["chunk_3"],
            "confidence": 0.98,
            "is_loop": False,
        },
    ],
})


def test_embedder_dimension_and_normalization():
    embedder = Embedder(force_fallback=True)
    vec = embedder.embed_text("Critical incident response protocol")
    assert len(vec) == 384
    norm = sum(v * v for v in vec)
    assert abs(norm - 1.0) < 1e-4


def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]

    assert abs(cosine_similarity(v1, v2) - 1.0) < 1e-5
    assert abs(cosine_similarity(v1, v3) - 0.0) < 1e-5


def test_vector_store_indexing_and_retrieval():
    embedder = Embedder(force_fallback=True)
    store = VectorStore(embedder=embedder)
    chunks = chunk_document(SAMPLE_LONG_SOP)

    store.index_chunks("sop_failover", chunks)

    results = store.search(query="BGP routing and DNS traffic cutover", document_id="sop_failover", top_k=2)
    assert len(results) == 2
    matched_ids = [c.id for c in results]
    assert any("chunk" in mid for mid in matched_ids)


def test_switchable_rag_flag():
    embedder = Embedder(force_fallback=True)
    store = VectorStore(embedder=embedder)
    mock_llm = LLMClient(mock_response=MOCK_EXTRACTED_JSON)

    agent = ExtractionAgent(
        llm_client=mock_llm,
        prompt_version="v1",
        vector_store=store,
    )

    res_no_rag = agent.extract(SAMPLE_LONG_SOP, use_rag=False)
    assert res_no_rag.success is True
    assert res_no_rag.rag_used is False
    assert len(res_no_rag.retrieved_chunk_ids) == 0
    assert len(res_no_rag.chunks) == 6

    res_rag = agent.extract(SAMPLE_LONG_SOP, use_rag=True, top_k=3)
    assert res_rag.success is True
    assert res_rag.rag_used is True
    assert len(res_rag.retrieved_chunk_ids) <= 3
    assert len(res_rag.chunks) <= 3


def test_database_models_schema():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    table_names = Base.metadata.tables.keys()
    assert "documents" in table_names
    assert "document_chunks" in table_names
    assert "process_models" in table_names
