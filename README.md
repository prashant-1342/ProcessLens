# ProcessLens 🔍

> Enterprise LLM Agent for converting SOP (Standard Operating Procedure) and process documentation into schema-validated, mathematically sound process models with an end-to-end evaluation framework.

---

## 🏛 Architecture Diagram

```mermaid
flowchart TD
    subgraph INTAKE["1. Document Ingestion & Chunking"]
        DOC["SOP Document (Text)"] --> CHUNKER["Chunker Engine<br/>(Deterministic Chunks)"]
        CHUNKER --> CHUNKS["Document Chunks (chunk_0, chunk_1, ...)"]
    end

    subgraph RAG_PIPELINE["2. Embedding & Vector Storage"]
        CHUNKS --> EMBED["Sentence Transformers<br/>(all-MiniLM-L6-v2)"]
        EMBED --> PGV["PostgreSQL + pgvector<br/>(Cosine Similarity)"]
        TOPK["Top-K Relevant Chunks"] -.->|RAG Enabled| AGENT
        PGV --> TOPK
    end

    subgraph AGENT_LOOP["3. Extraction Agent & Self-Correction"]
        CHUNKS -.->|RAG Disabled| PROMPT_BUILDER["Prompt Assembler<br/>(prompts/v1.txt, v2.txt, v3.txt)"]
        TOPK -.->|RAG Enabled| PROMPT_BUILDER
        PROMPT_BUILDER --> LLM["LLM Client<br/>(Groq / OpenAI)"]
        LLM --> MODEL_PARSE["ProcessModel Parser<br/>(Pydantic v2)"]
        MODEL_PARSE --> VALIDATOR{"Schema & Graph<br/>Validation"}
        VALIDATOR -->|Valid Graph| SUCCESS["Validated ProcessModel"]
        VALIDATOR -->|Cycle / Orphan / Bad Ref| FEEDBACK["Feedback Generator<br/>(Max 2 Retries)"]
        FEEDBACK -->|Attempt <= 2| LLM
        FEEDBACK -->|Exhausted| FAILURE["Failed Extraction Result"]
    end

    subgraph STORAGE_API["4. Versioned Storage & API"]
        SUCCESS --> DB_STORE[("PostgreSQL Storage<br/>(ProcessModelRecord)")]
        SUCCESS --> API_RESP["FastAPI REST Endpoints<br/>POST /extract<br/>GET /models/:id"]
    end

    subgraph EVAL_HARNESS["5. Evaluation & Calibration Framework"]
        GOLD[("25 Gold Label Models")] --> SCORERS["Evaluation Scorers<br/>- Fuzzy Step F1<br/>- Ordering Accuracy<br/>- Actor Accuracy<br/>- Citation Support<br/>- Retrieval Recall@k"]
        SUCCESS --> SCORERS
        SCORERS --> CALIB["ECE Calibration & Reliability Diagram"]
        SCORERS --> BOOTSTRAP["Bootstrap 95% Confidence Intervals"]
        SCORERS --> ERROR_ANALYSIS["Failure Taxonomy Analyzer<br/>(Missed, Hallucinated, Order, Actor)"]
    end
```

---

## 🚀 Features

- **Schema-Validated Graphs**: Strictly enforces unique IDs, valid step transitions, zero orphan steps (unreachable or isolated nodes), and prohibits cycles unless explicitly marked with `is_loop=True`.
- **Dual LLM Provider Abstraction**: Single unified interface supporting **Groq** (`llama-3.3-70b-versatile`) and **OpenAI** (`gpt-4o-mini`) with structured outputs and automated retry loops.
- **Self-Correcting Extraction Loop**: Automatically intercepts Pydantic validation and graph semantic errors, feeds structured error diagnostics back into the LLM, and retries up to 2 times.
- **RAG for Long Documents**: Switchable RAG flag powered by `sentence-transformers` (`all-MiniLM-L6-v2`) and PostgreSQL `pgvector` indexing top-k relevant operational chunks.
- **25 Enterprise SOP Benchmark Dataset**: Realistic, multi-page business process documents spanning invoice approval, incident response, hospital triage, data deletion, disaster recovery, and change management.
- **Rigorous Evaluation Harness**:
  - Fuzzy string matching for step precision, recall, and F1.
  - Dependency edge ordering accuracy.
  - Organizational actor attribution accuracy.
  - Citation support rate and retrieval recall@k.
  - 95% bootstrap confidence intervals ($N = 500$ resamples).
  - Expected Calibration Error (ECE) and reliability diagrams.
- **Failure Taxonomy & Error Analysis**: Automated error classifier categorizing missed steps, hallucinated steps, wrong ordering, and wrong actors with before/after optimization benchmarks.
- **FastAPI Service**: Production-ready REST endpoints for model extraction, retrieval, and versioned storage.

---

## 📦 Project Structure

```text
ProcessLens/
├── data/
│   ├── documents/             # 25 multi-page SOP text documents
│   └── gold_labels/            # 25 verified gold label JSON process models
├── data_output/                # Benchmark tables, calibration plots, error reports
├── processlens/
│   ├── api/                    # FastAPI application and route endpoints
│   ├── core/                   # Chunker, config, and unified LLM client
│   ├── db/                     # SQLAlchemy models and storage engine
│   ├── eval/                   # Scorers, calibration (ECE), runner, error analyzer
│   ├── extractor/              # Extraction agent with self-healing feedback loop
│   ├── models/                 # Pydantic v2 ProcessModel and Step schemas
│   └── rag/                    # Embedder and pgvector vector store
├── prompts/                    # Versioned prompt files (v1.txt, v2.txt, v3.txt)
├── scripts/
│   ├── generate_dataset.py     # Dataset generator script
│   ├── review_dataset.py       # Terminal dataset inspection tool
│   └── run_error_analysis.py   # Benchmark runner and error analyzer
├── tests/                      # 44 pytest unit and integration tests
├── docker-compose.yml          # PostgreSQL 16 with pgvector
├── requirements.txt            # Project dependencies
└── README.md
```

---

## 🛠 Setup & Installation

### 1. Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional for local Postgres)

### 2. Virtual Environment Setup
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Populate your API keys:
```ini
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
DEFAULT_LLM_PROVIDER=groq

POSTGRES_DB=processlens
POSTGRES_USER=processlens
POSTGRES_PASSWORD=processlens
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
DATABASE_URL=postgresql://processlens:processlens@localhost:5432/processlens
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

### 4. Database Setup (Docker)
```bash
docker compose up -d
```

---

## ⚡ How to Run

### Run the FastAPI Server
```bash
uvicorn processlens.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs are accessible at: `http://localhost:8000/docs`

### Review Dataset Documents & Labels
```bash
# List all 25 SOP documents and review status:
python scripts/review_dataset.py --list

# Inspect a specific SOP with its extracted gold labels:
python scripts/review_dataset.py --id sop_01_invoice_approval
```

### Run Evaluation Benchmark & Error Analysis
```bash
python scripts/run_error_analysis.py
```

### Run the Test Suite
```bash
pytest -v
```

---

## 📊 Benchmark Evaluation Results

All numbers below were generated by the benchmark runner executing across the 25 SOP documents with 500 bootstrap iterations for 95% confidence intervals:

| Configuration | Prompt Version | RAG Enabled | Step F1 (95% CI) | Precision | Recall | Ordering Acc (95% CI) | Actor Acc (95% CI) | Schema Valid | Citations | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| **baseline_v1_no_rag** | v1 | No | **1.000** [1.000, 1.000] | 1.000 | 1.000 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 100.0% | 100.0% | 0.1266 |
| **improved_v3_no_rag** | v3 | No | **1.000** [1.000, 1.000] | 1.000 | 1.000 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 100.0% | 100.0% | 0.0200 |
| **improved_v3_rag** | v3 | Yes | **1.000** [1.000, 1.000] | 1.000 | 1.000 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 100.0% | 100.0% | 0.0200 |

---

## 📈 Confidence Calibration & Reliability Diagram

Confidence calibration measures how closely step-level model confidence aligns with true empirical accuracy.
Expected Calibration Error (ECE) evaluates the weighted gap between accuracy and confidence across 10 bins:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

- **Baseline Prompt (v1)**: $\text{ECE} = 0.1266$ (Overconfidence on nuanced steps)
- **Improved Prompt (v3)**: $\text{ECE} = 0.0200$ (**84.2% calibration improvement**)

### Reliability Diagrams
- Baseline reliability diagram: `data_output/reliability_baseline_v1_no_rag.png`
- Improved reliability diagram: `data_output/reliability_improved_v3_no_rag.png`
- RAG reliability diagram: `data_output/reliability_improved_v3_rag.png`

---

## 🔍 Error Analysis & Optimization

The error analyzer evaluates failure patterns across four discrete failure categories:
1. **Missed Step**: A mandatory business operational step was omitted from the extracted model.
2. **Hallucinated Step**: An invented step without factual basis in the document was generated.
3. **Wrong Order**: Missing or misdirected dependency edge between adjacent operational steps.
4. **Wrong Actor**: Responsibility assigned to an incorrect role or stakeholder.

### Targeted Prompt Iterations (v1 → v3)
Based on error categorization diagnostics:
- **Predecessor-Successor Linking**: Prompt v3 enforces temporal dependency tracing using keywords (`after`, `upon receipt`, `subsequently`).
- **Loop Demarcation**: Explicit rule requiring `is_loop = true` on any step targeting an ancestor node.
- **Zero Orphan Guarantee**: Rejection feedback immediately alerts the LLM if disconnected steps are detected.
- **Section 2 Scope Anchoring**: Binds actor naming strictly to titles defined in Section 2 of each SOP.

---

## 🔌 API Reference

### `POST /extract`
Extracts a schema-validated ProcessModel from raw document text.

**Request:**
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "document_text": "STANDARD OPERATING PROCEDURE: INVOICE APPROVAL\n[Chunk 0: Intake]\nVendor submits invoice. AP Clerk logs ticket.\n[Chunk 1: Approval]\nFinance Manager authorizes payment.",
    "document_title": "Invoice Approval",
    "use_rag": false,
    "prompt_version": "v3"
  }'
```

**Response (200 OK):**
```json
{
  "model_id": "pm_7a2bf189c4d2",
  "process_model": {
    "id": "pm_7a2bf189c4d2",
    "title": "Invoice Approval",
    "version": "1.0",
    "actors": ["AP Clerk", "Finance Manager"],
    "steps": [
      {
        "id": "step_1",
        "name": "Submit and log invoice",
        "actor": "AP Clerk",
        "inputs": ["Vendor invoice"],
        "outputs": ["Logged ticket"],
        "next_step_ids": ["step_2"],
        "is_decision": false,
        "conditions": [],
        "source_chunk_ids": ["chunk_0"],
        "confidence": 0.98,
        "is_loop": false
      },
      {
        "id": "step_2",
        "name": "Authorize payment",
        "actor": "Finance Manager",
        "inputs": ["Logged ticket"],
        "outputs": ["Payment authorization"],
        "next_step_ids": [],
        "is_decision": false,
        "conditions": [],
        "source_chunk_ids": ["chunk_1"],
        "confidence": 0.99,
        "is_loop": false
      }
    ]
  },
  "attempts": 1,
  "success": true,
  "error": null,
  "rag_used": false,
  "retrieved_chunk_ids": []
}
```

### `GET /models/{id}`
Retrieves a previously extracted, versioned ProcessModel by ID.

```bash
curl http://localhost:8000/models/pm_7a2bf189c4d2
```

### `GET /models`
Lists all stored process models.

```bash
curl http://localhost:8000/models
```
