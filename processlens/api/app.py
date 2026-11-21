from typing import Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from processlens.db.storage import get_model, list_models, save_model
from processlens.extractor.agent import ExtractionAgent
from processlens.models.process import ProcessModel

app = FastAPI(
    title="ProcessLens API",
    version="1.0.0",
    description="LLM Agent for converting SOP documents into schema-validated Process Models",
)


class ExtractRequest(BaseModel):
    document_text: str = Field(..., min_length=10)
    document_title: Optional[str] = None
    document_id: Optional[str] = None
    use_rag: bool = Field(default=False)
    prompt_version: str = Field(default="v3")


class ExtractResponse(BaseModel):
    model_id: Optional[str] = None
    process_model: Optional[ProcessModel] = None
    attempts: int
    success: bool
    error: Optional[str] = None
    rag_used: bool
    retrieved_chunk_ids: list[str] = Field(default_factory=list)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "ProcessLens"}


@app.post("/extract", response_model=ExtractResponse)
def extract_process_model(request: ExtractRequest) -> ExtractResponse:
    agent = ExtractionAgent(prompt_version=request.prompt_version)
    result = agent.extract(
        document_text=request.document_text,
        document_title=request.document_title,
        use_rag=request.use_rag,
        document_id=request.document_id,
    )

    if not result.success or result.process_model is None:
        return ExtractResponse(
            model_id=None,
            process_model=None,
            attempts=result.attempts,
            success=False,
            error=result.error,
            rag_used=result.rag_used,
            retrieved_chunk_ids=result.retrieved_chunk_ids,
        )

    model_id = save_model(result.process_model, document_id=request.document_id)

    return ExtractResponse(
        model_id=model_id,
        process_model=result.process_model,
        attempts=result.attempts,
        success=True,
        rag_used=result.rag_used,
        retrieved_chunk_ids=result.retrieved_chunk_ids,
    )


@app.get("/models/{model_id}", response_model=ProcessModel)
def get_model_by_id(model_id: str) -> ProcessModel:
    model = get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Process model '{model_id}' not found")
    return model


@app.get("/models")
def list_all_models() -> list[dict[str, Any]]:
    return list_models()
