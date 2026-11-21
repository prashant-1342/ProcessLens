import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from pydantic import ValidationError
from processlens.core.chunker import DocumentChunk, chunk_document
from processlens.core.llm import LLMClient, get_llm_client
from processlens.models.process import ProcessModel

logger = logging.getLogger("processlens.extractor")


@dataclass
class ExtractionResult:
    process_model: Optional[ProcessModel]
    attempts: int
    retry_logs: list[dict[str, Any]] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None
    chunks: list[DocumentChunk] = field(default_factory=list)


class ExtractionAgent:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_version: str = "v1",
        max_retries: int = 2,
        prompts_dir: Optional[Path] = None,
    ):
        self.llm_client = llm_client or get_llm_client()
        self.prompt_version = prompt_version
        self.max_retries = max_retries
        self.prompts_dir = prompts_dir or Path("prompts")

    def _load_prompt_template(self, version: str) -> str:
        prompt_path = self.prompts_dir / f"{version}.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt version file not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")

    def extract(
        self,
        document_text: str,
        document_title: Optional[str] = None,
        retrieved_chunks: Optional[list[DocumentChunk]] = None,
    ) -> ExtractionResult:
        if retrieved_chunks is not None:
            chunks = retrieved_chunks
        else:
            chunks = chunk_document(document_text)

        chunks_text_representation = "\n\n".join(
            [f"[{c.id.upper()}]\n{c.text}" for c in chunks]
        )

        template = self._load_prompt_template(self.prompt_version)
        base_prompt = template.replace("{document_content}", chunks_text_representation)
        if document_title:
            base_prompt = f"DOCUMENT TITLE: {document_title}\n\n" + base_prompt

        retry_logs: list[dict[str, Any]] = []
        current_prompt = base_prompt
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                model = self.llm_client.generate_structured(
                    prompt=current_prompt,
                    response_model=ProcessModel,
                )

                chunk_id_set = {c.id.lower() for c in chunks}
                for step in model.steps:
                    validated_citations = []
                    for cid in step.source_chunk_ids:
                        cid_clean = cid.lower().strip()
                        if cid_clean in chunk_id_set or f"chunk_{cid_clean}" in chunk_id_set:
                            validated_citations.append(cid_clean)
                    if validated_citations:
                        step.source_chunk_ids = validated_citations

                return ExtractionResult(
                    process_model=model,
                    attempts=attempt + 1,
                    retry_logs=retry_logs,
                    success=True,
                    chunks=chunks,
                )

            except (ValidationError, ValueError, json.JSONDecodeError) as e:
                error_msg = str(e)
                last_error = error_msg
                retry_log_entry = {
                    "attempt": attempt + 1,
                    "error": error_msg,
                    "prompt_snippet": current_prompt[:300],
                }
                retry_logs.append(retry_log_entry)
                logger.warning(
                    f"Extraction validation failed on attempt {attempt + 1}: {error_msg}"
                )

                if attempt < self.max_retries:
                    current_prompt = (
                        base_prompt
                        + f"\n\nCRITICAL VALIDATION FEEDBACK (ATTEMPT {attempt + 1} FAILED):\n"
                        + f"Your previous generation was rejected due to this schema error:\n{error_msg}\n\n"
                        + "Instructions to fix:\n"
                        + "1. Ensure every target in next_step_ids exists in steps.\n"
                        + "2. Ensure every step is connected to the graph (no isolated orphan steps).\n"
                        + "3. If there is a back-edge or retry loop, set is_loop = true on the step.\n"
                        + "4. Return ONLY valid JSON conforming to the ProcessModel schema."
                    )

        return ExtractionResult(
            process_model=None,
            attempts=self.max_retries + 1,
            retry_logs=retry_logs,
            success=False,
            error=last_error,
            chunks=chunks,
        )
