import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_document(text: str, max_chunk_chars: int = 800, overlap_chars: int = 100) -> list[DocumentChunk]:
    text = text.strip()
    if not text:
        return []

    chunk_pattern = re.compile(r"(\[Chunk\s+(\d+)(?::\s*[^\]]*)?\])", re.IGNORECASE)
    matches = list(chunk_pattern.finditer(text))

    if matches:
        chunks: list[DocumentChunk] = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk_num = match.group(2)
            chunk_id = f"chunk_{chunk_num}"
            chunk_content = text[start:end].strip()
            chunks.append(DocumentChunk(id=chunk_id, text=chunk_content, metadata={"index": int(chunk_num)}))
        return chunks

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current_text = ""
    chunk_index = 0

    for paragraph in paragraphs:
        if current_text and len(current_text) + len(paragraph) > max_chunk_chars:
            chunks.append(
                DocumentChunk(
                    id=f"chunk_{chunk_index}",
                    text=current_text.strip(),
                    metadata={"index": chunk_index},
                )
            )
            chunk_index += 1
            current_text = current_text[-overlap_chars:] + " " + paragraph
        else:
            current_text = (current_text + "\n\n" + paragraph).strip()

    if current_text:
        chunks.append(
            DocumentChunk(
                id=f"chunk_{chunk_index}",
                text=current_text.strip(),
                metadata={"index": chunk_index},
            )
        )

    return chunks
