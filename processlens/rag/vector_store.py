import math
from typing import Optional
from processlens.core.chunker import DocumentChunk
from processlens.rag.embedder import Embedder, get_embedder


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


class VectorStore:
    def __init__(self, embedder: Optional[Embedder] = None):
        self.embedder = embedder or get_embedder()
        self._in_memory_chunks: dict[str, list[DocumentChunk]] = {}
        self._in_memory_embeddings: dict[str, list[list[float]]] = {}

    def index_chunks(self, document_id: str, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        self._in_memory_chunks[document_id] = list(chunks)
        self._in_memory_embeddings[document_id] = embeddings

    def search(
        self,
        query: str,
        document_id: Optional[str] = None,
        top_k: int = 3,
    ) -> list[DocumentChunk]:
        query_vec = self.embedder.embed_text(query)

        target_doc_ids = [document_id] if document_id else list(self._in_memory_chunks.keys())
        scored: list[tuple[float, DocumentChunk]] = []

        for d_id in target_doc_ids:
            chunks = self._in_memory_chunks.get(d_id, [])
            embeddings = self._in_memory_embeddings.get(d_id, [])

            for chunk, vec in zip(chunks, embeddings):
                sim = cosine_similarity(query_vec, vec)
                scored.append((sim, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    def search_for_extraction(
        self,
        document_id: str,
        top_k: int = 3,
    ) -> list[DocumentChunk]:
        queries = [
            "process execution steps actions procedures",
            "responsible actors roles departments",
            "decision points conditions approval branches",
        ]

        seen_ids = set()
        retrieved: list[DocumentChunk] = []

        for q in queries:
            matched = self.search(query=q, document_id=document_id, top_k=top_k)
            for chunk in matched:
                if chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    retrieved.append(chunk)

        doc_chunks = self._in_memory_chunks.get(document_id, [])
        order_map = {c.id: i for i, c in enumerate(doc_chunks)}
        retrieved.sort(key=lambda c: order_map.get(c.id, 999))

        return retrieved[: max(top_k, 3)]


_global_vector_store = None


def get_vector_store() -> VectorStore:
    global _global_vector_store
    if _global_vector_store is None:
        _global_vector_store = VectorStore()
    return _global_vector_store
