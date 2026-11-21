import hashlib
import math
from typing import Optional
from processlens.core.config import get_settings


class Embedder:
    def __init__(self, model_name: Optional[str] = None, force_fallback: bool = False):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name
        self.dimension = 384
        self.force_fallback = force_fallback
        self._st_model = None

    def _get_st_model(self):
        if self.force_fallback:
            return None
        if self._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(self.model_name)
            except Exception:
                self._st_model = False
        return self._st_model if self._st_model is not False else None

    def _compute_fallback_vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        words = text.lower().split()
        if not words:
            return vec

        for word in words:
            word_hash = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = word_hash % self.dimension
            sign = 1.0 if (word_hash // self.dimension) % 2 == 0 else -1.0
            vec[idx] += sign

            for n in range(2, 5):
                if len(word) >= n:
                    for i in range(len(word) - n + 1):
                        ngram = word[i:i + n]
                        ng_hash = int(hashlib.sha256(ngram.encode("utf-8")).hexdigest(), 16)
                        ng_idx = ng_hash % self.dimension
                        vec[ng_idx] += 0.5

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        st_model = self._get_st_model()
        if st_model is not None:
            try:
                embeddings = st_model.encode(texts, convert_to_numpy=True)
                return [e.tolist() for e in embeddings]
            except Exception:
                pass
        return [self._compute_fallback_vector(t) for t in texts]

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


_global_embedder = None


def get_embedder(force_fallback: bool = False) -> Embedder:
    global _global_embedder
    if _global_embedder is None or force_fallback:
        _global_embedder = Embedder(force_fallback=force_fallback)
    return _global_embedder
