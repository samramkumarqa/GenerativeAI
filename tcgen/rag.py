"""Retrieval: chunk the BRD, embed the chunks, and find the passages relevant to a requirement."""

from typing import Protocol

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedder:
    """Local ONNX embeddings via fastembed: no API key, no PyTorch."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.embed(texts)), dtype="float32")


def split_text(text: str, size: int, overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text(text)


class DocumentIndex:
    """An in-memory FAISS index over one document's chunks (cosine similarity)."""

    def __init__(self, text: str, embedder: Embedder):
        self.embedder = embedder
        self.chunks = split_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        vectors = self._normalise(embedder.embed(self.chunks))
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        faiss.normalize_L2(vectors)
        return vectors

    def search(self, query: str, k: int = config.TOP_K_PER_REQUIREMENT) -> list[int]:
        k = min(k, len(self.chunks))
        query_vec = self._normalise(self.embedder.embed([query]))
        _, ids = self.index.search(query_vec, k)
        return [int(i) for i in ids[0] if i >= 0]

    def context_for(self, queries: list[str], k: int = config.TOP_K_PER_REQUIREMENT) -> str:
        """Top-k chunks for each query, de-duplicated and kept in document order."""
        hits: set[int] = set()
        for q in queries:
            hits.update(self.search(q, k))
        return "\n\n---\n\n".join(self.chunks[i] for i in sorted(hits))
