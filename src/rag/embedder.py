"""Embedding provider abstraction.

Provides a common interface for different embedding backends (Ollama, Gemini, etc.).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import requests


@dataclass
class EmbeddingModel:
    name: str
    embedding_size: int


class Embedder(ABC):
    """Base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Return embedding vectors for the given texts."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Return a human-readable identifier for this embedder."""
        ...


class OllamaEmbedder(Embedder):
    """Fetch embeddings from a local Ollama instance."""

    def __init__(self, host: str, model: EmbeddingModel, timeout: float = 120.0):
        self._host = host.rstrip("/")
        self._model = model
        self._timeout = timeout

    def name(self) -> str:
        return f"ollama-{self._model.name}"

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        url = self._host + "/api/embeddings"
        embeddings: List[List[float]] = []

        for idx, text in enumerate(texts):
            if not text or not text.strip():
                print(f"Warning: Skipping empty text at index {idx}")
                embeddings.append([0.0] * self._model.embedding_size)
                continue

            try:
                response = requests.post(
                    url,
                    json={"model": self._model.name, "prompt": text},
                    timeout=self._timeout,
                )
                response.raise_for_status()
                data = response.json()

                embedding = data.get("embedding") or data.get("embeddings")

                if not embedding or (isinstance(embedding, list) and len(embedding) == 0):
                    print(f"Warning: Empty embedding for text at index {idx} (length: {len(text)})")
                    print(f"Text preview: {text[:100]}...")
                    print(f"Response: {data}")
                    embedding = [0.0] * self._model.embedding_size

                embeddings.append(embedding)

            except Exception as e:
                print(f"Error fetching embedding for text {idx}: {e}")
                print(f"Text preview: {text[:100]}...")
                raise

        return embeddings


class GeminiEmbedder(Embedder):
    """Fetch embeddings from the Gemini API."""

    DEFAULT_MODEL = "gemini-embedding-001"
    DEFAULT_DIMENSIONS = 768

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        output_dimensionality: int = DEFAULT_DIMENSIONS,
    ):
        from google import genai

        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._model = model
        self._output_dimensionality = output_dimensionality

    def name(self) -> str:
        return f"gemini-{self._model}-{self._output_dimensionality}"

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        from google.genai import types

        response = self._client.models.embed_content(
            model=self._model,
            contents=list(texts),
            config=types.EmbedContentConfig(
                output_dimensionality=self._output_dimensionality,
            ),
        )
        return [list(e.values) for e in response.embeddings]


def train_embeds(
    corpus_texts: Sequence[str],
    embedder,
    sentence_embedding_path: str,
    overwrite: bool = False,
) -> np.ndarray:
    """Embed corpus texts, caching the result as a .npy file at sentence_embedding_path.

    embedder can be any object with embed_documents() (langchain-style) or
    embed() (rag.embedder.Embedder-style).
    """
    if os.path.exists(sentence_embedding_path) and not overwrite:
        print(f"corpus loading from {sentence_embedding_path}")
        passage_embeddings = np.load(sentence_embedding_path)
    else:
        print(f"num rows {len(corpus_texts)}")
        if hasattr(embedder, "embed_documents"):
            passage_embeddings = embedder.embed_documents(list(corpus_texts))
        else:
            passage_embeddings = embedder.embed(list(corpus_texts))
        passage_embeddings = np.array(passage_embeddings).astype("float32")
        with open(sentence_embedding_path, "wb") as f:
            np.save(f, passage_embeddings)
        print(f"corpus saved to {sentence_embedding_path}")
    print(f"Num embeddings {passage_embeddings.shape[0]}")
    return passage_embeddings


class VectorIndex:
    """Brute-force cosine-similarity search over precomputed corpus embeddings."""

    def __init__(self, embedder_model, corpus, corpus_embeddings):
        self.vectorizer = embedder_model
        self.vector_index = corpus_embeddings
        self.items = np.array([i["doc_id"] for i in corpus])
        print("Index built successfully. ✅")

    def retrieve(self, query_string, threshold=0.1, top_k=10):
        from sklearn.metrics.pairwise import cosine_similarity

        query_vector = self.vectorizer.embed_documents([query_string])
        cosine_similarities = cosine_similarity(
            np.array(query_vector).reshape(1, -1), self.vector_index
        ).flatten()
        top_k_indices = cosine_similarities.argsort()[::-1][:top_k]
        top_k_scores = [round(i, 4) for i in cosine_similarities[top_k_indices]]
        return [{"idx": i, "score": s} for s, i in zip(top_k_scores, top_k_indices)]
