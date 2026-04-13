"""Embedding provider abstraction.

Provides a common interface for different embedding backends (Ollama, Gemini, etc.).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence

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
