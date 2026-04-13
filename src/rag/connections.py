"""
Shared helpers for working with external services (ChromaDB, Qdrant).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import chromadb
from agent.config import CHROMA_HOST, CHROMA_PORT, QDRANT_API_KEY, QDRANT_URL


class BaseConnection(ABC):
    @abstractmethod
    def get_collection(self, name: str) -> None:
        ...

    @abstractmethod
    def create_collection(self, name: str) -> None:
        ...

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        ...

    @abstractmethod
    def upsert(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: List[dict],
        ids: List[str],
        embeddings: List[List[float]],
    ) -> None:
        ...

    @abstractmethod
    def query(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int,
        include: List[str],
    ) -> dict:
        ...


class ChromaConnection(BaseConnection):
    def __init__(self, host: str | None = None, port: int | None = None):
        host = host or CHROMA_HOST
        port = port or CHROMA_PORT
        print(f"Connecting to ChromaDB service at http://{host}:{port}")
        self.client = chromadb.HttpClient(host=host, port=port)

    def get_collection(self, name: str):
        return self.client.get_collection(name)

    def create_collection(self, name: str):
        return self.client.create_collection(name)

    def delete_collection(self, name: str) -> None:
        self.client.delete_collection(name)

    def upsert(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: List[dict],
        ids: List[str],
        embeddings: List[List[float]],
    ) -> None:
        collection = self.client.get_collection(collection_name)
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )

    def query(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int,
        include: List[str],
    ) -> dict:
        collection = self.client.get_collection(collection_name)
        return collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=include,
        )


def get_qdrant_client(
    url: str | None = None,
    api_key: str | None = None,
):
    from qdrant_client import QdrantClient

    if url is None:
        url = QDRANT_URL
    if api_key is None:
        api_key = QDRANT_API_KEY
    return QdrantClient(url=url, api_key=api_key)
