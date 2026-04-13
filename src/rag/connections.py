"""
Shared helpers for working with external services (ChromaDB, Qdrant).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List

import chromadb
from dotenv import load_dotenv

load_dotenv()


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
    def __init__(self, host: str | None = None, port: int = 8000):
        if host is None:
            host = os.getenv('CHROMA_HOST', 'localhost')
        chroma_port = int(os.getenv('CHROMA_PORT', str(port)))
        print(f"Connecting to ChromaDB service at http://{host}:{chroma_port}")
        self.client = chromadb.HttpClient(host=host, port=chroma_port)

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
        url = os.environ["QDRANT_URL"]
    if api_key is None:
        api_key = os.environ["QDRANT_API_KEY"]
    return QdrantClient(url=url, api_key=api_key)
