"""
Shared helpers for working with external services (ChromaDB, Qdrant, Weaviate).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List
import socket

import chromadb

from rag.agent.config import (
    CHROMA_HOST,
    CHROMA_PORT,
    QDRANT_API_KEY,
    QDRANT_URL,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_HOST,
    WEAVIATE_HTTP_PORT,
)


class BaseConnection(ABC):
    @abstractmethod
    def get_collection(self, name: str) -> None:
        ...

    @abstractmethod
    def create_collection(self, name: str, schema=None) -> None:
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
    CONNECT_TIMEOUT_SECONDS = 5

    def __init__(self, host: str | None = None, port: int | None = None):
        host = host or CHROMA_HOST
        port = port or CHROMA_PORT
        print(f"Connecting to ChromaDB service at http://{host}:{port}")
        try:
            with socket.create_connection((host, port), timeout=self.CONNECT_TIMEOUT_SECONDS):
                pass
        except OSError as exc:
            raise ValueError(
                f"Could not connect to the Chroma server at http://{host}:{port} "
                f"within {self.CONNECT_TIMEOUT_SECONDS}s: {exc}"
            ) from exc
        self.client = chromadb.HttpClient(host=host, port=port)

    def get_collection(self, name: str):
        return self.client.get_collection(name)

    def create_collection(self, name: str, schema=None):
        if schema is not None:
            return self.client.create_collection(name=name, schema=schema)
        return self.client.create_collection(name=name)

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


class WeaviateConnection(BaseConnection):
    """Connection to a Weaviate server via the v4 client (weaviate-client package).

    Implements the same interface as ChromaConnection. Documents are stored in a
    `text` property; metadata fields are stored as separate string properties.
    Vectors are supplied explicitly (the compose service runs with
    DEFAULT_VECTORIZER_MODEL=none). Weaviate requires collection (class) names
    to start with an uppercase letter, so names like "docs2" become "Docs2".
    """

    CONNECT_TIMEOUT_SECONDS = 5

    def __init__(
        self,
        host: str | None = None,
        http_port: int | None = None,
        grpc_port: int | None = None,
    ):
        host = host or WEAVIATE_HOST
        http_port = http_port or WEAVIATE_HTTP_PORT
        grpc_port = grpc_port or WEAVIATE_GRPC_PORT
        try:
            import weaviate
        except ImportError as exc:
            raise ValueError(
                "The weaviate-client package is not installed. Install it with `pip install weaviate-client`."
            ) from exc
        print(f"Connecting to Weaviate service at http://{host}:{http_port}")
        try:
            with socket.create_connection((host, http_port), timeout=self.CONNECT_TIMEOUT_SECONDS):
                pass
        except OSError as exc:
            raise ValueError(
                f"Could not connect to the Weaviate server at http://{host}:{http_port} "
                f"within {self.CONNECT_TIMEOUT_SECONDS}s: {exc}"
            ) from exc
        self.client = weaviate.connect_to_local(host=host, port=http_port, grpc_port=grpc_port)

    @staticmethod
    def _class_name(name: str) -> str:
        # Weaviate class names must start with an uppercase letter.
        return name[:1].upper() + name[1:]

    def get_collection(self, name: str):
        class_name = self._class_name(name)
        if not self.client.collections.exists(class_name):
            try:
                available = sorted(self.client.collections.list_all())
            except Exception:
                available = []
            raise ValueError(
                f"Weaviate collection '{class_name}' does not exist"
                + (f" (available: {available})" if available else "")
            )
        return self.client.collections.get(class_name)

    def create_collection(self, name: str, schema=None):
        from weaviate.classes.config import Configure, DataType, Property

        return self.client.collections.create(
            name=self._class_name(name),
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="source_dir", data_type=DataType.TEXT),
                Property(name="source_file_name", data_type=DataType.TEXT),
                Property(name="length_chars", data_type=DataType.TEXT),
                Property(name="length_lines", data_type=DataType.TEXT),
            ],
        )

    def delete_collection(self, name: str) -> None:
        self.client.collections.delete(self._class_name(name))

    def upsert(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: List[dict],
        ids: List[str],
        embeddings: List[List[float]],
    ) -> None:
        from weaviate.classes.data import DataObject

        collection = self.get_collection(collection_name)
        with collection.batch.dynamic() as batch:
            for doc, meta, _id, vector in zip(documents, metadatas, ids, embeddings):
                props = {"text": doc}
                props.update({k: str(v) for k, v in meta.items()})
                batch.add_object(
                    properties=props,
                    uuid=_id,
                    vector=vector,
                )
        if len(collection.batch.failed_objects) > 0:
            failed = collection.batch.failed_objects[:3]
            raise RuntimeError(f"Weaviate batch upsert failed for {len(collection.batch.failed_objects)} objects, e.g. {failed}")

    def query(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int,
        include: List[str],
    ) -> dict:
        from weaviate.classes.query import MetadataQuery

        collection = self.get_collection(collection_name)
        results = {"ids": [], "documents": [], "metadatas": []}
        include_distances = "distances" in include
        if include_distances:
            results["distances"] = []
        for qe in query_embeddings:
            res = collection.query.near_vector(
                near_vector=qe,
                limit=n_results,
                return_metadata=MetadataQuery(distance=include_distances),
            )
            results["ids"].append([str(o.uuid) for o in res.objects])
            results["documents"].append([o.properties.get("text", "") for o in res.objects])
            results["metadatas"].append(
                [{k: v for k, v in o.properties.items() if k != "text"} for o in res.objects]
            )
            if include_distances:
                results["distances"].append([o.metadata.distance for o in res.objects])
        return results


def get_backend(backend: str) -> BaseConnection:
    """Instantiate the vector store connection for the selected backend."""
    if backend == "chroma":
        try:
            return ChromaConnection()
        except ValueError as exc:
            raise SystemExit(
                f"{exc}\n"
                f"Start the ChromaDB service first:\n"
                f"  make run-chroma\n"
                f"or directly:\n"
                f"  chroma run --host {CHROMA_HOST} --port {CHROMA_PORT} &"
            ) from exc
    if backend == "weaviate":
        try:
            return WeaviateConnection()
        except ValueError as exc:
            raise SystemExit(
                f"{exc}\n"
                f"Start the Weaviate service first:\n"
                f"  make run-weaviate"
            ) from exc
    raise SystemExit(f"Unsupported backend: {backend!r}. Available: chroma, weaviate")


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
