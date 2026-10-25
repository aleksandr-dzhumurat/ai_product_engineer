"""Centralized configuration loaded from environment / .env file."""
from __future__ import annotations

import os

from dotenv import load_dotenv

DOTENV_FILE = os.environ.get("DOTENV_FILE", ".env")
load_dotenv(DOTENV_FILE)

# LLM
# LLM_BASE_URL = os.environ["LLM_BASE_URL"]
# LLM_MODEL = os.environ["LLM_MODEL"]
# LLM_API_KEY = os.environ["NEBIUS_API_KEY"]

# Embeddings
# Previous model (English-centric): nomic-embed-text, 768 dims
# EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
# EMBEDDING_SIZE = int(os.environ.get("EMBEDDING_SIZE", "768"))
# Current model (multilingual, good for Russian): bge-m3, 1024 dims
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_SIZE = int(os.environ.get("EMBEDDING_SIZE", "1024"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Vector DB
VECTOR_DB_BACKEND = os.environ.get("VECTOR_DB_BACKEND", "chroma").lower()  # chroma | weaviate
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "documents")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.environ.get("WEAVIATE_HTTP_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))
QDRANT_URL = os.environ.get("QDRANT_URL") or os.environ.get("QDRANT_HOST")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

# Agent
MAX_ITERATIONS = 3

# Telegram bot
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN")
TG_BOT_DATA = os.environ.get("TG_BOT_DATA", "data/tg_bot")
VISION_MODEL = "moonshotai/Kimi-K2.6"

# Langfuse
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
