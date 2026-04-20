"""Local Ollama model setup utilities for a privacy-first medical RAG pipeline.

This module provides:
1) Verification and automatic pulling of required local Ollama models.
2) Lazy LangChain connector creation for chat and embeddings.
3) A sequential-loading guard to avoid keeping both heavy connectors active at once.

Design goal: memory safety on unified-memory laptops (e.g., macOS MacBook Air).
"""

from __future__ import annotations

import gc
import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional, Set

from ollama import Client

from backend_logging import configure_backend_logging

try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
except ImportError as exc:  # pragma: no cover - import guard for environment setup
    raise ImportError(
        "Missing dependency 'langchain-ollama'. Install it with: pip install langchain-ollama"
    ) from exc


DEFAULT_REASONING_MODEL = "llama3.2:latest"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

LOGGER = logging.getLogger("model_setup")


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for Ollama host and model names."""

    reasoning_model: str = DEFAULT_REASONING_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ollama_host: Optional[str] = None


def _extract_model_names(list_response: object) -> Set[str]:
    """Extract model names from `ollama.Client.list()` response safely.

    The Ollama Python client response format has changed across versions.
    This parser handles both dict-style and object-style variants.
    """
    names: Set[str] = set()

    models = None
    if isinstance(list_response, dict):
        models = list_response.get("models", [])
    else:
        models = getattr(list_response, "models", [])

    for item in models or []:
        if isinstance(item, dict):
            name = item.get("name") or item.get("model")
        else:
            name = getattr(item, "name", None) or getattr(item, "model", None)

        if isinstance(name, str) and name.strip():
            names.add(name)

    return names


def _base_model_name(model_name: str) -> str:
    """Normalize a model name by removing tag suffix when present.

    Examples:
        "nomic-embed-text" -> "nomic-embed-text"
        "nomic-embed-text:latest" -> "nomic-embed-text"
    """
    return model_name.split(":", 1)[0].strip()


def _model_is_available(required_model: str, local_models: Set[str]) -> bool:
    """Return True when a required model exists in local registry.

    Ollama registries often return model names with explicit tags (e.g. :latest).
    This matcher treats tagged and untagged forms as equivalent.
    """
    required = required_model.strip()
    if required in local_models:
        return True

    required_base = _base_model_name(required)
    local_bases = {_base_model_name(name) for name in local_models}
    return required_base in local_bases


def get_ollama_client(ollama_host: Optional[str] = None) -> Client:
    """Return an Ollama client connected to the local daemon.

    Args:
        ollama_host: Optional Ollama host, e.g. "http://127.0.0.1:11434".
            If omitted, the client's default host resolution is used.
    """
    if ollama_host:
        return Client(host=ollama_host)
    return Client()


def list_local_models(client: Optional[Client] = None, ollama_host: Optional[str] = None) -> Set[str]:
    """Return the set of locally available model names in Ollama."""
    active_client = client or get_ollama_client(ollama_host=ollama_host)
    response = active_client.list()
    return _extract_model_names(response)


def ensure_models_available(
    config: ModelConfig = ModelConfig(),
    client: Optional[Client] = None,
) -> Set[str]:
    """Ensure required reasoning and embedding models exist locally.

    If a required model is missing, this function pulls it from Ollama
    programmatically. Returns the final local model set after verification.

    Args:
        config: Model and host configuration.
        client: Optional pre-built Ollama client for dependency injection/testing.

    Returns:
        Set[str]: names of models available locally after ensuring requirements.

    Raises:
        RuntimeError: if pull succeeds but model still does not appear in local list.
    """
    configure_backend_logging()
    active_client = client or get_ollama_client(config.ollama_host)

    LOGGER.info(
        "Model Check: Ensuring local models are available (reasoning=%s, embedding=%s)",
        config.reasoning_model,
        config.embedding_model,
    )

    required = [config.reasoning_model, config.embedding_model]
    local_models = list_local_models(client=active_client)
    missing = [model_name for model_name in required if not _model_is_available(model_name, local_models)]

    for model_name in sorted(missing):
        # stream=False keeps control flow simple for orchestration scripts.
        LOGGER.info("Model Pull: Pulling missing model '%s'", model_name)
        active_client.pull(model=model_name, stream=False)

    if missing:
        local_models = list_local_models(client=active_client)
        still_missing = [
            model_name for model_name in required if not _model_is_available(model_name, local_models)
        ]
        if still_missing:
            raise RuntimeError(
                f"Failed to provision required Ollama models: {sorted(still_missing)}"
            )

    LOGGER.info("Model Check: Required local models are available")

    return local_models


class SequentialModelManager:
    """Lazy model connector manager enforcing one active connector at a time.

    This class intentionally avoids holding chat and embedding connector instances
    concurrently. Before instantiating one connector, it releases the other and
    triggers garbage collection to reduce Python-side memory retention.

    Note:
        Ollama daemon-side model residency is controlled by Ollama runtime policy.
        This class controls application-side object lifetimes and usage flow.
    """

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._chat_model: Optional[ChatOllama] = None
        self._embedding_model: Optional[OllamaEmbeddings] = None

    def ensure_models(self) -> Set[str]:
        """Verify and pull required Ollama models if needed."""
        with self._lock:
            return ensure_models_available(config=self.config)

    def get_chat_model(self, temperature: float = 0.0, **kwargs: Any) -> ChatOllama:
        """Return lazily initialized `ChatOllama`.

        Sequential memory rule:
            If an embedding connector exists, release it first.

        Args:
            temperature: Sampling temperature for clinical determinism.
            **kwargs: Forwarded to `ChatOllama` constructor.
        """
        with self._lock:
            if self._embedding_model is not None:
                self._embedding_model = None
                gc.collect()

            if self._chat_model is None:
                configure_backend_logging()
                chat_params: dict[str, Any] = {
                    "model": self.config.reasoning_model,
                    "base_url": self.config.ollama_host,
                    "temperature": temperature,
                }
                chat_params.update(kwargs)
                LOGGER.info("Initializing ChatOllama (%s)", self.config.reasoning_model)
                self._chat_model = ChatOllama(**chat_params)

            return self._chat_model

    def get_embedding_model(self, **kwargs: Any) -> OllamaEmbeddings:
        """Return lazily initialized `OllamaEmbeddings`.

        Sequential memory rule:
            If a chat connector exists, release it first.

        Args:
            **kwargs: Forwarded to `OllamaEmbeddings` constructor.
        """
        with self._lock:
            if self._chat_model is not None:
                self._chat_model = None
                gc.collect()

            if self._embedding_model is None:
                configure_backend_logging()
                embedding_params: dict[str, Any] = {
                    "model": self.config.embedding_model,
                    "base_url": self.config.ollama_host,
                }
                embedding_params.update(kwargs)
                LOGGER.info("Initializing OllamaEmbeddings (%s)", self.config.embedding_model)
                self._embedding_model = OllamaEmbeddings(**embedding_params)

            return self._embedding_model

    def release_all(self) -> None:
        """Release connector references and trigger garbage collection."""
        with self._lock:
            self._chat_model = None
            self._embedding_model = None
            gc.collect()


# Module-level singleton manager for lightweight reuse across scripts.
_DEFAULT_MANAGER = SequentialModelManager()


def ensure_local_ollama_models(config: ModelConfig = ModelConfig()) -> Set[str]:
    """Convenience function to provision required local models."""
    manager = SequentialModelManager(config=config)
    return manager.ensure_models()


def get_reasoning_engine(
    config: Optional[ModelConfig] = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> ChatOllama:
    """Get a lazy `ChatOllama` reasoning connector.

    Uses a module singleton by default for minimal overhead.
    Pass `config` to update model names/host for this process.
    """
    global _DEFAULT_MANAGER
    if config is not None:
        _DEFAULT_MANAGER = SequentialModelManager(config=config)
    return _DEFAULT_MANAGER.get_chat_model(temperature=temperature, **kwargs)


def get_embedding_engine(
    config: Optional[ModelConfig] = None,
    **kwargs: Any,
) -> OllamaEmbeddings:
    """Get a lazy `OllamaEmbeddings` connector.

    Uses a module singleton by default for minimal overhead.
    Pass `config` to update model names/host for this process.
    """
    global _DEFAULT_MANAGER
    if config is not None:
        _DEFAULT_MANAGER = SequentialModelManager(config=config)
    return _DEFAULT_MANAGER.get_embedding_model(**kwargs)


def release_loaded_connectors() -> None:
    """Release any loaded connector references from the singleton manager."""
    _DEFAULT_MANAGER.release_all()


__all__ = [
    "DEFAULT_REASONING_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "ModelConfig",
    "SequentialModelManager",
    "ensure_local_ollama_models",
    "ensure_models_available",
    "get_reasoning_engine",
    "get_embedding_engine",
    "release_loaded_connectors",
    "list_local_models",
]
