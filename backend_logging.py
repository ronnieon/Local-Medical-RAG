"""Shared backend logging configuration for local Medical RAG modules."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FILE_PATH = Path(__file__).with_name("backend_execution.log")
_HANDLER_NAME = "backend_execution_rotating_handler"


def configure_backend_logging() -> Path:
    """Configure INFO-level rotating log output for backend milestones.

    The handler rotates to prevent unbounded growth and keeps logs local.
    """
    formatter = logging.Formatter("[%(levelname)s] %(message)s")

    handler = RotatingFileHandler(
        LOG_FILE_PATH,
        mode="a",
        maxBytes=512_000,
        backupCount=2,
        encoding="utf-8",
    )
    handler.set_name(_HANDLER_NAME)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    backend_loggers = [
        "model_setup",
        "agent_orchestrator",
        "knowledge_builder",
        "evaluation_pipeline",
        "mimic_note_bridge",
    ]

    for logger_name in backend_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not any(existing.get_name() == _HANDLER_NAME for existing in logger.handlers):
            logger.addHandler(handler)

    # Suppress verbose transport noise in log file.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return LOG_FILE_PATH
