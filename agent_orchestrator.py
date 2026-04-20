"""Sequential multi-agent orchestration for local medical RAG.

Stages (strictly sequential):
1) Extraction Agent: MIMIC note -> strict JSON (age, symptoms, medications)
2) Retrieval Agent: extracted symptoms -> Chroma similarity search (local embeddings)
3) Synthesis Agent: patient JSON + retrieved chunks -> cited clinical summary

This script is optimized for local execution on unified-memory macOS machines.
No asynchronous execution is used.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

try:
    from langchain_chroma import Chroma
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency 'langchain-chroma'. Install it with: pip install langchain-chroma"
    ) from exc

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend_logging import configure_backend_logging
from model_setup import (
    ensure_local_ollama_models,
    get_embedding_engine,
    get_reasoning_engine,
    release_loaded_connectors,
)

LOGGER = logging.getLogger("agent_orchestrator")


class PatientExtraction(BaseModel):
    """Strict schema for extraction handoff."""

    age: Optional[int] = Field(
        default=None,
        description="Patient age in years when explicitly present, else null.",
    )
    primary_symptoms: List[str] = Field(
        default_factory=list,
        description="Primary reported symptoms/signs from the note.",
    )
    current_medications: List[str] = Field(
        default_factory=list,
        description="Current medications documented in the note.",
    )


class PipelineOutput(BaseModel):
    """Serializable final output payload for downstream phases."""

    extraction: PatientExtraction
    retrieval_query: str
    retrieved_chunks: List[Dict[str, Any]]
    synthesis: str


@dataclass
class OrchestratorConfig:
    """Runtime config for local sequential orchestration."""

    persist_directory: str = "./local_chroma_db"
    collection_name: str = "pmc_medical_literature"
    top_k: int = 4
    max_chunk_chars: int = 1400
    max_total_context_chars: int = 9000
    extraction_retries: int = 1

    def validate(self) -> None:
        if self.top_k < 3 or self.top_k > 5:
            raise ValueError("top_k must be between 3 and 5 to match retrieval constraints")
        if self.max_chunk_chars <= 0 or self.max_total_context_chars <= 0:
            raise ValueError("Context size controls must be positive")


def _clean_lines(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ...[truncated]"


def extraction_agent(note_text: str, extraction_retries: int = 1) -> PatientExtraction:
    """Extraction agent: raw note -> strict JSON via JsonOutputParser + Pydantic."""
    configure_backend_logging()
    LOGGER.info("Extraction Agent: Parsing MIMIC-IV text to JSON...")
    parser = JsonOutputParser(pydantic_object=PatientExtraction)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an information extraction agent for clinical notes. "
                "Extract only what is explicitly present in the note. "
                "Do not infer missing facts. "
                "Return strictly valid JSON following these instructions:\n{format_instructions}",
            ),
            (
                "human",
                "Clinical note:\n{note_text}",
            ),
        ]
    )

    llm = get_reasoning_engine(temperature=0.0)
    chain = prompt | llm

    last_error: Optional[Exception] = None
    attempts = max(1, extraction_retries + 1)

    for attempt in range(1, attempts + 1):
        LOGGER.info("Extraction Agent: Attempt %s/%s", attempt, attempts)
        response = chain.invoke(
            {
                "note_text": note_text,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        raw_text = getattr(response, "content", str(response))

        try:
            parsed = parser.parse(raw_text)
            return PatientExtraction.model_validate(parsed)
        except (ValidationError, Exception) as exc:  # parser may raise output parsing errors
            last_error = exc
            LOGGER.warning("Extraction parse failed on attempt %s: %s", attempt, exc)

    raise RuntimeError(f"Extraction agent failed to produce strict JSON: {last_error}")


def build_retrieval_query(extraction: PatientExtraction) -> str:
    """Convert extracted symptoms to a concise semantic medical query."""
    symptom_text = ", ".join(extraction.primary_symptoms) if extraction.primary_symptoms else "unspecified symptoms"
    meds_text = ", ".join(extraction.current_medications) if extraction.current_medications else "no listed medications"

    if extraction.age is None:
        age_text = "adult or unspecified age"
    else:
        age_text = f"{extraction.age}-year-old"

    return (
        f"Clinical evidence for {age_text} patient with symptoms: {symptom_text}. "
        f"Medication context: {meds_text}. Differential diagnosis and management evidence."
    )


def retrieval_agent(
    extraction: PatientExtraction,
    persist_directory: str,
    collection_name: str,
    top_k: int,
) -> tuple[str, List[Dict[str, Any]]]:
    """Retrieval agent: extracted JSON -> local Chroma top-k chunks."""
    configure_backend_logging()
    query = build_retrieval_query(extraction)
    LOGGER.info("Retrieval Agent: Querying ChromaDB for '%s'...", query)
    embeddings = get_embedding_engine()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    docs = vector_store.similarity_search(query, k=top_k)
    LOGGER.info("ChromaDB: Retrieved %s relevant chunks.", len(docs))

    retrieved: List[Dict[str, Any]] = []
    for idx, doc in enumerate(docs, start=1):
        metadata = dict(doc.metadata or {})
        retrieved.append(
            {
                "doc_id": f"Doc {idx}",
                "content": doc.page_content,
                "metadata": metadata,
            }
        )

    # Release retrieval objects aggressively before synthesis phase.
    del docs
    del vector_store
    gc.collect()
    release_loaded_connectors()

    return query, retrieved


def _build_citation_context(
    retrieved_chunks: List[Dict[str, Any]],
    max_chunk_chars: int,
    max_total_context_chars: int,
) -> str:
    """Build bounded synthesis context with explicit doc IDs for citations."""
    context_parts: List[str] = []
    running_total = 0

    for chunk in retrieved_chunks:
        header = f"[{chunk['doc_id']}]"
        content = _truncate(str(chunk.get("content", "")), max_chunk_chars)
        pmcid = chunk.get("metadata", {}).get("pmcid", "unknown")
        block = f"{header} PMCID={pmcid}\n{content}".strip()

        projected = running_total + len(block)
        if projected > max_total_context_chars:
            break

        context_parts.append(block)
        running_total = projected

    return "\n\n".join(context_parts).strip()


def synthesis_agent(
    extraction: PatientExtraction,
    retrieved_chunks: List[Dict[str, Any]],
    max_chunk_chars: int,
    max_total_context_chars: int,
) -> str:
    """Synthesis agent: patient JSON + retrieved docs -> cited summary."""
    configure_backend_logging()
    LOGGER.info("Synthesis Agent: Generating final summary with citations...")
    context = _build_citation_context(
        retrieved_chunks=retrieved_chunks,
        max_chunk_chars=max_chunk_chars,
        max_total_context_chars=max_total_context_chars,
    )

    if not context:
        return (
            "Insufficient retrieved evidence to generate a grounded clinical summary. "
            "Please ingest more literature or broaden retrieval coverage."
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a clinical synthesis assistant operating under strict grounding rules. "
                "Use only the evidence in the provided retrieved context. "
                "Do not introduce any medical fact not supported by the context. "
                "Every substantive claim must include inline citations in the form [Doc N]. "
                "If evidence is insufficient, state uncertainty explicitly. "
                "Output format:\n"
                "1) Clinical Summary\n"
                "2) Differential Considerations\n"
                "3) Evidence Limits",
            ),
            (
                "human",
                "Patient JSON:\n{patient_json}\n\nRetrieved Context:\n{retrieved_context}",
            ),
        ]
    )

    llm = get_reasoning_engine(temperature=0.0)
    chain = prompt | llm

    response = chain.invoke(
        {
            "patient_json": extraction.model_dump_json(indent=2),
            "retrieved_context": context,
        }
    )
    return _clean_lines(getattr(response, "content", str(response)))


def run_pipeline(note_text: str, config: OrchestratorConfig) -> PipelineOutput:
    """Execute extraction -> retrieval -> synthesis sequentially."""
    configure_backend_logging()
    config.validate()
    ensure_local_ollama_models()

    LOGGER.info("Stage 1/3: Extraction agent started")
    extraction = extraction_agent(note_text=note_text, extraction_retries=config.extraction_retries)
    LOGGER.info("Stage 1/3 complete")

    # Release model connector before retrieval phase to reduce memory pressure.
    release_loaded_connectors()
    gc.collect()

    LOGGER.info("Stage 2/3: Retrieval agent started")
    retrieval_query, retrieved_chunks = retrieval_agent(
        extraction=extraction,
        persist_directory=config.persist_directory,
        collection_name=config.collection_name,
        top_k=config.top_k,
    )
    LOGGER.info("Stage 2/3 complete: Retrieved %s chunks", len(retrieved_chunks))

    LOGGER.info("Stage 3/3: Synthesis agent started")
    synthesis = synthesis_agent(
        extraction=extraction,
        retrieved_chunks=retrieved_chunks,
        max_chunk_chars=config.max_chunk_chars,
        max_total_context_chars=config.max_total_context_chars,
    )
    LOGGER.info("Stage 3/3 complete")

    # Final cleanup for long-running workflows.
    release_loaded_connectors()
    gc.collect()

    return PipelineOutput(
        extraction=extraction,
        retrieval_query=retrieval_query,
        retrieved_chunks=retrieved_chunks,
        synthesis=synthesis,
    )


def _read_note_input(note_file: Optional[str], note_text: Optional[str]) -> str:
    if note_text and note_text.strip():
        return note_text.strip()
    if note_file:
        path = Path(note_file)
        return path.read_text(encoding="utf-8").strip()
    raise ValueError("Provide either --note-file or --note-text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential local agent orchestration for Medical RAG")
    parser.add_argument("--note-file", help="Path to text file containing a raw MIMIC-style note")
    parser.add_argument("--note-text", help="Raw clinical note string")
    parser.add_argument("--persist-directory", default="./local_chroma_db", help="Local Chroma DB path")
    parser.add_argument("--collection-name", default="pmc_medical_literature", help="Chroma collection name")
    parser.add_argument("--top-k", type=int, default=4, help="Top retrieval results (must be 3-5)")
    parser.add_argument("--max-chunk-chars", type=int, default=1400, help="Max chars per retrieved chunk in synthesis")
    parser.add_argument("--max-total-context-chars", type=int, default=9000, help="Max cumulative context chars")
    parser.add_argument("--extraction-retries", type=int, default=1, help="Extra retries for strict JSON parsing")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()
    note_text = _read_note_input(note_file=args.note_file, note_text=args.note_text)

    config = OrchestratorConfig(
        persist_directory=args.persist_directory,
        collection_name=args.collection_name,
        top_k=args.top_k,
        max_chunk_chars=args.max_chunk_chars,
        max_total_context_chars=args.max_total_context_chars,
        extraction_retries=args.extraction_retries,
    )

    output = run_pipeline(note_text=note_text, config=config)

    print(json.dumps(output.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
