"""Local, sequential MedQA evaluation pipeline for Medical RAG.

This script evaluates the Phase 3 orchestrator with Ragas while forcing all
judge/inference components to run via local Ollama models.

Key constraints enforced:
- No external APIs for judge LLM or embeddings.
- Strictly sequential sample processing (batch_size=1 for evaluation).
- Small MedQA subset (configurable) to protect unified memory.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset

from agent_orchestrator import OrchestratorConfig, run_pipeline
from model_setup import (
    ensure_local_ollama_models,
    get_embedding_engine,
    get_reasoning_engine,
    release_loaded_connectors,
)

LOGGER = logging.getLogger("evaluation_pipeline")


# Ragas imports vary by version; this compatibility block supports common layouts.
try:
    from ragas import evaluate
except ImportError as exc:  # pragma: no cover
    raise ImportError("Missing dependency 'ragas'. Install it with: pip install ragas") from exc

try:
    from ragas.embeddings import LangchainEmbeddingsWrapper
except ImportError:  # pragma: no cover
    from ragas.embeddings.base import LangchainEmbeddingsWrapper

try:
    from ragas.llms import LangchainLLMWrapper
except ImportError:  # pragma: no cover
    from ragas.llms.base import LangchainLLMWrapper

# Prefer answer_correctness when available; fall back to answer_relevancy.
answer_correctness_metric: Optional[Any] = None
answer_relevancy_metric: Optional[Any] = None

try:
    from ragas.metrics import answer_correctness, faithfulness

    answer_correctness_metric = answer_correctness
    HAS_ANSWER_CORRECTNESS = True
except ImportError:  # pragma: no cover
    from ragas.metrics import answer_relevancy, faithfulness

    answer_relevancy_metric = answer_relevancy
    HAS_ANSWER_CORRECTNESS = False


@dataclass
class EvalConfig:
    """Configuration for local MedQA evaluation."""

    medqa_jsonl_path: str
    subset_size: int = 30
    persist_directory: str = "./local_chroma_db"
    collection_name: str = "pmc_medical_literature"
    output_csv: str = "./evaluation_scores.csv"
    output_detailed_csv: str = "./evaluation_detailed.csv"
    top_k: int = 4
    max_chunk_chars: int = 1400
    max_total_context_chars: int = 9000

    def validate(self) -> None:
        if self.subset_size <= 0:
            raise ValueError("subset_size must be positive")
        if self.top_k < 3 or self.top_k > 5:
            raise ValueError("top_k must be between 3 and 5")


def resolve_medqa_jsonl_path(path_or_root: str, split: str = "dev", region: str = "US") -> str:
    """Resolve either a direct JSONL path or MedQA root folder into a JSONL file path."""
    candidate = Path(path_or_root)

    if candidate.is_file():
        return str(candidate)

    resolved = candidate / "questions" / region / f"{split}.jsonl"
    if resolved.is_file():
        return str(resolved)

    raise FileNotFoundError(
        f"Could not resolve MedQA JSONL from '{path_or_root}'. "
        f"Expected file or folder containing '{resolved.as_posix()}'."
    )


def load_medqa_subset(jsonl_path: str, subset_size: int) -> List[Dict[str, Any]]:
    """Load a sequential subset of MedQA records from local JSONL."""
    records: List[Dict[str, Any]] = []

    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue

            item = json.loads(raw)
            question = str(item.get("question", "")).strip()
            ground_truth = str(item.get("answer", "")).strip()
            options = item.get("options", {})

            if not question or not ground_truth:
                continue

            records.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "options": options,
                }
            )

            if len(records) >= subset_size:
                break

    return records


def _format_question_as_note(question: str, options: Dict[str, str]) -> str:
    """Adapt MedQA MCQ format into note-like text for the Phase 3 pipeline."""
    option_lines = []
    for key in sorted(options.keys()):
        option_lines.append(f"{key}. {options[key]}")

    options_text = "\n".join(option_lines)
    return (
        "Clinical Question (for diagnostic reasoning):\n"
        f"{question}\n\n"
        "Answer Choices:\n"
        f"{options_text}"
    )


def run_orchestrator_benchmark(
    medqa_records: List[Dict[str, Any]],
    orchestrator_config: OrchestratorConfig,
) -> List[Dict[str, Any]]:
    """Run orchestrator sequentially for each MedQA sample and capture outputs."""
    rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(medqa_records, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        options = item.get("options", {})

        note_text = _format_question_as_note(question, options)
        LOGGER.info("Benchmark sample %s/%s", idx, len(medqa_records))

        try:
            pipeline_out = run_pipeline(note_text=note_text, config=orchestrator_config)
            contexts = [str(chunk.get("content", "")) for chunk in pipeline_out.retrieved_chunks]

            rows.append(
                {
                    "question": question,
                    "answer": pipeline_out.synthesis,
                    "contexts": contexts,
                    "ground_truth": ground_truth,
                    "retrieval_query": pipeline_out.retrieval_query,
                }
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Skipping sample %s due to pipeline error: %s", idx, exc)

        # Force per-sample cleanup for unified-memory safety.
        release_loaded_connectors()
        gc.collect()

    return rows


def build_ragas_dataset(rows: List[Dict[str, Any]]) -> Dataset:
    """Convert benchmark rows into a HuggingFace dataset for Ragas."""
    dataset_payload = {
        "question": [row["question"] for row in rows],
        "answer": [row["answer"] for row in rows],
        "contexts": [row["contexts"] for row in rows],
        "ground_truth": [row["ground_truth"] for row in rows],
    }
    return Dataset.from_dict(dataset_payload)


def build_local_ragas_judges() -> Tuple[Any, Any]:
    """Instantiate and wrap local Ollama judge LLM and embeddings for Ragas."""
    ensure_local_ollama_models()

    local_llm = get_reasoning_engine(temperature=0.0)
    local_embeddings = get_embedding_engine()

    ragas_llm = LangchainLLMWrapper(local_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(local_embeddings)
    return ragas_llm, ragas_embeddings


def evaluate_with_ragas(dataset: Dataset) -> Tuple[Dict[str, float], List[str]]:
    """Compute faithfulness + answer correctness/relevance using local Ollama judge."""
    ragas_llm, ragas_embeddings = build_local_ragas_judges()

    metric_list: List[Any] = [faithfulness]
    metric_names: List[str] = ["faithfulness"]

    if HAS_ANSWER_CORRECTNESS and answer_correctness_metric is not None:
        metric_list.append(answer_correctness_metric)
        metric_names.append("answer_correctness")
    else:
        if answer_relevancy_metric is None:
            raise RuntimeError(
                "Neither answer_correctness nor answer_relevancy metric is available in ragas"
            )
        metric_list.append(answer_relevancy_metric)
        metric_names.append("answer_relevancy")
        LOGGER.warning("answer_correctness metric unavailable in installed ragas; using answer_relevancy")

    # Explicit local override: llm and embeddings passed directly.
    result: Any = evaluate(
        dataset=dataset,
        metrics=metric_list,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        batch_size=1,
        raise_exceptions=False,
    )

    # Depending on ragas version, result export APIs differ.
    raw_scores: Dict[str, Any] = {}
    if isinstance(result, dict):
        raw_scores = result
    elif hasattr(result, "to_dict") and callable(getattr(result, "to_dict")):
        maybe_scores = result.to_dict()
        if isinstance(maybe_scores, dict):
            raw_scores = maybe_scores
    elif hasattr(result, "scores"):
        maybe_scores = getattr(result, "scores")
        if isinstance(maybe_scores, dict):
            raw_scores = maybe_scores
    elif hasattr(result, "to_pandas") and callable(getattr(result, "to_pandas")):
        df = result.to_pandas()
        if len(df.index) > 0:
            first_row = df.iloc[0].to_dict()
            if isinstance(first_row, dict):
                raw_scores = first_row

    score_map: Dict[str, float] = {}
    for key, value in raw_scores.items():
        if isinstance(value, (int, float)):
            score_map[str(key)] = float(value)

    release_loaded_connectors()
    gc.collect()

    return score_map, metric_names


def write_aggregate_csv(output_csv: str, score_map: Dict[str, float], sample_count: int) -> None:
    """Write final aggregate metric scores to local CSV."""
    fieldnames = ["sample_count"] + sorted(score_map.keys())

    with Path(output_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        row: Dict[str, Any] = {"sample_count": sample_count}
        row.update(score_map)
        writer.writerow(row)


def write_detailed_csv(output_csv: str, rows: List[Dict[str, Any]]) -> None:
    """Write per-sample outputs for debugging and score interpretation."""
    fieldnames = ["question", "ground_truth", "answer", "retrieval_query", "contexts_json"]

    with Path(output_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "answer": row["answer"],
                    "retrieval_query": row["retrieval_query"],
                    "contexts_json": json.dumps(row["contexts"], ensure_ascii=False),
                }
            )


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Local MedQA evaluation for Medical RAG")
    parser.add_argument(
        "--medqa-jsonl",
        default="./MedQA-USMLE",
        help="Path to MedQA JSONL file OR MedQA-USMLE root folder",
    )
    parser.add_argument("--medqa-split", default="dev", choices=["train", "dev", "test"], help="MedQA split")
    parser.add_argument("--medqa-region", default="US", help="MedQA region folder, e.g. US")
    parser.add_argument("--subset-size", type=int, default=30, help="Number of questions to evaluate")
    parser.add_argument("--persist-directory", default="./local_chroma_db", help="Local Chroma DB path")
    parser.add_argument("--collection-name", default="pmc_medical_literature", help="Chroma collection name")
    parser.add_argument("--output-csv", default="./evaluation_scores.csv", help="Aggregate score CSV path")
    parser.add_argument("--output-detailed-csv", default="./evaluation_detailed.csv", help="Detailed run CSV path")
    parser.add_argument("--top-k", type=int, default=4, help="Retriever top-k (must be 3-5)")
    parser.add_argument("--max-chunk-chars", type=int, default=1400, help="Max chars per chunk in synthesis")
    parser.add_argument("--max-total-context-chars", type=int, default=9000, help="Max chars for synthesis context")

    args = parser.parse_args()

    resolved_medqa_path = resolve_medqa_jsonl_path(
        path_or_root=args.medqa_jsonl,
        split=args.medqa_split,
        region=args.medqa_region,
    )

    return EvalConfig(
        medqa_jsonl_path=resolved_medqa_path,
        subset_size=args.subset_size,
        persist_directory=args.persist_directory,
        collection_name=args.collection_name,
        output_csv=args.output_csv,
        output_detailed_csv=args.output_detailed_csv,
        top_k=args.top_k,
        max_chunk_chars=args.max_chunk_chars,
        max_total_context_chars=args.max_total_context_chars,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = parse_args()
    config.validate()

    LOGGER.info("Loading MedQA subset from %s", config.medqa_jsonl_path)
    medqa_subset = load_medqa_subset(config.medqa_jsonl_path, config.subset_size)
    LOGGER.info("Loaded %s MedQA samples", len(medqa_subset))

    orchestrator_config = OrchestratorConfig(
        persist_directory=config.persist_directory,
        collection_name=config.collection_name,
        top_k=config.top_k,
        max_chunk_chars=config.max_chunk_chars,
        max_total_context_chars=config.max_total_context_chars,
        extraction_retries=1,
    )

    benchmark_rows = run_orchestrator_benchmark(
        medqa_records=medqa_subset,
        orchestrator_config=orchestrator_config,
    )

    if not benchmark_rows:
        raise RuntimeError("No successful benchmark rows were produced; cannot run evaluation")

    LOGGER.info("Building Ragas dataset with %s rows", len(benchmark_rows))
    ragas_dataset = build_ragas_dataset(benchmark_rows)

    LOGGER.info("Running local Ragas evaluation (sequential batch_size=1)")
    score_map, metric_names = evaluate_with_ragas(ragas_dataset)

    write_aggregate_csv(config.output_csv, score_map, sample_count=len(benchmark_rows))
    write_detailed_csv(config.output_detailed_csv, benchmark_rows)

    LOGGER.info("Completed evaluation with metrics: %s", ", ".join(metric_names))
    LOGGER.info("Aggregate scores saved to %s", config.output_csv)
    LOGGER.info("Detailed outputs saved to %s", config.output_detailed_csv)


if __name__ == "__main__":
    main()
