"""Bridge structured MIMIC-IV demo tables into note-like text for Phase 3.

This script creates synthetic clinical notes from structured hospital tables and
runs them through the existing sequential agent orchestrator.

Rationale:
- The MIMIC-IV demo dataset in this workspace does not include free-text notes.
- We generate a compact note surrogate using demographics, admission context,
  diagnosis descriptions, and medications.

Design constraints:
- Fully local execution only.
- Sequential processing only (no async/concurrency).
- Memory-aware loading and per-record processing.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

from agent_orchestrator import OrchestratorConfig, run_pipeline

LOGGER = logging.getLogger("mimic_note_bridge")


@dataclass
class BridgeConfig:
    """Runtime settings for MIMIC structured-to-note bridge."""

    mimic_root: str = "./mimic-iv-clinical-database-demo-2.2"
    max_admissions: int = 20
    output_jsonl: str = "./mimic_bridge_outputs.jsonl"
    persist_directory: str = "./local_chroma_db"
    collection_name: str = "pmc_medical_literature"
    top_k: int = 4
    max_chunk_chars: int = 1400
    max_total_context_chars: int = 9000


def _csv_reader(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def _load_patient_demographics(hosp_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load anchor demographics keyed by subject_id."""
    patients_path = hosp_dir / "patients.csv"
    demo: Dict[str, Dict[str, str]] = {}

    for row in _csv_reader(patients_path):
        subject_id = str(row.get("subject_id", "")).strip()
        if not subject_id:
            continue
        demo[subject_id] = {
            "gender": str(row.get("gender", "")).strip(),
            "anchor_age": str(row.get("anchor_age", "")).strip(),
            "anchor_year": str(row.get("anchor_year", "")).strip(),
        }

    return demo


def _load_icd_dictionary(hosp_dir: Path) -> Dict[Tuple[str, str], str]:
    """Load ICD code-to-title map using d_icd_diagnoses.csv."""
    dictionary_path = hosp_dir / "d_icd_diagnoses.csv"
    mapping: Dict[Tuple[str, str], str] = {}

    for row in _csv_reader(dictionary_path):
        code = str(row.get("icd_code", "")).strip()
        version = str(row.get("icd_version", "")).strip()
        title = str(row.get("long_title", "")).strip()
        if code and version and title:
            mapping[(code, version)] = title

    return mapping


def _load_diagnoses_by_hadm(hosp_dir: Path, icd_map: Dict[Tuple[str, str], str]) -> DefaultDict[str, List[str]]:
    """Load diagnoses grouped by hospital admission ID."""
    diagnoses_path = hosp_dir / "diagnoses_icd.csv"
    grouped: DefaultDict[str, List[str]] = defaultdict(list)

    for row in _csv_reader(diagnoses_path):
        hadm_id = str(row.get("hadm_id", "")).strip()
        if not hadm_id:
            continue

        code = str(row.get("icd_code", "")).strip()
        version = str(row.get("icd_version", "")).strip()
        title = icd_map.get((code, version), f"ICD-{version} {code}")

        if title and title not in grouped[hadm_id]:
            grouped[hadm_id].append(title)

    return grouped


def _load_meds_by_hadm(hosp_dir: Path) -> DefaultDict[str, List[str]]:
    """Load medications grouped by hospital admission ID."""
    meds_path = hosp_dir / "prescriptions.csv"
    grouped: DefaultDict[str, List[str]] = defaultdict(list)

    for row in _csv_reader(meds_path):
        hadm_id = str(row.get("hadm_id", "")).strip()
        drug = str(row.get("drug", "")).strip()
        if not hadm_id or not drug:
            continue
        if drug not in grouped[hadm_id]:
            grouped[hadm_id].append(drug)

    return grouped


def _format_list(items: List[str], limit: int = 12) -> str:
    if not items:
        return "Not documented"
    trimmed = items[:limit]
    return "; ".join(trimmed)


def synthesize_note_from_admission(
    admission_row: Dict[str, str],
    demographics: Dict[str, Dict[str, str]],
    diagnoses_by_hadm: DefaultDict[str, List[str]],
    meds_by_hadm: DefaultDict[str, List[str]],
) -> str:
    """Build a compact synthetic note string from structured MIMIC data."""
    subject_id = str(admission_row.get("subject_id", "")).strip()
    hadm_id = str(admission_row.get("hadm_id", "")).strip()

    demo = demographics.get(subject_id, {})
    age = demo.get("anchor_age", "Unknown")
    gender = demo.get("gender", "Unknown")

    admission_type = str(admission_row.get("admission_type", "")).strip() or "Unknown"
    admission_location = str(admission_row.get("admission_location", "")).strip() or "Unknown"
    discharge_location = str(admission_row.get("discharge_location", "")).strip() or "Unknown"
    race = str(admission_row.get("race", "")).strip() or "Unknown"
    language = str(admission_row.get("language", "")).strip() or "Unknown"

    diagnoses_text = _format_list(diagnoses_by_hadm.get(hadm_id, []))
    meds_text = _format_list(meds_by_hadm.get(hadm_id, []))

    return (
        "MIMIC Structured Clinical Summary\n"
        f"Subject ID: {subject_id}\n"
        f"Hospital Admission ID: {hadm_id}\n"
        f"Demographics: {age}-year-old, gender {gender}, race {race}, language {language}.\n"
        f"Encounter: {admission_type} admission from {admission_location}, discharge to {discharge_location}.\n"
        f"Diagnoses: {diagnoses_text}.\n"
        f"Current/Prescribed Medications: {meds_text}.\n"
        "Symptoms are not directly available in this demo table subset; infer only from documented diagnoses/medications if explicitly supported."
    )


def run_bridge(config: BridgeConfig) -> Dict[str, int]:
    """Run sequential structured-to-note conversion and orchestrator inference."""
    mimic_root = Path(config.mimic_root)
    hosp_dir = mimic_root / "hosp"
    admissions_path = hosp_dir / "admissions.csv"

    if not admissions_path.is_file():
        raise FileNotFoundError(f"Missing admissions file at {admissions_path}")

    LOGGER.info("Loading patient demographics")
    demographics = _load_patient_demographics(hosp_dir)

    LOGGER.info("Loading ICD dictionary")
    icd_map = _load_icd_dictionary(hosp_dir)

    LOGGER.info("Loading diagnoses grouped by admission")
    diagnoses_by_hadm = _load_diagnoses_by_hadm(hosp_dir, icd_map)

    LOGGER.info("Loading medications grouped by admission")
    meds_by_hadm = _load_meds_by_hadm(hosp_dir)

    orchestrator_config = OrchestratorConfig(
        persist_directory=config.persist_directory,
        collection_name=config.collection_name,
        top_k=config.top_k,
        max_chunk_chars=config.max_chunk_chars,
        max_total_context_chars=config.max_total_context_chars,
        extraction_retries=1,
    )

    processed = 0
    failed = 0

    output_path = Path(config.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting sequential bridge run for up to %s admissions", config.max_admissions)

    with output_path.open("w", encoding="utf-8") as out_handle:
        for row in _csv_reader(admissions_path):
            if processed >= config.max_admissions:
                break

            processed += 1
            hadm_id = str(row.get("hadm_id", "")).strip()
            LOGGER.info("Processing admission %s (%s/%s)", hadm_id or "unknown", processed, config.max_admissions)

            note_text = synthesize_note_from_admission(
                admission_row=row,
                demographics=demographics,
                diagnoses_by_hadm=diagnoses_by_hadm,
                meds_by_hadm=meds_by_hadm,
            )

            try:
                pipeline_out = run_pipeline(note_text=note_text, config=orchestrator_config)
                record = {
                    "subject_id": row.get("subject_id"),
                    "hadm_id": row.get("hadm_id"),
                    "synthetic_note": note_text,
                    "extraction": pipeline_out.extraction.model_dump(),
                    "retrieval_query": pipeline_out.retrieval_query,
                    "retrieved_chunks": pipeline_out.retrieved_chunks,
                    "synthesis": pipeline_out.synthesis,
                }
            except Exception as exc:  # pragma: no cover
                failed += 1
                LOGGER.warning("Pipeline failed for hadm_id=%s: %s", hadm_id, exc)
                record = {
                    "subject_id": row.get("subject_id"),
                    "hadm_id": row.get("hadm_id"),
                    "synthetic_note": note_text,
                    "error": str(exc),
                }

            out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "processed": processed,
        "failed": failed,
        "succeeded": processed - failed,
        "output_records": processed,
    }


def parse_args() -> BridgeConfig:
    parser = argparse.ArgumentParser(description="Run MIMIC structured-to-note bridge with local orchestrator")
    parser.add_argument(
        "--mimic-root",
        default="./mimic-iv-clinical-database-demo-2.2",
        help="Path to MIMIC-IV demo root folder",
    )
    parser.add_argument("--max-admissions", type=int, default=20, help="Number of admissions to process")
    parser.add_argument(
        "--output-jsonl",
        default="./mimic_bridge_outputs.jsonl",
        help="Output JSONL path for per-admission orchestration outputs",
    )
    parser.add_argument("--persist-directory", default="./local_chroma_db", help="Local Chroma DB path")
    parser.add_argument("--collection-name", default="pmc_medical_literature", help="Chroma collection name")
    parser.add_argument("--top-k", type=int, default=4, help="Retriever top-k (must be 3-5)")
    parser.add_argument("--max-chunk-chars", type=int, default=1400, help="Max chars per chunk in synthesis")
    parser.add_argument("--max-total-context-chars", type=int, default=9000, help="Max synthesis context chars")

    args = parser.parse_args()

    return BridgeConfig(
        mimic_root=args.mimic_root,
        max_admissions=args.max_admissions,
        output_jsonl=args.output_jsonl,
        persist_directory=args.persist_directory,
        collection_name=args.collection_name,
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
    if config.top_k < 3 or config.top_k > 5:
        raise ValueError("top_k must be between 3 and 5")

    summary = run_bridge(config)
    LOGGER.info("Bridge run complete: %s", summary)


if __name__ == "__main__":
    main()
