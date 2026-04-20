"""Streamlit dashboard for the local Medical RAG system.

Tabs:
- Clinical Assistant
- Live System Logs
- Evaluation Dashboard

Key design constraints:
- Fully local operation and graceful failures.
- Sequential workflow execution (single active run).
- Session state persistence to avoid accidental recomputation.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from langchain_chroma import Chroma

from agent_orchestrator import OrchestratorConfig, extraction_agent, retrieval_agent, synthesis_agent
from backend_logging import LOG_FILE_PATH, configure_backend_logging
from model_setup import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_REASONING_MODEL,
    ensure_local_ollama_models,
    get_embedding_engine,
    get_reasoning_engine,
    list_local_models,
    release_loaded_connectors,
)

TARGET_REASONING_MODEL = "llama3:8b-instruct"
TARGET_EMBEDDING_MODEL = "nomic-embed-text"

DEFAULT_COLLECTION = "pmc_medical_literature"
DEFAULT_PERSIST_DIR = "./local_chroma_db"
DEFAULT_EVAL_CSV = "./evaluation_scores.csv"
DEFAULT_EVAL_DETAILED_CSV = "./evaluation_detailed.csv"
MAX_LOG_LINES = 200
PROJECT_DIR = Path(__file__).resolve().parent
DEMO_SCRIPT_PATH = PROJECT_DIR / "run_demo_fast.sh"


class WorkflowRunner:
    """Background sequential runner for a single workflow at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._result: Optional[Dict[str, Any]] = None
        self._error: Optional[str] = None

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def start(self, note_text: str, persist_directory: str, collection_name: str) -> bool:
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._result = None
            self._error = None

        worker = threading.Thread(
            target=self._run,
            args=(note_text, persist_directory, collection_name),
            daemon=True,
            name="clinical-workflow-runner",
        )
        self._thread = worker
        worker.start()
        return True

    def _run(self, note_text: str, persist_directory: str, collection_name: str) -> None:
        try:
            configure_backend_logging()
            ensure_local_ollama_models()

            config = OrchestratorConfig(
                persist_directory=persist_directory,
                collection_name=collection_name,
                top_k=4,
                max_chunk_chars=1400,
                max_total_context_chars=9000,
                extraction_retries=1,
            )

            extraction = extraction_agent(note_text=note_text, extraction_retries=config.extraction_retries)
            retrieval_query, retrieved_chunks = retrieval_agent(
                extraction=extraction,
                persist_directory=config.persist_directory,
                collection_name=config.collection_name,
                top_k=config.top_k,
            )
            synthesis = synthesis_agent(
                extraction=extraction,
                retrieved_chunks=retrieved_chunks,
                max_chunk_chars=config.max_chunk_chars,
                max_total_context_chars=config.max_total_context_chars,
            )

            result = {
                "extraction": extraction.model_dump(),
                "retrieval_query": retrieval_query,
                "retrieved_chunks": retrieved_chunks,
                "synthesis": synthesis,
            }

            with self._lock:
                self._result = result
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
        finally:
            release_loaded_connectors()
            with self._lock:
                self._running = False

    def pop_result(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            result = self._result
            self._result = None
            return result

    def pop_error(self) -> Optional[str]:
        with self._lock:
            error = self._error
            self._error = None
            return error


class DemoScriptRunner:
    """Background runner for one-click fast demo script execution."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._error: Optional[str] = None
        self._completed: bool = False

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def start(self, script_path: Path, cwd: Path, log_path: Path) -> bool:
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._error = None
            self._completed = False

        worker = threading.Thread(
            target=self._run,
            args=(script_path, cwd, log_path),
            daemon=True,
            name="demo-script-runner",
        )
        self._thread = worker
        worker.start()
        return True

    def _run(self, script_path: Path, cwd: Path, log_path: Path) -> None:
        try:
            configure_backend_logging()
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write("[INFO] Demo Runner: Starting run_demo_fast.sh...\n")
                log_handle.flush()

                process = subprocess.Popen(
                    ["bash", str(script_path)],
                    cwd=str(cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                if process.stdout is not None:
                    for line in process.stdout:
                        log_handle.write(f"[INFO] Demo Runner: {line.rstrip()}\n")
                        log_handle.flush()

                return_code = process.wait()
                if return_code != 0:
                    raise RuntimeError(f"run_demo_fast.sh failed with exit code {return_code}")

                log_handle.write("[INFO] Demo Runner: run_demo_fast.sh completed successfully.\n")
                log_handle.flush()

            with self._lock:
                self._completed = True
        except Exception as exc:
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(f"[ERROR] Demo Runner: {exc}\n")
                log_handle.flush()
            with self._lock:
                self._error = str(exc)
        finally:
            with self._lock:
                self._running = False

    def pop_error(self) -> Optional[str]:
        with self._lock:
            error = self._error
            self._error = None
            return error

    def pop_completed(self) -> bool:
        with self._lock:
            completed = self._completed
            self._completed = False
            return completed


@st.cache_resource(show_spinner=False)
def get_workflow_runner() -> WorkflowRunner:
    return WorkflowRunner()


@st.cache_resource(show_spinner=False)
def get_demo_script_runner() -> DemoScriptRunner:
    return DemoScriptRunner()


@st.cache_resource(show_spinner=False)
def init_backend_logging() -> Path:
    return configure_backend_logging()


@st.cache_resource(show_spinner=False)
def get_cached_llm() -> Any:
    """Cache local reasoning connector in process memory."""
    return get_reasoning_engine(temperature=0.0)


@st.cache_resource(show_spinner=False)
def get_cached_embeddings() -> Any:
    """Cache local embedding connector in process memory."""
    return get_embedding_engine()


@st.cache_resource(show_spinner=False)
def get_cached_vector_store(persist_directory: str, collection_name: str) -> Chroma:
    """Cache local Chroma connection in process memory."""
    embeddings = get_cached_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


def _init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "is_processing": False,
        "last_note": "",
        "extraction": None,
        "retrieval_query": None,
        "retrieved_chunks": None,
        "synthesis": None,
        "last_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _read_log_tail(log_path: Path, max_lines: int = MAX_LOG_LINES) -> str:
    if not log_path.exists():
        return "[INFO] No logs yet. Run the diagnostic workflow to generate backend logs."

    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    tail = lines[-max_lines:]
    return "\n".join(tail) if tail else "[INFO] Log file is empty."


def _extract_latest_demo_payload(log_path: Path) -> Optional[Dict[str, Any]]:
    """Extract the latest JSON payload emitted by Demo Runner from backend logs."""
    if not log_path.exists():
        return None

    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    prefix = "[INFO] Demo Runner:"

    collected_blocks: List[str] = []
    collecting = False
    brace_balance = 0
    block_lines: List[str] = []

    for line in lines:
        if not line.startswith(prefix):
            continue

        payload_line = line[len(prefix) :].lstrip()
        if not collecting and payload_line.startswith("{"):
            collecting = True
            block_lines = [payload_line]
            brace_balance = payload_line.count("{") - payload_line.count("}")
            if brace_balance <= 0:
                collected_blocks.append("\n".join(block_lines))
                collecting = False
            continue

        if collecting:
            block_lines.append(payload_line)
            brace_balance += payload_line.count("{") - payload_line.count("}")
            if brace_balance <= 0:
                collected_blocks.append("\n".join(block_lines))
                collecting = False

    for raw_block in reversed(collected_blocks):
        try:
            parsed = json.loads(raw_block)
            if isinstance(parsed, dict) and "extraction" in parsed:
                return parsed
        except Exception:
            continue

    return None


def _clear_logs(log_path: Path) -> None:
    log_path.write_text("", encoding="utf-8")


def _highlight_citations(text: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        citation = match.group(0)
        return (
            "<span style='background-color:#eef6ff;border:1px solid #aac7f5;"
            "padding:1px 6px;border-radius:6px;font-weight:600;'>"
            f"{citation}</span>"
        )

    return re.sub(r"\[Doc\s+\d+\]", replacer, text)


def _load_eval_csv(path: str) -> Optional[pd.DataFrame]:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None


def _compute_batch_accuracy(detailed_df: pd.DataFrame, batch_size: int = 5) -> pd.DataFrame:
    df = detailed_df.copy()
    if "ground_truth" not in df.columns or "answer" not in df.columns:
        return pd.DataFrame(columns=["batch", "accuracy"])

    def _is_match(row: pd.Series) -> float:
        truth = str(row["ground_truth"]).strip().lower()
        pred = str(row["answer"]).strip().lower()
        return 1.0 if truth and truth in pred else 0.0

    df["match"] = df.apply(_is_match, axis=1)
    df["idx"] = range(1, len(df) + 1)
    df["batch"] = ((df["idx"] - 1) // batch_size) + 1

    grouped = df.groupby("batch", as_index=False)["match"].mean()
    grouped.rename(columns={"match": "accuracy"}, inplace=True)
    grouped["batch"] = grouped["batch"].astype(str)
    return grouped


def _sync_runner_outputs() -> None:
    runner = get_workflow_runner()
    demo_runner = get_demo_script_runner()
    st.session_state["is_processing"] = runner.is_running() or demo_runner.is_running()

    result = runner.pop_result()
    if result is not None:
        st.session_state["extraction"] = result.get("extraction")
        st.session_state["retrieval_query"] = result.get("retrieval_query")
        st.session_state["retrieved_chunks"] = result.get("retrieved_chunks")
        st.session_state["synthesis"] = result.get("synthesis")
        st.session_state["last_error"] = None

    error = runner.pop_error()
    if error is not None:
        st.session_state["last_error"] = error

    demo_error = demo_runner.pop_error()
    if demo_error is not None:
        st.session_state["last_error"] = demo_error

    if demo_runner.pop_completed():
        st.session_state["demo_completed_message"] = "Fast demo pipeline completed successfully."


def render_sidebar() -> Dict[str, str]:
    with st.sidebar:
        st.title("Medical RAG Dashboard")

        st.markdown("---")
        st.subheader("System Status")
        st.caption("Pipeline mode: 100% local execution")
        st.write(f"Target reasoning model: {TARGET_REASONING_MODEL}")
        st.write(f"Target embedding model: {TARGET_EMBEDDING_MODEL}")
        st.caption(f"Backend defaults: {DEFAULT_REASONING_MODEL} | {DEFAULT_EMBEDDING_MODEL}")

        try:
            local_models = sorted(list_local_models())
            st.success("Ollama reachable")
            st.caption("Local models: " + (", ".join(local_models) if local_models else "none"))
        except Exception:
            st.warning("Ollama not reachable. Start it with: ollama serve")

        st.markdown("---")
        st.subheader("Runtime Settings")
        persist_directory = st.text_input("Chroma persist directory", value=DEFAULT_PERSIST_DIR)
        collection_name = st.text_input("Chroma collection", value=DEFAULT_COLLECTION)

    return {
        "persist_directory": persist_directory,
        "collection_name": collection_name,
    }


def render_clinical_tab(persist_directory: str, collection_name: str) -> None:
    st.header("Clinical Assistant")
    st.caption("Paste a raw clinical note and run the sequential local diagnostic workflow.")

    note_text = st.text_area(
        "Clinical Note",
        value=st.session_state.get("last_note", ""),
        height=220,
        placeholder="Paste a MIMIC-style clinical note here...",
    )

    run_clicked = st.button(
        "Run Diagnostic Workflow",
        type="primary",
        disabled=st.session_state.get("is_processing", False),
    )

    demo_clicked = st.button(
        "Run Full Demo Pipeline (One Click)",
        disabled=st.session_state.get("is_processing", False),
        help="Runs run_demo_fast.sh in the background and streams output to Live System Logs.",
    )

    if demo_clicked:
        if not DEMO_SCRIPT_PATH.exists():
            st.warning("run_demo_fast.sh not found in project root.")
        else:
            started = get_demo_script_runner().start(
                script_path=DEMO_SCRIPT_PATH,
                cwd=PROJECT_DIR,
                log_path=LOG_FILE_PATH,
            )
            if not started:
                st.warning("A workflow is already running. Please wait for it to finish.")

    if run_clicked:
        if not note_text.strip():
            st.warning("Please provide a clinical note before running the workflow.")
        else:
            # Warm expensive resources once per process.
            try:
                get_cached_llm()
                get_cached_embeddings()
                get_cached_vector_store(persist_directory, collection_name)
            except Exception:
                st.warning("Could not warm cached backend resources. Continuing with runtime initialization.")

            st.session_state["last_note"] = note_text
            started = get_workflow_runner().start(
                note_text=note_text,
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
            if not started:
                st.warning("A workflow is already running. Please wait for it to finish.")

    if st.session_state.get("is_processing"):
        st.info("Workflow is running sequentially. You can monitor progress in the Live System Logs tab.")

    demo_msg = st.session_state.pop("demo_completed_message", None)
    if demo_msg:
        st.success(demo_msg)

    if st.session_state.get("last_error"):
        st.warning(
            "Unable to complete workflow. "
            "Please verify Ollama service is running and local models are available."
        )
        st.caption(f"Error details: {st.session_state['last_error']}")

    extraction = st.session_state.get("extraction")
    retrieved_chunks = st.session_state.get("retrieved_chunks")
    synthesis = st.session_state.get("synthesis")

    if extraction:
        st.subheader("Step 1 Output: Extracted JSON")
        st.json(extraction)

    if retrieved_chunks:
        st.subheader("Step 2 Output: Retrieved Context")
        retrieval_query = st.session_state.get("retrieval_query", "")
        if retrieval_query:
            st.caption(f"Query: {retrieval_query}")

        for chunk in retrieved_chunks:
            doc_id = chunk.get("doc_id", "Doc")
            metadata = chunk.get("metadata", {})
            pmcid = metadata.get("pmcid", "unknown")
            with st.expander(f"{doc_id} | PMCID {pmcid}"):
                st.write(chunk.get("content", ""))
                st.caption(json.dumps(metadata, ensure_ascii=False))

    if synthesis:
        st.subheader("Step 3 Output: Clinical Synthesis")
        st.markdown(_highlight_citations(synthesis), unsafe_allow_html=True)

    demo_payload = _extract_latest_demo_payload(LOG_FILE_PATH)
    if demo_payload is not None:
        st.subheader("Patient Data from Live Logs")
        st.caption("Parsed from the latest Demo Runner payload in backend_execution.log")

        extraction_payload = demo_payload.get("extraction")
        if extraction_payload:
            st.markdown("**Extraction JSON**")
            st.json(extraction_payload)

        retrieval_query = demo_payload.get("retrieval_query")
        if isinstance(retrieval_query, str) and retrieval_query.strip():
            st.markdown("**Retrieval Query**")
            st.code(retrieval_query, language="text")

        retrieved_chunks = demo_payload.get("retrieved_chunks")
        if isinstance(retrieved_chunks, list) and retrieved_chunks:
            st.markdown("**Retrieved Chunks**")
            for chunk in retrieved_chunks:
                doc_id = chunk.get("doc_id", "Doc")
                metadata = chunk.get("metadata", {})
                pmcid = metadata.get("pmcid", "unknown")
                with st.expander(f"{doc_id} | PMCID {pmcid}"):
                    st.write(chunk.get("content", ""))
                    st.caption(json.dumps(metadata, ensure_ascii=False))

        synthesis_payload = demo_payload.get("synthesis")
        if isinstance(synthesis_payload, str) and synthesis_payload.strip():
            st.markdown("**Synthesis Output**")
            st.markdown(_highlight_citations(synthesis_payload), unsafe_allow_html=True)


def render_logs_tab(log_path: Path) -> None:
    st.header("Live System Logs")
    st.caption("Filtered backend milestones from local execution (tail capped at last 200 lines).")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Clear Logs"):
            _clear_logs(log_path)
            st.success("Log file cleared.")
    with c2:
        st.caption(f"Log file: {log_path.name}")

    log_placeholder = st.empty()
    log_text = _read_log_tail(log_path, max_lines=MAX_LOG_LINES)
    log_placeholder.code(log_text, language="bash")

    if st.session_state.get("is_processing"):
        st.info("Workflow running. Logs auto-refresh every 2 seconds.")


def render_evaluation_tab() -> None:
    st.header("Evaluation Dashboard")
    st.caption("Reads local benchmark outputs generated by evaluation_pipeline.py")

    aggregate_path = st.text_input("Aggregate CSV", value=DEFAULT_EVAL_CSV)
    detailed_path = st.text_input("Detailed CSV", value=DEFAULT_EVAL_DETAILED_CSV)

    aggregate_df = _load_eval_csv(aggregate_path)
    if aggregate_df is None or aggregate_df.empty:
        st.warning("Aggregate evaluation CSV not found yet. Run evaluation_pipeline.py first.")
        return

    row = aggregate_df.iloc[-1]
    faithfulness = row.get("faithfulness", None)
    context_precision = row.get("context_precision", None)

    c1, c2 = st.columns(2)
    with c1:
        if isinstance(faithfulness, (int, float)):
            st.metric("Faithfulness", f"{faithfulness:.3f}")
        else:
            st.metric("Faithfulness", "N/A")

    with c2:
        if isinstance(context_precision, (int, float)):
            st.metric("Context Precision", f"{context_precision:.3f}")
        else:
            st.metric("Context Precision", "N/A")
            st.caption("Context Precision is not present in the current CSV schema.")

    st.subheader("Batch Accuracy Trend")
    detailed_df = _load_eval_csv(detailed_path)
    if detailed_df is None or detailed_df.empty:
        st.info("Detailed CSV not found. Run evaluation with --output-detailed-csv to enable batch chart.")
        return

    batch_df = _compute_batch_accuracy(detailed_df, batch_size=5)
    if batch_df.empty:
        st.info("Detailed CSV is missing required columns: ground_truth and answer.")
        return

    st.bar_chart(batch_df.set_index("batch")["accuracy"])
    st.caption("Accuracy uses a local exact-string containment proxy by batch (size=5).")


def main() -> None:
    st.set_page_config(page_title="Local Medical RAG Dashboard", page_icon="+", layout="wide")
    init_backend_logging()
    _init_session_state()
    _sync_runner_outputs()

    sidebar = render_sidebar()

    clinical_tab, logs_tab, eval_tab = st.tabs(
        ["Clinical Assistant", "Live System Logs", "Evaluation Dashboard"]
    )

    with clinical_tab:
        render_clinical_tab(
            persist_directory=sidebar["persist_directory"],
            collection_name=sidebar["collection_name"],
        )

    with logs_tab:
        render_logs_tab(LOG_FILE_PATH)

    with eval_tab:
        render_evaluation_tab()

    # Polling loop for near real-time updates while long workflow is running.
    if st.session_state.get("is_processing"):
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()
