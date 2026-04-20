# Local Agentic Medical RAG (Privacy-First Clinical Decision Support)

## 1) Why This Project Exists

Modern clinical decision support systems are often blocked by three practical constraints:

1. Data privacy risk: sensitive patient data should not leave local infrastructure.
2. Hallucination risk: generic LLM responses can introduce unsupported medical claims.
3. Hardware constraints: many student/research environments run on limited local hardware (for example unified-memory laptops) instead of large GPU servers.

This project is designed to address those constraints directly with a local-first, retrieval-grounded architecture:

- Local-only model execution via Ollama.
- Retrieval-Augmented Generation (RAG) from indexed biomedical literature.
- Strict intermediate structures (JSON handoffs between agents).
- Sequential execution to reduce memory pressure and avoid thermal collapse.

The core objective is not to "train" a new model from scratch, but to orchestrate robust local reasoning on top of existing local models and curated retrieval context.

---

## 2) High-Level Logic

At a systems level, the pipeline has four phases:

1. Model Provisioning (Phase 1)
- Ensure required local Ollama models are available.
- Initialize reasoning and embedding connectors lazily.

2. Knowledge Base Build (Phase 2)
- Query PMC open-access literature through NCBI E-utilities.
- Parse XML, clean noise, split text semantically.
- Embed chunks locally and persist them in Chroma.

3. Agentic Orchestration (Phase 3)
- Extraction Agent: note text -> strict JSON.
- Retrieval Agent: JSON -> semantic query -> top-k chunk retrieval.
- Synthesis Agent: JSON + retrieved chunks -> grounded summary with citations.

4. Evaluation (Phase 4)
- Benchmark on MedQA subset.
- Use local LLM-as-judge setup (Ragas) for faithfulness and answer quality metrics.

UI phases then expose this in Streamlit with live logs and one-click demo runs.

---

## 3) Core Design Principles

1. Privacy-first
- No external hosted LLM API is required for core operation.
- Data stays on machine.

2. Sequential over concurrent
- All expensive operations are intentionally serialized.
- Prevents unified-memory spikes and swap-heavy behavior.

3. Deterministic handoffs
- Structured extraction schema (Pydantic + JSON parser).
- Explicit retrieval context passed to synthesis.

4. Operational observability
- Milestone logging to rotating local log file.
- Streamlit live log viewer for operator visibility.

5. Demo practicality
- Fast-mode script to run reduced subsets in minutes.

---

## 4) Data Sources

### 4.1 MedQA-USMLE
Path:
- `./MedQA-USMLE`

Use in system:
- Evaluation benchmark only.
- Questions are converted to note-like prompts for pipeline testing.

### 4.2 MIMIC-IV Demo
Path:
- `./mimic-iv-clinical-database-demo-2.2`

Important note:
- Demo subset here contains structured tables, not raw note events.

How this project handles it:
- `mimic_note_bridge.py` synthesizes note-like text from structured fields:
  - admissions
  - diagnoses (via ICD mapping)
  - prescriptions
  - demographics

### 4.3 PMC Literature (Dynamic Ingestion)
Source:
- NCBI E-utilities API (open access filter)

Use in system:
- Retrieval corpus for the RAG store.

---

## 5) External API Usage

### 5.1 NCBI E-search
Purpose:
- Discover relevant open-access PMC IDs.

Pattern:
- `esearch.fcgi?db=pmc&term=<query>+AND+open+access[filter]&retmode=json`

### 5.2 NCBI E-fetch
Purpose:
- Fetch full XML article content by PMCID.

Pattern:
- `efetch.fcgi?db=pmc&id=<pmcid>&retmode=xml`

### 5.3 Rate-limiting
Implemented policy:
- Mandatory sleep between calls (`1.5s`) in ingestion script.

Why:
- Avoid NCBI throttling / IP ban.

---

## 6) Models and Runtime

### 6.1 Reasoning Model
Current backend default:
- `llama3.2:latest`

UI target label (informational):
- `llama3:8b-instruct`

Why this mismatch may appear:
- The runtime was adjusted to a pullable tag in local Ollama registry while preserving target intent in UI messaging.

### 6.2 Embedding Model
Default:
- `nomic-embed-text`

### 6.3 Vector Store
- Chroma local persistent directory (default `./local_chroma_db`).

### 6.4 Orchestration Stack
- LangChain-style components + custom workflow sequencing.

---

## 7) Codebase Walkthrough

### 7.1 `model_setup.py`
Responsibilities:
- Local model registry checks via Ollama client.
- Automatic pull of missing required models.
- Lazy connector manager for Chat + Embeddings.
- Mutual release behavior to avoid keeping both heavy connectors active simultaneously.

Key behavior:
- Tagged/untagged model matching (`model:latest` vs `model`).

### 7.2 `knowledge_builder.py`
Responsibilities:
- PMC search and fetch.
- XML parsing and cleanup.
- Semantic chunking (`chunk_size=1200`, `chunk_overlap=200`).
- Sequential embedding and Chroma persistence.

Key behavior:
- Strictly sequential doc-by-doc ingestion.

### 7.3 `agent_orchestrator.py`
Responsibilities:
- Multi-agent sequence:
  - extraction
  - retrieval
  - synthesis
- Strict JSON extraction via parser + Pydantic schema.
- Context-bounded synthesis with inline citation prompt constraints.

### 7.4 `mimic_note_bridge.py`
Responsibilities:
- Convert structured MIMIC demo rows into synthetic note strings.
- Run orchestrator for each admission sequentially.
- Emit JSONL per case output.

### 7.5 `evaluation_pipeline.py`
Responsibilities:
- MedQA loading and subsetting.
- Sequential benchmark loop through orchestrator.
- Local Ragas judge integration.
- CSV output generation for aggregate + detailed results.

Known platform caveat:
- If Python lacks `_lzma`, HuggingFace datasets import can fail.

### 7.6 `backend_logging.py`
Responsibilities:
- Shared rotating-file milestone logging setup.
- `backend_execution.log` as local log sink.
- Rotation controls to prevent indefinite log growth.

### 7.7 `app.py`
Responsibilities:
- Streamlit dashboard with tabs:
  - Clinical Assistant
  - Live System Logs
  - Evaluation Dashboard
- One-click demo script execution in background.
- Clinical workflow execution with session state persistence.
- Live log tail view and clear action.
- Patient data extraction from log payload for UI display.

### 7.8 `run_all.sh`
Purpose:
- Full end-to-end run profile (slow, comprehensive).

### 7.9 `run_demo_fast.sh`
Purpose:
- Fast demonstration profile (reduced data sizes).
- Optional stage toggles.
- Graceful skip of evaluation when `_lzma` is unavailable.

### 7.10 `requirements.txt`
Purpose:
- Core dependencies for local execution.

---

## 8) Parameter Reference

## 8.1 `knowledge_builder.py`

| Parameter | Default | Meaning |
|---|---:|---|
| `--search-term` | required | Literature query string |
| `--max-results` | 5 | Number of PMC papers to ingest |
| `--chunk-size` | 1200 | Chunk length for semantic splitting |
| `--chunk-overlap` | 200 | Overlap between chunks |
| `--persist-directory` | `./local_chroma_db` | Chroma persistence path |
| `--collection-name` | `pmc_medical_literature` | Chroma collection name |
| `--ollama-host` | `None` | Optional custom Ollama endpoint |

## 8.2 `agent_orchestrator.py`

| Parameter | Default | Meaning |
|---|---:|---|
| `--note-file` | None | Input note file path |
| `--note-text` | None | Raw note string |
| `--persist-directory` | `./local_chroma_db` | Chroma persistence path |
| `--collection-name` | `pmc_medical_literature` | Chroma collection |
| `--top-k` | 4 | Retrieval results count (3-5 expected) |
| `--max-chunk-chars` | 1400 | Per-chunk synthesis truncation bound |
| `--max-total-context-chars` | 9000 | Total synthesis context bound |
| `--extraction-retries` | 1 | JSON extraction retry count |

## 8.3 `mimic_note_bridge.py`

| Parameter | Default | Meaning |
|---|---:|---|
| `--mimic-root` | `./mimic-iv-clinical-database-demo-2.2` | MIMIC demo root |
| `--max-admissions` | 20 | Number of admissions processed |
| `--output-jsonl` | `./mimic_bridge_outputs.jsonl` | Output JSONL |
| `--persist-directory` | `./local_chroma_db` | Chroma path |
| `--collection-name` | `pmc_medical_literature` | Chroma collection |
| `--top-k` | 4 | Retrieval top-k |
| `--max-chunk-chars` | 1400 | Synthesis chunk bound |
| `--max-total-context-chars` | 9000 | Synthesis context bound |

## 8.4 `evaluation_pipeline.py`

| Parameter | Default | Meaning |
|---|---:|---|
| `--medqa-jsonl` | `./MedQA-USMLE` | File path or MedQA root |
| `--medqa-split` | `dev` | `train`/`dev`/`test` split |
| `--medqa-region` | `US` | Region subfolder |
| `--subset-size` | 30 | Number of questions evaluated |
| `--persist-directory` | `./local_chroma_db` | Chroma path |
| `--collection-name` | `pmc_medical_literature` | Chroma collection |
| `--output-csv` | `./evaluation_scores.csv` | Aggregate score CSV |
| `--output-detailed-csv` | `./evaluation_detailed.csv` | Per-sample output CSV |
| `--top-k` | 4 | Retrieval top-k |
| `--max-chunk-chars` | 1400 | Synthesis chunk bound |
| `--max-total-context-chars` | 9000 | Synthesis context bound |

## 8.5 `run_demo_fast.sh` defaults

| Variable | Default | Effect |
|---|---:|---|
| `MAX_RESULTS` | 2 | Smaller corpus build |
| `MIMIC_MAX_ADMISSIONS` | 2 | Small bridge run |
| `MEDQA_SUBSET` | 5 | Fast evaluation subset |
| `TOP_K` | 3 | Smaller retrieval fanout |
| `RUN_BUILD_KB` | auto | Build KB only if missing |
| `RUN_SMOKE` | yes | Run single-case smoke step |
| `RUN_MIMIC_BRIDGE` | yes | Run bridge step |
| `RUN_EVAL` | yes | Run eval unless lzma missing |
| `SKIP_SETUP` | no | Skip dependency install if yes |

---

## 9) Execution Profiles

## 9.1 Full profile (slow, comprehensive)
- Use `run_all.sh`
- Typical for full end-to-end validation.
- Can take hours depending on local hardware and thermal conditions.

## 9.2 Fast demo profile (recommended for presentations)
- Use `run_demo_fast.sh`
- Reduced subset sizes and optional stage skipping.
- Designed for minutes-scale demonstrations.

## 9.3 Streamlit one-click demo
- Start app with venv Python module invocation.
- Trigger fast demo from Clinical Assistant tab.
- Monitor progress in Live System Logs tab.

---

## 10) Suggested Demo Flow (Practical)

1. Ensure Ollama service is running.
2. Launch Streamlit app.
3. Click one-click demo button.
4. Watch logs live.
5. Show parsed patient data section in Clinical Assistant.
6. Show synthesis with citations and retrieved chunks.

This demonstrates:
- local model execution,
- retrieval grounding,
- structured extraction,
- observable system pipeline.

---

## 11) Artifacts Generated

Common output artifacts:

- `./local_chroma_db` (vector store)
- `./mimic_bridge_outputs.jsonl` or `./mimic_bridge_outputs_demo.jsonl`
- `./evaluation_scores.csv` or `./evaluation_scores_demo.csv`
- `./evaluation_detailed.csv` or `./evaluation_detailed_demo.csv`
- `./backend_execution.log` and rotated backups

---

## 12) Current Limitations

1. MIMIC free-text notes are not present in demo subset.
- Bridge uses synthetic note generation from structured tables.

2. Evaluation can fail on Python builds missing `_lzma`.
- Fast demo script now detects and skips evaluation gracefully.

3. UI status model label vs runtime default may differ.
- UI target references may show `llama3:8b-instruct` intent while runtime currently uses `llama3.2:latest` for compatibility.

4. Retrieval quality depends on corpus/topic overlap.
- If corpus is narrow, clinically unrelated chunks may be returned.

---

## 13) Troubleshooting

### `ModuleNotFoundError: langchain_chroma`
Cause:
- Streamlit launched from global interpreter, not project venv.

Fix:
- Run Streamlit via:
  - `.../.venv/bin/python -m streamlit run app.py`

### `Failed to provision required Ollama models`
Cause:
- Model tag mismatch or unavailable local pull.

Fix:
- Verify with `ollama list`.
- Pull required models manually if needed.

### `_lzma` missing during evaluation
Cause:
- Python build missing lzma extension.

Fix options:
1. Rebuild Python with lzma support.
2. Use fast demo path and skip evaluation stage.

### Long runtimes / thermal throttling
Mitigations:
- Use fast demo profile.
- Reduce subset sizes further.
- Keep machine on power and reduce background load.

---

## 14) Security and Privacy Posture

This project is designed for local execution. Still, practical safety depends on your environment:

- Keep data and logs on trusted local storage.
- Avoid copying patient-like text to cloud tooling.
- Monitor log content if sharing demo recordings.

---

## 15) Future Roadmap Ideas

1. Add strict citation verification pass (claim-to-chunk validation).
2. Add retrieval reranking for stronger clinical relevance.
3. Add richer evaluation metrics (context precision/recall, calibration).
4. Add stage-selective pipeline flags in one unified runner script.
5. Add multimodal extension path (if future data includes medical images).

---

## 16) Quick Start (Concise)

1. Create venv and install requirements.
2. Start Ollama (`ollama serve`).
3. Run `run_demo_fast.sh` for fast showcase.
4. Launch Streamlit app for interactive demo.

---

This README is intentionally detailed to support both demonstration and maintainability.
