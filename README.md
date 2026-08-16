# KnowledgeManagerLLM

A conversational AI assistant that helps caregivers manage a patient's therapy schedule. The caregiver interacts in natural language; the assistant uses LLM tool-calling to add, modify, and remove therapy activities while enforcing safety checks and scheduling constraints.

## Architecture overview

```
┌─────────────────────────────────────────────────────────┐
│                     LLM (Ollama / OpenAI)               │
│          tool-calling via OpenAI-compatible API         │
└───────────────────────┬─────────────────────────────────┘
                        │ tools
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
    therapy.json    ChromaDB       PostgreSQL
   (working state)  (RAG store)   (version history)
```

### Key components

| File | Role |
|---|---|
| `chat.py` | Core `Chat` class – sends messages, executes tools, manages conversation history, runs end-of-session processing |
| `agent.py` | Base `Agent` class and specialised subclasses (`TherapyManagerAgent`, `TherapyCheckAgent`) – each agent owns its prompt, tools, context injection, and tool execution logic |
| `tools.py` | All tool implementations: CRUD on therapy activities, scheduling conflict detection, RAG lookups |
| `vector_db.py` | `VectorDBManager` – wraps ChromaDB; manages 4 collections (see below) |
| `sql_db.py` | `DatabaseManager` – wraps PostgreSQL via SQLAlchemy; stores patients and versioned therapy snapshots |
| `session_extractor.py` | End-of-session LLM extraction: saves conflict resolutions and patient preferences to ChromaDB |
| `session_manager.py` | `SessionManager` – handles runtime therapy snapshots (save/restore) and past session loading from disk |
| `log_parser.py` | Parses `chat.log` files and reconstructs conversation history for session resume |
| `scenario_loader.py` | Loads scenario files, installs therapy state, and converts therapy data to natural language for the `CaregiverAgent` |
| `prompts.py` | System prompt for the main assistant and extraction prompts for `session_extractor` |
| `config_loader.py` | Loads all settings from the `.env` file |
| `main.py` | Terminal entry point – supports both interactive and agent-driven modes |
| `chat_interface.py` | Streamlit web UI entry point |
| `test.py` | Batch test runner – executes multiple scenarios automatically and produces structured evaluation reports |

### Agent architecture

The system uses a **supervisor/worker** multi-agent pattern:

- **`TherapyManagerAgent`** (supervisor) – manages the conversation with the caregiver, handles therapy CRUD operations, and delegates to worker agents when needed.
- **`TherapyCheckAgent`** (worker) – invoked automatically before any activity is added or updated; checks compatibility with the patient's current medications, history, and preferences.
- **`CaregiverAgent`** – simulates a caregiver interacting with the system, driven by a natural language script describing objectives. Used in agent-driven mode and batch testing.
- **`JudgeAgent`** – evaluates completed conversations against the scenario objectives and produces a structured JSON report (checklist of completed, partial, failed, and not-attempted objectives).

Each agent is self-contained: it owns its system prompt, tool declarations, context injection (`inject_context`), and tool execution (`execute_tool`). `Chat` acts as the orchestrator and is the only component that knows all agents exist; agents do not know about each other.

Adding a new worker agent requires only: creating a new subclass, instantiating it in `Chat.__init__`, and registering it in `_agent_registry`. No other files need to change.

### ChromaDB collections

| Collection | R/W | Contents |
|---|---|---|
| `medicines` | Read-only at runtime | Pharmacological data indexed from `.md` files in `data/medicines/` |
| `patient_history` | Read-only at runtime | Historical safety events per patient (seeded from `data/patients/<id>/history.json`) |
| `conflict_resolutions` | Read-Write | Past scheduling conflict resolutions, extracted automatically at session end |
| `patient_preferences` | Read-Write | Patient habits and preferences, extracted automatically at session end |

### Session lifecycle

1. **Startup** – load patient from PostgreSQL → write `data/therapy.json` → seed ChromaDB collections → initialise chat with context (datetime, current activities, patient preferences).
2. **Conversation loop** – for each caregiver message the LLM may call up to 10 tools in sequence:
   - `get_medicine_data` – RAG lookup against the medicines collection (mandatory before any medicine activity).
   - `get_patient_preferences` – retrieve known patient habits for personalised suggestions.
   - `add_therapy_activity` / `update_therapy_activity` / `remove_therapy_activity` – mutate `therapy.json`; each write automatically triggers a scheduling conflict check and a patient-history safety check, and saves a runtime therapy snapshot.
   - `delegate_to_activity_checker` – delegate an activity to `TherapyCheckAgent` for safety validation before it is added.
   - `get_therapy_activities` – read the full current schedule.
   - `get_current_datetime` – get current date/time.
   - `save_session` – trigger end-of-session processing (see step 3).
3. **End of session** (triggered by `exit`/`quit` command or the `save_session` tool) –
   - LLM extracts conflict resolutions from the conversation and persists them to ChromaDB.
   - LLM extracts patient preferences from the conversation and upserts them in ChromaDB.
   - Current `therapy.json` is saved as a new versioned snapshot in PostgreSQL.

### Scheduling conflict detection

Conflicts are detected deterministically in `tools.py`:

- Activities are compared by time overlap **and** day-of-week overlap **and** `valid_from`/`valid_until` date-range overlap.
- When a conflict is found, two alternative times are suggested (anticipate / postpone).
- Past resolution hints are retrieved from ChromaDB and included in the tool response so the LLM can surface them to the caregiver.
- The LLM **never resolves conflicts autonomously** – it always asks the caregiver.

### Conversation rewind and therapy snapshots

Both the terminal and web interfaces support rewinding the conversation to any previous point:

- Every time a therapy activity is added, updated, or removed, a **therapy snapshot** is saved in memory (and persisted to `therapy_snapshots.json` in the session log folder).
- Each snapshot stores the full `therapy.json` state and a **logical message index** (counting only `user` and `assistant` messages) so that indices remain stable across save/load cycles.
- On rewind, the conversation history is truncated and the most recent snapshot at or before the rewind point is restored to `therapy.json`, keeping the therapy state consistent with the conversation.

### Past session resume

Previous sessions can be reloaded into the current context:

- Each session is logged in `logs/<patient_id>/<session_id>/` (see Data files below).
- `SessionManager` reads `chat.log` to reconstruct the conversation history and `therapy_snapshots.json` to restore the therapy timeline.
- In the Streamlit UI, past sessions are listed in the sidebar and can be loaded with one click.
- A `[LOAD]` entry is appended to the current `chat.log` whenever a past session is loaded, for traceability.

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
.venv\Scripts\activate
```

> If PowerShell rejects the activation script, run the following once per user:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

### 2. Install dependencies

From the repository root (with the venv active):

```bash
pip install -e .
```

### 3. Create the `.env` file

Copy `.env.example` to `.env` and fill in every variable:

```ini
# Two backends are configured independently:
#   MAIN — the system under test (therapy manager + checker)
#   SIM  — the test harness (simulated caregiver + judge)
# Each has its own provider AND model. Any SIM_* left empty falls back to MAIN.
# PROVIDER: openai | groq | ollama; empty = inferred from the keys below
# (OpenAI first, then Groq), falling back to Ollama.

PROVIDER=ollama
MODEL=gpt-oss:20b

SIM_PROVIDER=
SIM_MODEL=

# Credentials and endpoints belong to the provider, not the role
OLLAMA_URL=http://localhost:11434
# OPENAI_API_KEY=sk-...
# GROQ_API_KEY=gsk_...

# Example – grade a locally served model with a cloud one:
#   PROVIDER=ollama       MODEL=gpt-oss:20b
#   SIM_PROVIDER=groq     SIM_MODEL=openai/gpt-oss-20b
# Example – both on Groq, two models = two separate quotas:
#   PROVIDER=groq         MODEL=openai/gpt-oss-120b
#   SIM_PROVIDER=groq     SIM_MODEL=openai/gpt-oss-20b

# Sent on every call; supported by the gpt-oss family (low|medium|high),
# rejected by models without a reasoning mode – leave empty for those.
REASONING_EFFORT=low

# LLM request timeout in seconds
LLM_TIMEOUT=120

# PostgreSQL connection
DB_HOST=localhost
DB_PORT=5432
DB_NAME=therapy_db
DB_USER=postgres
DB_PASSWORD=password

# Patient loaded at startup
DEFAULT_PATIENT_ID=1

# Logging levels: DEBUG | INFO | WARNING | ERROR
FILE_LOG_LEVEL=DEBUG
TERMINAL_LOG_LEVEL=WARNING

# Set to 1 to log NVIDIA GPU info at startup
CHECK_NVIDIA_GPU=0
```

### 4. Start the required services

#### PostgreSQL

A running PostgreSQL instance is required. Tables are created automatically on first run.  
Refer to the [PostgreSQL documentation](https://www.postgresql.org/) for installation instructions.

#### LLM backend

**Ollama (local)** – install Ollama, pull the model, then start the server:

```bash
ollama pull qwen2.5:14b
ollama serve
```

The model must support the `/api/chat` endpoint (i.e. be a chat model, not a completion-only model).

**OpenAI** – no local server needed; just set `OPENAI_API_KEY` in `.env`.

**Groq** – no local server needed; set `GROQ_API_KEY` and `PROVIDER=groq`. Groq enforces
per-model rate limits (the free tier gives 30 RPM / 8K TPM / 1K RPD / 200K TPD on both
`openai/gpt-oss-120b` and `openai/gpt-oss-20b`), so requests are paced client-side before
being sent. The limits follow the provider of each role automatically — leave `LLM_*` /
`SIM_LLM_*` unset unless the account has a different tier. A batch stops with a clear
message when a daily quota runs out, and the tokens consumed are reported in
`logs/batch_results/<batch_id>/results.json`. On the 8K TPM free tier the per-minute budget
is the binding constraint: pacing makes a scenario take several minutes, and a long
conversation can grow past what a single request is allowed to carry.

---

## Running the application

All commands must be run from the `src/` directory.

### Terminal interface

```bash
cd src
python main.py
```

Type `exit`, `quit`, or `esci` to end the session. The therapy state is saved to PostgreSQL automatically.

#### CLI options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--input <file>` | `-i` | *(none)* | Path to a scenario script (`.md` or any text format) describing the caregiver's objectives. If omitted, runs in interactive mode. |
| `--delay <seconds>` | `-d` | `0` | Pause between turns in agent mode. Useful for readability. |

**Agent mode** – a `CaregiverAgent` reads the script and drives the conversation autonomously:

```bash
python main.py --input scenarios/1/scenario_script.md
python main.py --input scenarios/1/scenario_script.md --delay 2
```

The script is a free-form markdown file describing the caregiver's objectives. Its full content is passed verbatim to the `CaregiverAgent` — formatting and structure are the author's responsibility:

```markdown
# Add a morning walk

## Context
The patient tolerates light exercise well.

## Objectives
1. Add a 30-minute morning walk every Monday, Wednesday and Friday at 08:00.
2. If the assistant reports a conflict, follow its suggestion.
3. Once the activity is confirmed, end the conversation.
```

### Streamlit web interface

```bash
cd src
streamlit run chat_interface.py
```

---

### Batch testing

`test.py` runs multiple scenarios automatically, without human interaction, and produces structured evaluation reports. Each scenario is a self-contained folder under `scenarios/` containing a `scenario.json` file that defines the patient, the initial therapy state, and the objectives for the `CaregiverAgent`.

Each scenario runs in **stateless mode**: no database is used and `therapy.json` is overwritten at the start of each scenario from the scenario file itself, ensuring full isolation between runs.

At the end of each scenario the `JudgeAgent` evaluates the conversation against the objectives and produces a structured JSON result. All scenario logs and results are saved under `logs/batch_results/<batch_id>/`.

#### CLI options

| Flag | Default | Description |
|---|---|---|
| `--from <id>` | `1` | First scenario ID to run (inclusive). |
| `--to <id>` | same as `--from` | Last scenario ID to run (inclusive). Omit to run a single scenario. |
| `--delay <seconds>` | `0` | Pause between conversation turns. |
| `--max-turns <n>` | `30` | Maximum number of turns per scenario before forcing termination. |

#### Examples

```bash

# Run all the scenarios
python test.py

# Run a single scenario
python test.py --from 1 --to 1

# Run a range of scenarios
python test.py --from 1 --to 20
```

#### Scenario file format

Each scenario lives in `scenarios/<id>/scenario.json` and follows the same structure as `therapy.json`, with an additional `objectives` field containing the script passed to the `CaregiverAgent`:

```json
{
  "patient_id": 1,
  "patient_full_name": "Mario Rossi",
  "gender": "Male",
  "birth_date": "1945-06-10T00:00:00",
  "age": 80,
  "medical_conditions": ["Hypertension"],
  "activities": [ "..." ],
  "expired_activities": [],
  "objectives": "# Scenario 1\n\n## Objectives\n1. Add a 30-minute evening walk..."
}
```

#### Output structure

```
logs/batch_results/<batch_id>/
├── batch.log               # global batch log (errors, skips, summary)
├── results.json            # all evaluation results aggregated
└── <scenario_id>/
    ├── scenario_<id>.log   # per-scenario log
    ├── chat.log            # USER/ASSISTANT transcript
    └── full.log            # full log including tool calls
    └── evaluation.json     # result of the judge evaluation
```

`results.json` contains one entry per scenario with the `JudgeAgent` evaluation:

```json
{
  "batch_id": "20260630_143000",
  "results": [
    {
      "scenario_id": 1,
      "overall_status": "completed",
      "turns": 6,
      "patient": "Mario Rossi",
      "elapsed_seconds": 42.3,
      "objectives": [
        {
          "id": 1,
          "description": "Add a 30-minute evening walk on Tuesday and Thursday at 17:00",
          "status": "completed",
          "evidence": "Assistant confirmed: activity added for Tue, Thu at 17:00",
          "notes": null
        }
      ],
      "summary": "All objectives were completed successfully within 6 turns."
    }
  ],
  "failed": []
}
```

If a scenario raises an unhandled exception it is logged in `batch.log` under `failed`, and the batch continues with the next scenario.

---

## Data files

| Path | Description |
|---|---|
| `data/therapy.json` | Working copy of the current patient's therapy (overwritten at startup and mutated during a session) |
| `data/medicines/*.md` | Pharmacological data files indexed into ChromaDB on startup. Add new `.md` files here to extend the medicine knowledge base |
| `data/patients/<id>/history.json` | Seed safety-event history for a patient |
| `data/patients/<id>/preferences.json` | Seed preferences for a patient |
| `data/patients/<id>/conflict_resolutions.json` | Seed past conflict resolutions for a patient |
| `scenarios/<id>/scenario.json` | Self-contained test scenario: patient data, initial therapy, and caregiver objectives |
| `chromadb/` | Persistent ChromaDB store (auto-created) |
| `logs/<patient_id>/<session_id>/chat.log` | Chat-only log (USER/ASSISTANT messages); used for session resume |
| `logs/<patient_id>/<session_id>/full.log` | Full log including tool calls, errors, and system events |
| `logs/<patient_id>/<session_id>/agent_<name>.log` | Per-agent log for each active agent |
| `logs/<patient_id>/<session_id>/therapy_snapshots.json` | Runtime therapy snapshots with logical message indices; used for conversation rewind |
| `logs/batch_results/<batch_id>/results.json` | Aggregated evaluation results for a batch test run |