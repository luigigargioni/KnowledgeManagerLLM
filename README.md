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
| `prompts.py` | System prompt for the main assistant and extraction prompts for `session_extractor` |
| `config_loader.py` | Loads all settings from the `.env` file |
| `main.py` | Terminal entry point – supports both interactive and file-driven modes |
| `chat_interface.py` | Streamlit web UI entry point |

### Agent architecture

The system uses a **supervisor/worker** multi-agent pattern:

- **`TherapyManagerAgent`** (supervisor) – manages the conversation with the caregiver, handles therapy CRUD operations, and delegates to worker agents when needed.
- **`TherapyCheckAgent`** (worker) – invoked automatically before any activity is added or updated; checks compatibility with the patient's current medications, history, and preferences.

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
# LLM backend – use one of the two options below

# Option A: local Ollama
MODEL=gpt-oss:20b
OLLAMA_URL=http://localhost:11434

# Option B: OpenAI cloud (set a valid key; the Ollama settings are ignored)
# OPENAI_API_KEY=sk-...
# MODEL=gpt-5.4-mini

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
| `--input <file>` | `-i` | *(none)* | Path to a text file containing one user message per line. If omitted, runs in interactive mode. |
| `--delay <seconds>` | `-d` | `0` | Pause between messages when running in file mode. Useful for readability. |

**File mode** – run a pre-written conversation script non-interactively:

```bash
python main.py --input tests/test_add_activity.txt
python main.py --input tests/test_add_activity.txt --delay 1
```

The input file format is plain text, one message per line. Empty lines and lines starting with `#` are ignored:

```text
# test_add_activity.txt
What can you do?

Add a morning walk, 30 minutes, every Monday Wednesday Friday at 8:00
Yes, confirm the activity
exit
```

This mode is designed for automated testing: the assistant processes each line as a user message and prints the responses to stdout. The session is saved to PostgreSQL at the end exactly as in interactive mode.

### Streamlit web interface

```bash
cd src
streamlit run chat_interface.py
```

---

## Data files

| Path | Description |
|---|---|
| `data/therapy.json` | Working copy of the current patient's therapy (overwritten at startup and mutated during a session) |
| `data/medicines/*.md` | Pharmacological data files indexed into ChromaDB on startup. Add new `.md` files here to extend the medicine knowledge base |
| `data/patients/<id>/history.json` | Seed safety-event history for a patient |
| `data/patients/<id>/preferences.json` | Seed preferences for a patient |
| `data/patients/<id>/conflict_resolutions.json` | Seed past conflict resolutions for a patient |
| `chromadb/` | Persistent ChromaDB store (auto-created) |
| `logs/<patient_id>/<session_id>/chat.log` | Chat-only log (USER/ASSISTANT messages); used for session resume |
| `logs/<patient_id>/<session_id>/full.log` | Full log including tool calls, errors, and system events |
| `logs/<patient_id>/<session_id>/agent_<name>.log` | Per-agent log for each active agent |
| `logs/<patient_id>/<session_id>/therapy_snapshots.json` | Runtime therapy snapshots with logical message indices; used for conversation rewind |