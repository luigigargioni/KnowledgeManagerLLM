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
| `tools.py` | All tool implementations: CRUD on therapy activities, scheduling conflict detection, RAG lookups |
| `vector_db.py` | `VectorDBManager` – wraps ChromaDB; manages 4 collections (see below) |
| `sql_db.py` | `DatabaseManager` – wraps PostgreSQL via SQLAlchemy; stores patients and versioned therapy snapshots |
| `session_extractor.py` | End-of-session LLM extraction: saves conflict resolutions and patient preferences to ChromaDB |
| `prompts.py` | System prompt for the main assistant and extraction prompts for `session_extractor` |
| `config_loader.py` | Loads all settings from the `.env` file |
| `main.py` | Terminal entry point |
| `chat_interface.py` | Streamlit web UI entry point |

### ChromaDB collections

| Collection | R/W | Contents |
|---|---|---|
| `medicines` | Read-only at runtime | Pharmacological data indexed from `.md` files in `data/medicines/` |
| `patient_history` | Read-only at runtime | Historical safety events per patient (seeded from `data/patients/<id>/history.json`) |
| `conflict_resolutions` | Read-Write | Past scheduling conflict resolutions, extracted automatically at session end |
| `patient_preferences` | Read-Write | Patient habits and preferences, extracted automatically at session end |

### Session lifecycle

1. **Startup** – load patient from PostgreSQL → write `data/therapy.json` → seed ChromaDB collections → initialise chat with context (datetime, current activities, patient preferences).
2. **Conversation loop** – for each caregiver message the LLM may call up to 5 tools in sequence:
   - `get_medicine_data` – RAG lookup against the medicines collection (mandatory before any medicine activity).
   - `get_patient_preferences` – retrieve known patient habits for personalised suggestions.
   - `add_therapy_activity` / `update_therapy_activity` / `remove_therapy_activity` – mutate `therapy.json`; each write automatically triggers a scheduling conflict check and a patient-history safety check.
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
MODEL=qwen2.5:14b
OLLAMA_URL=http://localhost:11434

# Option B: OpenAI cloud (set a valid key; the Ollama settings are ignored)
# OPENAI_API_KEY=sk-...
# MODEL=gpt-4o

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
| `logs/` | Application log files |