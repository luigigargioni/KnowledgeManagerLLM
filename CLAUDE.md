# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A conversational assistant that helps a caregiver manage a patient's therapy schedule. The caregiver speaks natural language; an LLM uses tool-calling to add/update/remove therapy activities while deterministic code enforces scheduling constraints and RAG lookups enforce safety checks.

The repo doubles as a **research harness**: most of the code exists so that scenarios can be replayed automatically and graded, not just so a human can chat.

## Commands

All commands run from `src/` (module imports are flat: `import tools`, `from agents.agent import Agent` — running from the repo root will fail).

```bash
pip install -e .              # from repo root; requires Python >= 3.14
cd src

python main.py                                    # interactive terminal chat
python main.py --input <script.md>                # agent mode: CaregiverAgent drives the conversation
python main.py -i <script.md> --delay 2           # the file is free text, passed verbatim to the agent

streamlit run chat_interface.py                   # web UI

python test.py                                    # batch: every scenario in scenarios/
python test.py --from 1 --to 1                    # a single scenario
python test.py --from 1 --to 20 --max-turns 40

ruff check .                  # lint (E, F, I; line-length 88, E501 ignored)
ruff format .                 # format (double quotes, spaces)
```

There is **no unit test suite**. `test.py` is the scenario batch runner — "running the tests" means running scenarios and reading the `JudgeAgent` verdicts. Doing so needs a live LLM backend and takes minutes per scenario.

Runtime prerequisites: an LLM backend (Ollama at `OLLAMA_URL`, or OpenAI with `OPENAI_API_KEY`) and — for `main.py` / the Streamlit UI only — PostgreSQL. `test.py` runs stateless without a database. Config lives in `.env` (copy from `.env.example`); everything is read in `config_loader.py`.

## Architecture

### Multi-agent structure

`Chat` (in `chat.py`, aliased `OllamaChat`) is the orchestrator and the only component that knows all agents exist — agents never reference each other. Every agent subclasses `agents/agent.py:Agent` and owns four things: its system prompt, its tool declarations, `inject_context()` (system messages pushed at construction), and `execute_tool()`.

- `TherapyManagerAgent` — supervisor; talks to the caregiver, performs therapy CRUD.
- `TherapyCheckAgent` (`checker_agent`) — worker; medicine/history/interaction safety checks. Constructed with `zero_shot=True`, so `_run_agent_loop` calls `reset_agent()` after each reply and it re-injects fresh context every time.
- `CaregiverAgent` — simulates the human user from a scenario script; ends by emitting `exit`.
- `JudgeAgent` — grades a finished conversation, returns JSON. Also `zero_shot`.

A worker is exposed to the supervisor via `Agent.as_tool_declaration()`, which produces a `delegate_to_<name>` function. Adding a worker means: subclass, instantiate in `Chat.__init__`, register in `self._agent_registry`, and append its tool declaration to the supervisor's tools. Nothing else changes.

`Chat._run_agent_loop()` is the single tool-calling loop used by both the supervisor and delegated workers (max 10 iterations). `Chat.execute_tool()` intercepts only delegation and `save_session`; everything else falls through to the agent's own `execute_tool()`.

### Two conversation drivers

`main.py --input` builds a **LangGraph** loop (`agent_graph.py`): `caregiver → therapy_manager → caregiver`, terminating when `is_exit_message()` matches. `test.py` drives the same two parties with a **plain Python loop** instead, because it needs to inject deferred context mid-conversation (see below). Both paths must stay behaviourally equivalent; `main.py:run_agent_mode_old` is dead code kept for reference.

### Evaluation is diff-based, not transcript-based

The judge is given a programmatic diff (`therapy_diff.diff_therapies` / `render_diff`) between the initial and final therapy, computed from `Chat._therapy_snapshots`, and its prompt states that the diff — not the transcript — is authoritative. This exists because the assistant reliably claims changes it never applied. Do not "simplify" the judge to read only the transcript.

Relatedly, `scenario_loader.split_objectives()` withholds conditional clauses (`If the assistant…`, `Verify that the assistant…`) from the caregiver until after the assistant's first reply, so the caregiver cannot leak the answer into its own opening message. `test.py` delivers the withheld part as a system message at `turn >= 1`.

### State: three stores plus a file

| Store | Role |
|---|---|
| `data/therapy.json` | The working state. Every tool in `tools.py` reads and rewrites this file; it is the single source of truth during a session. |
| ChromaDB (`chromadb/`) | 4 collections — `medicines` and `patient_history` are read-only at runtime; `conflict_resolutions` and `patient_preferences` are written at end of session by `session_extractor.py`. |
| PostgreSQL (`sql_db.py`) | Patients and versioned therapy snapshots. Optional: passing `database_manager=None` gives stateless mode. |
| `Chat._therapy_snapshots` | In-memory list of `{message_idx, therapy}`, persisted to `therapy_snapshots.json`, used for conversation rewind and for the judge's diff. |

`message_idx` counts only **visible turns** (`utils.visible_turns` / `is_visible_turn`: user turns, plus assistant turns with content and no `tool_calls`). Transcript building, snapshot indexing, the Streamlit UI and knowledge extraction all rely on this same definition — changing it in one place desynchronises indices everywhere.

### Deterministic vs. LLM responsibilities

Scheduling is deliberately **not** the LLM's job. `tools.py` compares time overlap ∧ day-of-week overlap ∧ `valid_from`/`valid_until` range overlap, validates dependency existence and temporal ordering, and assigns `activity_id` itself (`<category-prefix>_<NNN>`) — a caller-supplied id is logged and discarded. `add_therapy_activity` / `update_therapy_activity` return `status: "failure"` with two suggested alternative times plus any `past_resolution_hints` from ChromaDB; the LLM surfaces the conflict and **always** asks the caregiver rather than resolving it. Its prompt forbids reporting an outcome not read from a tool result.

### RAG thresholds

`vector_db.py` opens with a block of cosine-distance constants, each with a comment recording the measurement that produced its value (e.g. why both patient-history thresholds sit at 0.85). These are tuned against the current dataset and the `all-MiniLM-L6-v2` default embedding function — treat the comments as the changelog and update them if you retune.

`test.py` resets and re-seeds the whole vector store before every batch, then calls `verify_seed()` and aborts if incomplete: seeding is idempotent by document id, so a store left over from an older dataset would silently grade the run against the wrong knowledge base.

## Gotchas

- **Provider selection is implicit**: `LLM_PROVIDER` is derived in `config_loader.py` as `"openai" if OPENAI_API_KEY else "ollama"`. The `PROVIDER` and `GROQ_API_KEY` keys in `.env.example` are not read by any code.
- `_run_agent_loop` passes `reasoning_effort="low"` on every completion call — fine for `gpt-oss`/OpenAI reasoning models, rejected by others.
- `Chat.conversation_history` stays empty; the real history is `chat.chat_agent.conversation_history`. `end_session()` still passes the empty list to the extractors, so end-of-session extraction currently sees nothing.
- `prompts.py` holds the *old* agent prompts; the live ones are inline in each `agents/*.py`. Only `_CONFLICT_EXTRACTION_PROMPT` and `_PREFERENCE_EXTRACTION_PROMPT` are still used (by `session_extractor.py`).
- The README has drifted: scenarios are `scenarios/<id>.json` (not `scenarios/<id>/scenario.json`), and it still lists `agent.py`, `session_manager.py` and `log_parser.py`, which no longer exist (their code moved into `agents/`, `chat.py` and `utils.py`). Prefer the source.
- `is_exit_message()` intentionally matches a trailing keyword, not an exact message, because the caregiver agent appends `exit` to closing pleasantries.

## Data layout

`data/medicines/*.md` — drop a new file here to extend the pharmacological knowledge base. `data/patients/<id>/` — `history.json`, `preferences.json`, `conflict_resolutions.json` seeds. `scenarios/<id>.json` — a full therapy document plus an `objectives` field holding the caregiver script. Session output goes to `logs/<session_id>/` (`full.log`, `chat.log`, `agent_<name>.log`, `therapy_snapshots.json`); batch output to `logs/batch_results/<batch_id>/`, plus a styled Excel workbook written by `results_extractor.py`.
