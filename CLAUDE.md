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

Runtime prerequisites: an LLM backend (Ollama at `OLLAMA_URL`, OpenAI with `OPENAI_API_KEY`, or Groq with `GROQ_API_KEY`) and — for `main.py` / the Streamlit UI only — PostgreSQL. `test.py` runs stateless without a database. Config lives in `.env` (copy from `.env.example`); everything is read in `config_loader.py`.

## Architecture

### The LLM backends — two roles, independently configured

`config_loader` builds two `LLMConfig` objects, and `llm_client` is the only place a client is constructed:

| Role | Config | Client | Who uses it |
|---|---|---|---|
| system under test | `MAIN_LLM` (`PROVIDER`, `MODEL`, …) | `make_main_client()` | `Chat` (therapy manager + checker), `session_extractor` |
| simulation | `SIM_LLM` (`SIM_PROVIDER`, `SIM_MODEL`, …) | `make_sim_client()` | caregiver in `test.py` / `agent_graph.py` / `main.py`, `JudgeAgent` |

Each role has its own **provider and model**, so a locally served model can be graded by a cloud one. Every `SIM_*` setting falls back to its `MAIN` counterpart when empty; credentials and endpoints belong to the provider (`GROQ_API_KEY`, `OLLAMA_URL`, …) and are shared by both roles. All three providers speak the OpenAI protocol, so only the base URL and key differ. `PROVIDER` left empty is inferred: OpenAI key → Groq key → Ollama.

A client knows its own model, so `create(...)` may omit `model=`. Keeping the roles apart matters on Groq, where quotas are counted per model, and it keeps the simulated user off the model being measured.

The client returned by `make_client()` is a `RateLimitedClient` proxy exposing the usual `.chat.completions.create(...)`. It paces requests against a 60-second sliding window *before* sending, using limits carried by the config (`LLM_RPM`/`LLM_TPM`, `SIM_LLM_RPM`/…), reconciles its token estimate with the `usage` the provider reports — the char→token ratio is *calibrated per quota* after the first responses, because these prompts tokenise near 7.5 chars/token and a naive estimate halves throughput — and retries 429s using `Retry-After`. Quotas are keyed `provider:model`, so two roles on the same model share one budget. Two failures are made explicit instead of being retried blindly:

- `DailyQuotaExceeded` — `LLM_RPD`/`LLM_TPD` reached, or the provider asked for a wait longer than `LLM_MAX_RETRY_WAIT`. `test.py` catches it and stops the batch, keeping the scenarios already graded.
- `RequestTooLarge` — HTTP 413: the prompt alone exceeds the per-minute token budget, so no wait can help. On Groq's free tier (8K TPM) a therapy-manager request starts around 3.6K tokens and grows with every tool result, so long conversations do hit this.

Limits default to 0 (disabled) for OpenAI and Ollama, and to the Groq free-tier figures for `groq` — per role, so `PROVIDER=ollama` with `SIM_PROVIDER=groq` throttles only the simulation side.

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

### Where the tokens go

Measured on Groq (`prompt_tokens` from the API, scenario 1, `openai/gpt-oss-120b`), one therapy-manager request costs:

| Component | Tokens | Re-sent on |
|---|---|---|
| system prompt (after compaction; was 1396) | ~1130 | every call |
| 9 tool declarations | 564 | every call |
| injected context (datetime + therapy) | 281 | every call |
| each user turn + tool result | ~290 | accumulates |

One 6-turn scenario ran 17 manager calls + 5 checker calls ≈ 75K tokens on the assistant side alone, so **the system prompts and the tool schemas are over half of a scenario's budget** — they are re-sent in full on every iteration of the agent loop. On a tokens-per-minute-capped tier that is throughput, not just cost.

Three things are already done and should stay done:

- tool results are serialised compactly (`tools._tool_json`; `indent=2` cost ~35% of the payload in pure whitespace);
- the checker injects its context once per reset, not twice;
- both system prompts were compacted by removing repetition only. In the manager, several rules were stated in two or three sections at once (conflict resolution, id handling, "rely on the tools for overlap checks"): static cost per call 2413 → 1975 tokens. In the checker, a prose `# TOOLS` list repeated what `_CHECK_TOOLS` already declares — and repeated the two most important when-to-call rules a third time inside the steps: 1414 → 1279 tokens per call.

Keep both deduplicated. A tool's when-to-call instruction belongs in its schema description, which the model always sees; a behavioural rule belongs in exactly one section of the prompt. Restating either "for emphasis" costs on every iteration of the loop.

### Evaluation is diff-based, not transcript-based

The judge is given a programmatic diff (`therapy_diff.diff_therapies` / `render_diff`) between the initial and final therapy, computed from `Chat._therapy_snapshots`, and its prompt states that the diff — not the transcript — is authoritative. This exists because the assistant reliably claims changes it never applied. Do not "simplify" the judge to read only the transcript.

Relatedly, `scenario_loader.split_objectives()` withholds the scenario title, the preamble and the conditional clauses (`If the assistant…`, `Verify that the assistant…`) from the caregiver, so it cannot leak the expected answer into its own opening message — 71 of the 105 scenarios have such a clause, and it usually fixes the expected end state ("accept its suggested alternative time", "do not proceed with it"), which is what the judge grades. `test.py` delivers the withheld part as a system message on the *event*, not on a turn number: only once the assistant has itself raised the point. If it never does, the caregiver never learns it and `evaluation["branch_exercised"]` records `False`.

The gate has two independent halves, because there are two kinds of issue:

- **Deterministic.** `Chat._record_issue_signals` inspects every tool result — every agent's tool calls funnel through `Chat.execute_tool` — and records the causes the system itself detected: the `issue` field that `tools.py` attaches to blocking failures (`schedule_conflict`, `missing_dependency`, `temporal_ordering`, `dependency_blocked`), plus `vector_db.MEDICINE_NOT_FOUND_MARKER` from a medicine lookup. `chat.turn_issues` holds what fired while producing the last reply; `chat.issue_signals_seen` the whole run. Since a signal proves a problem was found but not that it was passed on, delivery also needs `utils.assistant_handed_back()` — a deliberately weak "did it ask anything at all" check. A signal with no hand-back is logged as a warning: that is the assistant swallowing a finding.
- **Textual.** `utils.assistant_raised_issue()` keyword-matches the reply, and still fires on its own. It covers what no tool can report — a contraindication the model derives itself from the medicine document and the patient's conditions, with every tool call returning success. `_NEGATED_ISSUE_RE` strips negated occurrences first, because the all-clear is worded with the same nouns as the alarm ("found no conflicts").

Only **blocking** causes count as signals: the action did not happen and cannot until the caregiver chooses. What is left out is as deliberate as what is in — validation slips carry no `issue` field, and patient-history hits are not signals at all even though they are detectable. Measured on scenarios 14/17/18/20, a history hit occurs on essentially every request (permissive threshold, and the checker queries it every time) while the write still succeeds; counting it fired the gate on turn 2 of all four, and in 17 the caregiver answered "yes, add it at the suggested alternative time" one turn before any alternative existed. Whether a history note is worth raising is the assistant's judgement, so it belongs to the textual half.

Do not collapse the two halves: signals alone miss the LLM-judgement branch, keywords alone miss unusual wording and fire on turns where nothing happened.

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

- **Provider selection can be implicit**: with `PROVIDER` empty, `config_loader.py` picks OpenAI if `OPENAI_API_KEY` is set, then Groq if `GROQ_API_KEY` is set, else Ollama. A `.env` holding several keys therefore needs `PROVIDER` set explicitly.
- `_run_agent_loop` passes `reasoning_effort` from its client's config (default `"low"`) on every completion call. The `gpt-oss` family accepts it on every provider; plain chat models answer 400. `llm_client` catches that specific 400, drops the parameter for that quota and retries, so switching provider does not require touching the knob — one wasted request per model per process, and a `model rejected 'reasoning_effort'` warning in the log. Set `REASONING_EFFORT=` / `SIM_REASONING_EFFORT=` empty to avoid even that.
- `Chat.conversation_history` stays empty; the real history is `chat.chat_agent.conversation_history`. `end_session()` still passes the empty list to the extractors, so end-of-session extraction currently sees nothing.
- `prompts.py` holds the *old* agent prompts; the live ones are inline in each `agents/*.py`. Only `_CONFLICT_EXTRACTION_PROMPT` and `_PREFERENCE_EXTRACTION_PROMPT` are still used (by `session_extractor.py`).
- The README has drifted: scenarios are `scenarios/<id>.json` (not `scenarios/<id>/scenario.json`), and it still lists `agent.py`, `session_manager.py` and `log_parser.py`, which no longer exist (their code moved into `agents/`, `chat.py` and `utils.py`). Prefer the source.
- `is_exit_message()` intentionally matches a trailing keyword, not an exact message, because the caregiver agent appends `exit` to closing pleasantries.

## Data layout

`data/medicines/*.md` — drop a new file here to extend the pharmacological knowledge base. `data/patients/<id>/` — `history.json`, `preferences.json`, `conflict_resolutions.json` seeds. `scenarios/<id>.json` — a full therapy document plus an `objectives` field holding the caregiver script. Session output goes to `logs/<session_id>/` (`full.log`, `chat.log`, `agent_<name>.log`, `therapy_snapshots.json`); batch output to `logs/batch_results/<batch_id>/`, plus a styled Excel workbook written by `results_extractor.py`.
