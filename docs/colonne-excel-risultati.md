# Columns of the results Excel sheet

File: `logs/batch_results/all_results.xlsx`, sheet **Results** — one row per scenario.
The columns are defined in `src/results_extractor.py` (`_OBJECTIVE_COLUMNS`) and populated in
`src/test.py` (`run_scenario`) from the `JudgeAgent` verdict enriched by the harness.

The file is **cumulative**: every batch appends rows without touching the previous ones. A
column introduced by a newer version of the code is appended at the end of the existing
sheet (`_sync_headers`), so rows are always written by column name and never by position —
which is why columns added by hand at the end (e.g. the review ones) survive later batches.
Cells are truncated at 32767 characters (the xlsx format limit) with a `[… truncated]` note.

---

## Run identification

| Column | Meaning |
|---|---|
| `test_date` | Date/time the row was written |
| `batch_id` | Batch identifier; used to locate the log directory at `logs/batch_results/<batch_id>/` |
| `scenario_id` | Scenario number in `scenarios/` (1 to 100) |
| `patient` | `First Last(patient_id)` of the scenario's patient |

## Outcome and cost

| Column | Meaning |
|---|---|
| `overall_status` | Overall status according to the `JudgeAgent`: `completed` / `partial` / `failed` / `not_attempted`, or `error` if the scenario crashed and there is no verdict. In the cases flagged by `branch_clamped` the harness overrides the judge and caps it at `partial`. Colours the whole row |
| `turns` | Number of conversation turns consumed |
| `elapsed_seconds` | Scenario duration in seconds |

## Deterministic block — what the code establishes, not what a model judges

This is the part a reviewer should be able to read without opening the logs: all of it is
computed in code during the run.

| Column | Meaning |
|---|---|
| `changed_activities` | A single line naming every activity the conversation touched: prefix `+` (added), `-` (removed), `~` (modified, followed by the names of the changed fields only). This is the column to scan to catch a change nobody asked for; what actually changed is in `applied_changes` |
| `issue_signals` | The blocking causes the system itself raised: `schedule_conflict`, `missing_dependency`, `temporal_ordering`, `dependency_blocked`, the medicine-not-found marker, plus the safety gate's `safety_blocked` / `safety_caution` / `safety_check_required` refusals. `none` if nothing blocked |
| `branch_outcome` | Whether the scenario's conditional part ("if the assistant detects X…") was delivered to the caregiver agent, which only happens once the chatbot raises the point on its own: `exercised` (delivered), `not_raised_no_change` (never raised, nothing changed), `not_raised_but_change_applied` (never raised but the therapy was changed anyway — often the model sidestepped the problem upstream rather than missing it), `n/a` if the scenario has no conditional clause |
| `branch_clamped` | `no`, or `objectives [n] failed→partial`: the judge failed an objective whose conditional clause was never delivered to the caregiver, which therefore could not follow it, and the harness capped the grade at `partial`. It marks a limitation of the harness, not a defect of the system under test (`test.clamp_undelivered_branch`) |
| `safety_verdicts` | Every verdict of the agent that checks whether an action is safe, with turn, severity (`blocking` / `caution` / `remark`, plus `(untyped)` when the severity is missing) and activity name. This is where to read **why** a write was refused — or why one that should have been refused was not. A trailing `[!]` line reports unparsable verdicts (the gate failed open) or writes attempted before any check |
| `unsupported_claims` | Replies announcing a change with no write tool behind it, with the turn number. `none` if there were none |
| `history_warnings_retrieved` | Patient-history events retrieved by RAG, warning level only (`info` events are excluded): they feed the **safety** checks and should be passed on to the caregiver. Scheduling conflicts do not come from here — they are computed deterministically by `tools.py`. Read alongside the transcript to see whether a surfaced risk was actually relayed |

## Objectives

| Column | Meaning |
|---|---|
| `objectives_scripted` | How many objectives (steps) the scenario's script asked for |
| `objectives_status` | Compact string with the initial of each graded objective's status, e.g. `C,P,F`: **C** completed, **P** partial, **F** failed, **N** not_attempted (never raised by the caregiver). Compared against `objectives_scripted` it separates "failed" from "never asked" |
| `objectives` | The full scenario text. The caregiver only receives part of it: the title, the preamble and the conditional clauses are withheld so it cannot leak the expected answer (`scenario_loader.split_objectives`) |
| `judge_check` | The judge's output **per objective**: task description, status, supporting evidence and notes. The judge's overall `summary` is not included. If `overall_status = error` it holds the error message and the first 500 characters of the judge's unparsed raw output instead |

## Raw material for inspection

| Column | Meaning |
|---|---|
| `applied_changes` | The full list of changes applied during the conversation: the programmatic initial→final diff, i.e. exactly what the judge was shown (grading is diff-based, not transcript-based) |
| `conversation` | The full transcript between the caregiver agent and the system |
| `initial_therapy` | JSON of the starting therapy |
| `final_therapy` | JSON of the therapy at the end of the conversation (empty on `error`) |

## Manually added columns

`reviewer`, `reviewer_status` and `notes` do not exist in the code: they were appended to the
sheet by hand for the manual analysis phase — respectively the initials of whoever reviews
that row, the scenario's actual status after review, and any notes for the team.
