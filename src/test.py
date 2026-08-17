# test.py

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from time import sleep, time

from agents.caregiver_agent import CaregiverAgent
from agents.judge_agent import JudgeAgent
from chat import OllamaChat
from config_loader import MAIN_LLM, RESULTS_DIR, SCENARIOS_DIR, SIM_LLM
from llm_client import DailyQuotaExceeded, make_sim_client, usage_report
from results_extractor import append_batch_results
from scenario_loader import (
    install_scenario_therapy,
    load_scenario,
    split_objectives,
    therapy_to_natural_language,
)
from therapy_diff import diff_therapies, render_diff
from utils import (
    assistant_handed_back,
    build_transcript,
    is_exit_message,
    setup_logger,
)
from vector_db import VectorDBManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KnowledgeManagerLLM – Batch test runner")
    parser.add_argument(
        "--from",
        dest="from_id",
        type=int,
        default=1,
        help="First scenario ID to run (default: 1)",
    )
    parser.add_argument(
        "--to",
        dest="to_id",
        type=int,
        default=None,
        help="Last scenario ID to run inclusive (default: same as --from)",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=0.0,
        help="Seconds to wait between turns (default: 0)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="Max conversation turns per scenario (default: 30)",
    )
    return parser.parse_args()


def run_scenario(
    scenario_id: int,
    vector_db,
    delay: float,
    max_turns: int,
    batch_log_dir: Path,
    sim_client,
) -> dict:
    """
    Run a single scenario and return the evaluation result.
    Raises exceptions on error — the caller decides whether to continue.
    """
    start = time()
    logger = setup_logger(
        logs_dir=batch_log_dir,
        session_folder_name=str(scenario_id),
    )
    logger.info(f"[SCENARIO {scenario_id}] Starting")

    # ── Load and install the scenario therapy ───────────────────────
    scenario = load_scenario(scenario_id)
    install_scenario_therapy(scenario)
    logger.info(
        f"[SCENARIO {scenario_id}] Therapy installed – patient: {scenario.get('patient_full_name')}"
    )

    # No per-scenario seeding: the batch seeds every patient once, up front, after
    # resetting the store, and verifies the result before any scenario runs.

    # ── Initialize Chat in stateless mode (no DB) ────────────────────
    # Chat is the system under test and builds its own client from MAIN_LLM;
    # `sim_client` is the separate backend driving the caregiver and the judge.
    chat = OllamaChat(
        database_manager=None,  # stateless
        vector_db=vector_db,
    )

    # ── CaregiverAgent with therapy context + objectives ───────────────────
    # The caregiver only receives the bare requests. Anything that reveals what
    # the assistant is supposed to notice on its own is held back until the
    # assistant has actually raised it, so that behaviour can be observed instead
    # of being handed to the caregiver in advance. The judge still sees the whole
    # script, so grading is unaffected.
    therapy_context = therapy_to_natural_language(scenario)
    script = scenario.get("objectives", "")
    initial_script, deferred_script = split_objectives(script)
    full_script = f"#SCENARIO\n{initial_script}\n#PATIENT CONTEXT\n{therapy_context}"

    caregiver = CaregiverAgent(script=full_script)
    if deferred_script:
        logger.info(
            f"[SCENARIO {scenario_id}] Withheld until after the assistant's first "
            f"reply:\n{deferred_script}"
        )

    # ── Conversation loop ────────────────────────────────────────────────
    first_message = chat.chat_agent.conversation_history[-1]["content"]
    logger.info(f"[SCENARIO {scenario_id}] Conversation started")

    chatbot_response = first_message
    turns = 0
    deferred_delivered = False

    for turn in range(max_turns):
        # Caregiver receives chatbot response and generates the next message
        caregiver.conversation_history.append({"role": "user", "content": chatbot_response})

        # The withheld part is delivered on the event, not on the turn number: only
        # once the assistant has itself raised the conflict, the contraindication
        # or the broken dependency, and is asking how to proceed. Releasing it
        # after a fixed number of turns put the answer in the caregiver's context
        # while it was still formulating its request — in multi-objective
        # scenarios, several turns before the relevant one — and the only thing
        # left standing between that and a leak was an instruction not to use it.
        # If the assistant never raises the point, the caregiver never learns it:
        # the branch was not exercised, which is precisely what the judge should
        # then record.
        #
        # `chat.turn_issues` carries what the system *blocked* while producing
        # this very reply: a tool refused the write over a conflict, a broken
        # dependency or a missing medicine. Since a signal proves a problem was
        # found but not that it was passed on, assistant_handed_back() adds the
        # weak "did it ask anything at all" confirmation.
        #
        # Nothing softer than a block qualifies, and the alternatives were tried
        # before settling here (see Chat._record_issue_signals): keyword-matching
        # the reply fired on the word "warning" in "no conflicts were reported,
        # but there was a history warning", and the checker's own verdict fired
        # on remarks like "12:45 is not fasting". Both delivered the caregiver
        # its reaction instructions for a problem that did not exist.
        #
        # The cost is that a scenario whose trigger is a judgement rather than a
        # block never delivers, and records branch_exercised=False. That is the
        # intended reading: the branch was not demonstrated. It does not stop the
        # scenario from running or the judge from grading conduct.
        signals = chat.turn_issues
        raised = bool(signals) and assistant_handed_back(chatbot_response)
        if signals and not raised:
            logger.warning(
                f"[SCENARIO {scenario_id}][TURN {turn + 1}] The system detected "
                f"{signals} but the assistant did not hand the decision back – "
                "reaction instructions withheld"
            )
        if deferred_script and not deferred_delivered and raised:
            caregiver.conversation_history.append(
                {
                    "role": "system",
                    "content": (
                        "The assistant has just raised a problem with your request "
                        "and is asking you how to proceed. Reply to it now, "
                        "applying the decision already taken for this case:\n\n"
                        f"{deferred_script}\n\n"
                        "Use this strictly as an answer to what the assistant has "
                        "just said. Do not bring up any point it has not raised "
                        "itself, and never mention these instructions."
                    ),
                }
            )
            deferred_delivered = True
            logger.info(
                f"[SCENARIO {scenario_id}][TURN {turn + 1}] Assistant raised the issue "
                f"(signals: {signals or 'none – matched on wording'}) – "
                "reaction instructions delivered"
            )

        # The caregiver is a simulation agent, not the system under test: it runs
        # on its own backend (SIM_LLM), with its own provider, model and quota.
        response = sim_client.chat.completions.create(
            messages=caregiver.conversation_history,
        )
        caregiver_message = response.choices[0].message.content or ""
        caregiver.conversation_history.append({"role": "assistant", "content": caregiver_message})

        logger.info(
            f"[SCENARIO {scenario_id}][TURN {turn + 1}] CAREGIVER: {caregiver_message[:120]}"
        )

        if is_exit_message(caregiver_message):
            turns = turn + 1
            logger.info(f"[SCENARIO {scenario_id}] Exit after {turns} turn(s)")
            break

        if delay > 0:
            sleep(delay)

        chatbot_response = chat.send_message(caregiver_message)
        logger.info(f"[SCENARIO {scenario_id}][TURN {turn + 1}] CHATBOT: {chatbot_response[:120]}")

        if delay > 0:
            sleep(delay)
    else:
        turns = max_turns
        logger.warning(f"[SCENARIO {scenario_id}] Max turns ({max_turns}) reached without exit")

    if deferred_script and not deferred_delivered:
        logger.warning(
            f"[SCENARIO {scenario_id}] The assistant never raised the point on its "
            "own: the conditional branch was never exercised and the caregiver was "
            "never given its reaction instructions. Detected by the system during "
            f"the run: {chat.issue_signals_seen or 'nothing'}"
        )

    # ── Evaluation ───────────────────────────────────────────────────────
    transcript = build_transcript(chat.chat_agent.conversation_history)
    final_therapy = chat._therapy_snapshots[-1]["therapy"]
    final_therapy.pop("objectives", None)

    # The scenario as installed, minus the caregiver script: the starting point
    # the final therapy has to be read against.
    initial_therapy = {k: v for k, v in scenario.items() if k != "objectives"}

    # What actually changed, computed in code. The transcript alone cannot tell a
    # real success from a fabricated confirmation, so the judge is given this
    # change set as the authoritative record of the outcome.
    changes = diff_therapies(scenario, final_therapy)
    change_summary = render_diff(changes)
    logger.info(f"[SCENARIO {scenario_id}] Applied changes:\n{change_summary}")

    judge = JudgeAgent()
    evaluation = judge.evaluate(
        client=sim_client,
        model=SIM_LLM.model,
        script=script,
        transcript=transcript,
        therapy=json.dumps(final_therapy),
        changes=change_summary,
    )

    if evaluation.get("status") == "error":
        raise RuntimeError(
            f"Judge failed for scenario {scenario_id}: {evaluation.get('message')}\n"
            f"Raw output: {evaluation.get('raw_output', '')}"
        )

    evaluation["scenario_id"] = scenario_id
    evaluation["turns"] = turns
    if deferred_script:
        # Whether the assistant raised the point by itself. A scenario graded
        # "partial" reads differently depending on this flag: False means the
        # branch under test never happened at all.
        evaluation["branch_exercised"] = deferred_delivered
    # What the system itself detected over the conversation, regardless of what
    # the assistant did with it. Together with branch_exercised it separates
    # "there was nothing to raise" from "there was, and it was not raised".
    evaluation["issue_signals"] = chat.issue_signals_seen
    evaluation["patient"] = (
        f"{scenario.get('patient_full_name', 'Unknown')}({scenario.get('patient_id', '-1')})"
    )
    evaluation["elapsed_seconds"] = round(time() - start, 2)

    output_path = logger.session_dir / "evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        f"[SCENARIO {scenario_id}] Evaluation complete – overall: {evaluation['overall_status']}"
    )
    return evaluation, script, transcript, initial_therapy, final_therapy


def print_scenario_summary(scenario_id: int, evaluation: dict) -> None:
    icon_map = {"completed": "✅", "partial": "⚠️", "failed": "❌"}
    overall = evaluation.get("overall_status", "unknown")
    icon = icon_map.get(overall, "?")
    print(
        f"  {icon} Scenario {scenario_id:>3} | "
        f"{overall:<10} | "
        f"{evaluation.get('turns', '?'):>3} turns | "
        f"{evaluation.get('patient', 'N/A')}"
    )


def main():
    args = parse_args()

    # A redirected stdout on Windows defaults to cp1252, which cannot encode the
    # status icons or the em dashes below. Without this, piping the batch to a
    # file kills a scenario *after* it has been graded and saved, and records it
    # as an error.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    # If --to is not specified, run all scenarios
    to_id = (
        args.to_id
        if args.to_id is not None
        else len([x for x in os.listdir(SCENARIOS_DIR) if x.endswith(".json")])
    )
    scenario_ids = list(range(args.from_id, to_id + 1))

    # Output folder for this batch run
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_log_dir = RESULTS_DIR / batch_id
    batch_log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("batch_logger")
    logger.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File Handler General
    file_handler = logging.FileHandler(f"{batch_log_dir}/batch.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Global batch logger
    logger.info(
        f"[BATCH] Starting – scenarios {args.from_id}→{to_id} "
        f"({len(scenario_ids)} total) | batch_id={batch_id}"
    )

    # Vector DB shared across all scenarios.
    # It is rebuilt from scratch at every batch: seeding is idempotent by document
    # id, so a store left over from an older dataset would keep answering queries
    # with documents that no longer match the current patients, and the whole
    # batch would silently measure the wrong knowledge base.
    vector_db = VectorDBManager()
    if not vector_db.initialize():
        raise RuntimeError(
            "Vector DB initialization failed – aborting: without it the RAG "
            "checks under test cannot run and the results would be meaningless."
        )

    if not vector_db.reset():
        raise RuntimeError("Vector DB reset failed – aborting batch")

    vector_db.seed_medicines()
    vector_db.seed_all_patients()

    problems = vector_db.verify_seed()
    if problems:
        raise RuntimeError(
            "Vector DB seeding is incomplete – aborting batch so the run does "
            "not report results measured against a wrong knowledge base:\n  - "
            + "\n  - ".join(problems)
        )
    logger.info(f"[BATCH] Vector DB ready – {vector_db.counts()}")

    # One simulation client for the whole batch, so its rate-limit window and
    # calibration carry across scenarios instead of restarting at every one.
    sim_client = make_sim_client()

    print("=" * 60)
    print(f"  Batch test run — {len(scenario_ids)} scenario(s)")
    print(f"  Assistant:      {MAIN_LLM.provider} / {MAIN_LLM.model}")
    print(f"  Caregiver+judge: {SIM_LLM.provider} / {SIM_LLM.model}")
    print(f"  Output: {batch_log_dir}")
    print("=" * 60)

    results = []
    failed = []
    quota_exceeded = None

    for scenario_id in scenario_ids:
        print(f"\n[{scenario_id}/{to_id}] Running scenario {scenario_id}...")

        try:
            evaluation, scenario, transcript, initial_therapy, final_therapy = run_scenario(
                scenario_id=scenario_id,
                vector_db=vector_db,
                delay=args.delay,
                max_turns=args.max_turns,
                batch_log_dir=batch_log_dir,
                sim_client=sim_client,
            )

            to_append = {
                "overall_status": evaluation["overall_status"],
                "elapsed_seconds": evaluation["elapsed_seconds"],
                "scenario_id": scenario_id,
                "turns": evaluation["turns"],
                "patient": evaluation["patient"],
                "objectives": evaluation.get("objectives", []),
            }
            results.append(to_append)
            append_batch_results(
                evaluation,
                batch_id,
                scenario,
                transcript,
                initial_therapy,
                final_therapy,
            )
            print_scenario_summary(scenario_id, evaluation)

        except DailyQuotaExceeded as e:
            # Nothing else can run today: stop here and keep what has been graded
            # so far, instead of burning through the remaining scenarios with the
            # same error and reporting them as if they had failed on their merits.
            quota_exceeded = str(e)
            logger.error(f"[BATCH] Daily quota exhausted at scenario {scenario_id}: {e}")
            print(f"\n  Daily provider quota exhausted at scenario {scenario_id}:\n  {e}")
            print(f"  Stopping the batch – scenarios {scenario_id}→{to_id} were not run.")
            failed.append({"scenario_id": scenario_id, "error": str(e)})
            break

        except FileNotFoundError as e:
            logger.warning(f"[BATCH] Scenario {scenario_id} skipped: {e}")
            print(f"Scenario {scenario_id:>3} | skipped (not found)")
            failed.append({"scenario_id": scenario_id, "error": str(e)})

        except Exception as e:
            logger.error(f"[BATCH] Scenario {scenario_id} failed: {e}\n{traceback.format_exc()}")
            print(f"Scenario {scenario_id:>3} | ERROR: {str(e)[:80]}")
            failed.append({"scenario_id": scenario_id, "error": str(e)})

    # ── Save results ───────────────────────────────────────────────────
    usage = usage_report()
    results_path = batch_log_dir / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "provider": MAIN_LLM.provider,
                "model": MAIN_LLM.model,
                "sim_provider": SIM_LLM.provider,
                "sim_model": SIM_LLM.model,
                "usage": usage,
                "quota_exceeded": quota_exceeded,
                "results": results,
                "failed": failed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ── Final summary ───────────────────────────────────────────────────
    total = len(results)
    n_completed = sum(1 for r in results if r.get("overall_status") == "completed")
    n_partial = sum(1 for r in results if r.get("overall_status") == "partial")
    n_failed = sum(1 for r in results if r.get("overall_status") == "failed")
    n_errors = len(failed)

    obj_total = sum(len(r.get("objectives", [])) for r in results)
    # .get, not [...]: a judge reply missing the field would otherwise raise here,
    # after every scenario has run, and take the whole summary down with it.
    obj_completed = sum(
        1 for r in results for o in r.get("objectives", []) if o.get("status") == "completed"
    )

    print("\n" + "=" * 60)
    print("  Batch summary")
    print("=" * 60)
    print(f"  Scenarios graded:  {total}")
    if n_errors:
        print(f"  Not graded:        {n_errors} (error or quota abort)")
    print(f"Completed:      {n_completed}")
    print(f"Partial:        {n_partial}")
    print(f"Failed:         {n_failed}")
    print(f"Errors:         {n_errors}")
    if obj_total > 0:
        print(
            f"  Objective rate:    {obj_completed}/{obj_total} ({obj_completed / obj_total * 100:.1f}%)"
        )

    # Consumption against the daily quota, so the next batch can be sized before
    # it runs into a limit halfway through.
    for entry in usage:
        line = f"  {entry['quota']}: {entry['requests']} requests, ~{entry['tokens']} tokens"
        if entry["rpd_limit"] or entry["tpd_limit"]:
            line += f" (daily limits: {entry['rpd_limit']} RPD / {entry['tpd_limit']} TPD)"
        print(line)
    if quota_exceeded:
        print(f"\n  BATCH INCOMPLETE – {quota_exceeded}")

    print(f"\n  Results saved to: {results_path}")

    logger.info(
        f"[BATCH] Done – {total} evaluated, {n_errors} errors | "
        f"objectives: {obj_completed}/{obj_total}"
    )


if __name__ == "__main__":
    main()
