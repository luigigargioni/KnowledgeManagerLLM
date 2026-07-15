# test.py

import argparse
import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from time import time

from agents.caregiver_agent import CaregiverAgent
from agents.judge_agent import JudgeAgent
from chat import OllamaChat
from config_loader import MODEL, RESULTS_DIR, SCENARIOS_DIR
from scenario_loader import (
    install_scenario_therapy,
    load_scenario,
    therapy_to_natural_language,
)
from utils import build_transcript, setup_logger
from vector_db import VectorDBManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KnowledgeManagerLLM – Batch test runner"
    )
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
) -> dict:
    """
    Esegue un singolo scenario e restituisce il risultato della valutazione.
    Lancia eccezioni in caso di errore — il caller decide se continuare.
    """
    start = time()
    logger = setup_logger(
        logs_dir=batch_log_dir,
        session_folder_name=str(scenario_id),
    )
    logger.info(f"[SCENARIO {scenario_id}] Starting")

    # ── Carica e installa la terapia dello scenario ───────────────────────
    scenario = load_scenario(scenario_id)
    install_scenario_therapy(scenario)
    logger.info(
        f"[SCENARIO {scenario_id}] Therapy installed – patient: {scenario.get('patient_full_name')}"
    )

    # ── Inizializza Chat in modalità stateless (no DB) ────────────────────
    chat = OllamaChat(
        model=MODEL,
        database_manager=None,  # stateless
        vector_db=vector_db,
    )

    # ── CaregiverAgent con contesto terapia + obiettivi ───────────────────
    therapy_context = therapy_to_natural_language(scenario)
    script = scenario.get("objectives", "")
    full_script = f"#SCENARIO\n{script}\n#PATIENT CONTEXT\n{therapy_context}"

    caregiver = CaregiverAgent(script=full_script)

    # ── Loop conversazione ────────────────────────────────────────────────
    first_message = chat.chat_agent.conversation_history[-1]["content"]
    logger.info(f"[SCENARIO {scenario_id}] Conversation started")

    chatbot_response = first_message
    turns = 0

    for turn in range(max_turns):
        # Caregiver riceve risposta chatbot e genera il prossimo messaggio
        caregiver.conversation_history.append(
            {"role": "user", "content": chatbot_response}
        )
        response = chat.client.chat.completions.create(
            model=chat.model,
            messages=caregiver.conversation_history,
        )
        caregiver_message = response.choices[0].message.content or ""
        caregiver.conversation_history.append(
            {"role": "assistant", "content": caregiver_message}
        )

        logger.info(
            f"[SCENARIO {scenario_id}][TURN {turn + 1}] CAREGIVER: {caregiver_message[:120]}"
        )

        if caregiver_message.strip().lower() in ["exit", "quit", "esci"]:
            turns = turn + 1
            logger.info(f"[SCENARIO {scenario_id}] Exit after {turns} turn(s)")
            break

        if delay > 0:
            time.sleep(delay)

        chatbot_response = chat.send_message(caregiver_message)
        logger.info(
            f"[SCENARIO {scenario_id}][TURN {turn + 1}] CHATBOT: {chatbot_response[:120]}"
        )

        if delay > 0:
            time.sleep(delay)
    else:
        turns = max_turns
        logger.warning(
            f"[SCENARIO {scenario_id}] Max turns ({max_turns}) reached without exit"
        )

    # ── Valutazione ───────────────────────────────────────────────────────
    transcript = build_transcript(chat.chat_agent.conversation_history)
    judge = JudgeAgent()
    evaluation = judge.evaluate(
        client=chat.client, model=chat.model, script=script, transcript=transcript
    )

    if evaluation.get("status") == "error":
        raise RuntimeError(
            f"Judge failed for scenario {scenario_id}: {evaluation.get('message')}\n"
            f"Raw output: {evaluation.get('raw_output', '')}"
        )

    evaluation["scenario_id"] = scenario_id
    evaluation["turns"] = turns
    evaluation["patient"] = scenario.get("patient_full_name", "Unknown")
    evaluation["elapsed_seconds"] = round(time() - start, 2)

    output_path = logger.session_dir / "evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(
        f"[SCENARIO {scenario_id}] Evaluation complete – "
        f"overall: {evaluation['overall_status']}"
    )
    return evaluation


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

    # Se --to non è specificato, girano tutti
    to_id = (
        args.to_id
        if args.to_id is not None
        else len([x for x in os.listdir(SCENARIOS_DIR) if x.endswith(".json")])
    )
    scenario_ids = list(range(args.from_id, to_id + 1))

    # Cartella di output per questa batch run
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

    # Logger globale del batch
    logger.info(
        f"[BATCH] Starting – scenarios {args.from_id}→{to_id} "
        f"({len(scenario_ids)} total) | batch_id={batch_id}"
    )

    # Vector DB condiviso tra tutti gli scenari
    vector_db = VectorDBManager()
    vdb_available = vector_db.initialize()
    if vdb_available:
        vector_db.seed_medicines()
    else:
        logger.warning("[BATCH] Vector DB not available – RAG features disabled")
        vector_db = None

    print("=" * 60)
    print(f"  Batch test run — {len(scenario_ids)} scenario(s)")
    print(f"  Output: {batch_log_dir}")
    print("=" * 60)

    results = []
    failed = []

    for scenario_id in scenario_ids:
        print(f"\n[{scenario_id}/{to_id}] Running scenario {scenario_id}...")

        try:
            evaluation = run_scenario(
                scenario_id=scenario_id,
                vector_db=vector_db,
                delay=args.delay,
                max_turns=args.max_turns,
                batch_log_dir=batch_log_dir,
            )

            to_append = {
                "overall_status": evaluation["overall_status"],
                "elapsed_seconds": evaluation["elapsed_seconds"],
                "scenario_id": scenario_id,
                "turns": evaluation["turns"],
                "patient": evaluation["patient"],
            }
            results.append(to_append)
            print_scenario_summary(scenario_id, evaluation)

        except FileNotFoundError as e:
            logger.warning(f"[BATCH] Scenario {scenario_id} skipped: {e}")
            print(f"Scenario {scenario_id:>3} | skipped (not found)")
            failed.append({"scenario_id": scenario_id, "error": str(e)})

        except Exception as e:
            logger.error(
                f"[BATCH] Scenario {scenario_id} failed: {e}\n{traceback.format_exc()}"
            )
            print(f"Scenario {scenario_id:>3} | ERROR: {str(e)[:80]}")
            failed.append({"scenario_id": scenario_id, "error": str(e)})

    # ── Salva risultati ───────────────────────────────────────────────────
    results_path = batch_log_dir / "results.json"
    results_path.write_text(
        json.dumps(
            {"batch_id": batch_id, "results": results, "failed": failed},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ── Sommario finale ───────────────────────────────────────────────────
    total = len(results)
    n_completed = sum(1 for r in results if r.get("overall_status") == "completed")
    n_partial = sum(1 for r in results if r.get("overall_status") == "partial")
    n_failed = sum(1 for r in results if r.get("overall_status") == "failed")
    n_errors = len(failed)

    obj_total = sum(len(r.get("objectives", [])) for r in results)
    obj_completed = sum(
        1
        for r in results
        for o in r.get("objectives", [])
        if o["status"] == "completed"
    )

    print("\n" + "=" * 60)
    print("  Batch summary")
    print("=" * 60)
    print(f"  Scenarios run:     {total + n_errors}")
    print(f"Completed:      {n_completed}")
    print(f"Partial:        {n_partial}")
    print(f"Failed:         {n_failed}")
    print(f"Errors:         {n_errors}")
    if obj_total > 0:
        print(
            f"  Objective rate:    {obj_completed}/{obj_total} ({obj_completed / obj_total * 100:.1f}%)"
        )
    print(f"\n  Results saved to: {results_path}")

    logger.info(
        f"[BATCH] Done – {total} evaluated, {n_errors} errors | "
        f"objectives: {obj_completed}/{obj_total}"
    )


if __name__ == "__main__":
    main()
