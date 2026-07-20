# main.py
import argparse
import json
import sys
from pathlib import Path
from time import time

from langchain_core.messages import AIMessage

import tools as tools
from agent_graph import build_therapy_graph
from agents.caregiver_agent import CaregiverAgent
from agents.judge_agent import JudgeAgent
from chat import OllamaChat
from config_loader import DEFAULT_PATIENT_ID, MODEL
from sql_db import DatabaseManager
from utils import build_transcript, get_system_info, setup_logger
from vector_db import VectorDBManager

logger = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KnowledgeManagerLLM - LLM Chat Interface"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help=(
            "Path to a script file (.md, .txt, or any text format) describing "
            "the caregiver's objectives. If omitted, runs in interactive mode."
        ),
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=0.0,
        help="Seconds to wait between turns in agent mode (default: 0).",
    )
    return parser.parse_args()


def read_script(path: Path) -> str:
    """
    Legge lo script da passare al CaregiverAgent.
    Il contenuto viene passato all'agente senza alcuna modifica —
    la formattazione e la struttura sono responsabilità di chi scrive lo script.
    """
    if not path.exists():
        raise FileNotFoundError(f"Script file not found: {path}")

    script = path.read_text(encoding="utf-8").strip()

    if not script:
        raise ValueError(f"Script file is empty: {path}")

    return script


def run_agent_mode(chat, script: str, delay: float) -> None:

    caregiver = CaregiverAgent(script=script)
    graph = build_therapy_graph(chat, caregiver)

    first_message = chat.chat_agent.conversation_history[-1]["content"]
    print(f"Assistant: {first_message}\n")

    for event in graph.stream(
        {
            "messages": [AIMessage(content=first_message)],
            "session_ended": False,
        },
        {"recursion_limit": 30},
    ):
        for node_name, state in event.items():
            last = state["messages"][-1].content
            if node_name == "caregiver":
                print(f"You (agent): {last}\n")
            elif node_name == "therapy_manager":
                print(f"Assistant: {last}\n")

        if delay > 0:
            time.sleep(delay)

    # ── Evaluation ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Evaluation")
    print("=" * 60)

    transcript = build_transcript(chat.chat_agent.conversation_history)
    judge = JudgeAgent()
    evaluation = judge.evaluate(
        client=chat.client,
        model=chat.model,
        script=script,
        transcript=transcript,
        therapy=json.dumps(chat._therapy_snapshots[-1]),
    )

    evaluation["script"] = script

    if evaluation.get("status") == "error":
        print(f"[Judge] Evaluation failed: {evaluation.get('message')}")
        return None

    print(f"\nOverall: {evaluation['overall_status'].upper()}")

    # Saving of the evalutaion json
    output_path = logger.session_dir / "evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[Judge] Evaluation saved to {output_path}")

    return evaluation


def run_agent_mode_old(chat: OllamaChat, script: str, delay: float) -> None:
    """
    Modalità agente: un CaregiverAgent genera i messaggi autonomamente
    seguendo lo script, finché non produce 'exit'.
    """
    import time as time_mod

    caregiver = CaregiverAgent(script=script)
    client = chat.client  # stesso client LLM usato da Chat

    print(f"[Agent mode] Script loaded ({len(script)} chars)\n")

    # Il primo messaggio del caregiver viene generato a partire dal
    # contesto iniziale (terapia attuale) senza input esterno
    chatbot_response = chat.chat_agent.conversation_history[-1]["content"]
    print(f"Assistant: {chatbot_response}\n")

    max_turns = 30  # safety cap per evitare loop infiniti

    for turn in range(max_turns):
        # Il caregiver riceve la risposta del chatbot come messaggio in arrivo
        caregiver.conversation_history.append(
            {
                "role": "user",
                "content": chatbot_response,
            }
        )

        # Il caregiver genera la sua prossima mossa
        response = client.chat.completions.create(
            model=chat.model,
            messages=caregiver.conversation_history,
            tools=caregiver.tools or None,
        )
        caregiver_message = response.choices[0].message.content or ""
        caregiver.conversation_history.append(
            {
                "role": "assistant",
                "content": caregiver_message,
            }
        )

        print(f"You (agent): {caregiver_message}\n")

        # Condizione di uscita
        if caregiver_message.strip().lower() in ["exit", "quit", "esci"]:
            logger.info(f"[AGENT] Caregiver agent sent exit after {turn + 1} turn(s)")
            break

        if delay > 0:
            time_mod.sleep(delay)

        # Il chatbot risponde al messaggio del caregiver
        start = time()
        chatbot_response = chat.send_message(caregiver_message)
        elapsed = time() - start

        print(f"Assistant: {chatbot_response}")
        print(f"[{elapsed:.2f}s]\n")

        if delay > 0:
            time_mod.sleep(delay)

    else:
        logger.warning("[AGENT] Max turns reached without exit signal")
        print("[Max turns reached – ending session]")


def run_interactive_mode(chat: OllamaChat) -> None:
    """Modalità interattiva: l'utente digita i messaggi da tastiera."""
    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit", "esci"]:
                break

            if not user_input:
                continue

            start = time()
            response = chat.send_message(user_input)
            elapsed = time() - start

            if response:
                print(f"\nAssistant: {response}")
                print(f"[{elapsed:.2f}s]\n")

        except EOFError:
            break


def main():
    args = parse_args()

    cpu_info, ram_info, gpu_info = get_system_info()
    logger.info(
        f"[SYS] CPU:{cpu_info['model']} Cores:{cpu_info['cores']}/{cpu_info['threads']} "
        f"RAM:{ram_info:.0f} GB"
    )
    for info in gpu_info:
        logger.info(f"[SYS] GPU {info['gpu']}: {info['name']} ({info['memory']})")

    # ── Vector DB ─────────────────────────────────────────────────────────
    vector_db = VectorDBManager()
    vdb_available = vector_db.initialize()
    if vdb_available:
        seeded = vector_db.seed_medicines()
        logger.info(
            f"[CONFIG] Vector DB ready – {seeded} medicine file(s) newly indexed"
        )
        vector_db.seed_patient_data(str(DEFAULT_PATIENT_ID))
    else:
        logger.warning("[CONFIG] Vector DB not available – RAG features disabled")
        vector_db = None

    # ── Database ──────────────────────────────────────────────────────────
    db = DatabaseManager()
    db_available = db.connect()
    if db_available:
        logger.info("[CONFIG] Database connected")
        db.seed_test_data(patient_id=str(DEFAULT_PATIENT_ID))
        db.load_session(int(DEFAULT_PATIENT_ID))
    else:
        logger.warning(
            "[CONFIG] Database not available – session will not be persisted"
        )

    # ── Chat ──────────────────────────────────────────────────────────────
    chat = OllamaChat(
        model=MODEL,
        database_manager=db if db_available else None,
        vector_db=vector_db,
    )

    print("=" * 60)
    print("  KnowledgeManagerLLM - LLM Chat Interface")
    print("=" * 60)
    print(f"Model: {MODEL}")
    if args.input:
        print(f"Mode:  agent  |  script: {args.input}")
    else:
        print("Mode:  interactive  |  'exit' or 'quit' to end")
    print("=" * 60)

    # ── Main loop ─────────────────────────────────────────────────────────
    try:
        if args.input:
            script = read_script(args.input)
            run_agent_mode(chat, script, args.delay)
        else:
            run_interactive_mode(chat)

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"[ERROR] {e}")
        print(f"Error: {e}")
        sys.exit(1)

    # ── Fine sessione ─────────────────────────────────────────────────────
    result = chat.end_session()
    if result.get("status") == "success":
        v_id = result.get("version", {}).get("id")
        if v_id:
            print(f"\n[Therapy saved – version #{v_id}]")
    print("Goodbye!")


if __name__ == "__main__":
    main()
