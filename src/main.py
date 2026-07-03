import argparse
from pathlib import Path
from time import time

import tools as tools
from chat import OllamaChat
from config_loader import DEFAULT_PATIENT_ID, MODEL
from sql_db import DatabaseManager
from utils import get_system_info, setup_logger
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
        help="Path to a text file containing one user message per line. "
        "If omitted, runs in interactive mode.",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=0.0,
        help="Seconds to wait between messages in file mode (default: 0).",
    )
    return parser.parse_args()


def read_messages_from_file(path: Path) -> list[str]:
    """
    Legge i messaggi utente da un file di testo.
    - Una riga = un messaggio
    - Righe vuote e righe che iniziano con # vengono ignorate
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    messages = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            messages.append(stripped)

    if not messages:
        raise ValueError(f"No messages found in {path}")

    return messages


def message_generator(args: argparse.Namespace):
    """
    Restituisce un iteratore di messaggi utente.
    In modalità file: legge dal file e termina automaticamente.
    In modalità interattiva: legge da stdin finché l'utente non esce.
    """
    if args.input:
        messages = read_messages_from_file(args.input)
        logger.info(f"[MODE] File mode – {len(messages)} messages from {args.input}")
        for msg in messages:
            yield msg
    else:
        logger.info("[MODE] Interactive mode")
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                # stdin chiuso (es. pipe), termina silenziosamente
                return
            if user_input:
                yield user_input
            elif user_input.lower() in ["exit", "quit", "esci"]:
                return


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
        print(f"Mode:  file ({args.input})")
    else:
        print("Mode:  interactive  |  'exit' or 'quit' to end")
    print("=" * 60)

    # first_message = chat.get_first_message()
    # logger.info(f"[CHAT] ASSISTANT: {first_message}")
    # print(f"\nAssistant: {first_message}\n")

    # ── Main loop ─────────────────────────────────────────────────────────
    try:
        for user_input in message_generator(args):
            # Gestione uscita in modalità interattiva
            if not args.input and user_input.lower() in ["exit", "quit", "esci"]:
                break

            # In modalità file stampa il messaggio come se l'utente lo avesse scritto
            if args.input:
                print(f"You: {user_input}")

            start = time()
            response = chat.send_message(user_input)
            elapsed = time() - start

            if response:
                print(f"\nAssistant: {response}")
                if args.input:
                    print(f"[{elapsed:.2f}s]")
                print()

            # Pausa opzionale tra messaggi in modalità file
            if args.input and args.delay > 0:
                import time as time_mod

                time_mod.sleep(args.delay)

    except KeyboardInterrupt:
        print("\n\nInterrupted.")

    # ── Fine sessione ─────────────────────────────────────────────────────
    result = chat.end_session()
    if result.get("status") == "success":
        v_id = result.get("version", {}).get("id")
        if v_id:
            print(f"\n[Therapy saved – version #{v_id}]")
    print("Goodbye!")


if __name__ == "__main__":
    main()
