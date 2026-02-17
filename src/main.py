import logging
import sys
from datetime import datetime
from pathlib import Path
from time import time

import prompts as prompts
import tools as tools
from chat import OllamaChat
from config_loader import FILE_LOG_LEVEL, MODEL, PATIENT_ID, TERMINAL_LOG_LEVEL
from database import DatabaseManager
from utils import get_system_info


def setup_logger():
    """Configura il logger per scrivere su file di sessione nella cartella logs e su terminale"""

    logs_dir = Path("../logs")
    logs_dir.mkdir(exist_ok=True)

    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"chat_session_{session_timestamp}.log"

    logger = logging.getLogger("knowledge_manager")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File Handler
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(FILE_LOG_LEVEL)
    file_handler.setFormatter(file_formatter)

    # Terminal Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(TERMINAL_LOG_LEVEL)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"[SESSION] New chat session started ID:{session_timestamp}")

    return logger


logger = setup_logger()


def main():
    cpu_info, ram_info, gpu_info = get_system_info()

    logger.info(
        f"[SYS] CPU:{cpu_info['model']} Cores:{cpu_info['cores']}/{cpu_info['threads']} RAM:{ram_info:.0f} GB"
    )

    if len(gpu_info) > 0:
        for info in gpu_info:
            logger.info(f"[SYS] GPU {info['gpu']}: {info['name']} ({info['memory']})")

    # Database connection
    db = DatabaseManager()
    db_available = db.connect()
    if db_available:
        logger.info("[CONFIG] Database connected")
        db.load_session(PATIENT_ID)
        db.seed_test_data()
    else:
        logger.warning(
            "[CONFIG] Database not available - session will not be persisted"
        )

    # Chat data
    model_name = MODEL
    system_prompt = prompts.system

    # Chat initialization
    chat = OllamaChat(model=model_name, system_prompt=system_prompt)

    print("=" * 60)
    print("  OLLAMA CHAT - Local LLM Interface")
    print("=" * 60)
    print(f"Model: {model_name}")
    print("Commands: 'exit' or 'quit' to end session")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            # Manual exit
            if user_input.lower() in ["exit", "quit", "esci"]:
                if db_available:
                    result = db.save_session()
                    if result["status"] == "success":
                        v_id = result["version"]["id"]
                        print(f"\n[Terapia salvata nel database - versione #{v_id}]")

                logger.info("[SESSION] Chat session ended by user")
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            start = time()
            response = chat.send_message(user_input)
            logger.debug(f"[TIMING] Total elapsed time: {time() - start:.2f}s")
            logger.info(f"[CHAT] ASSISTANT: {response}")

            if response:
                print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            if db_available:
                result = db.save_session()
                if result["status"] == "success":
                    v_id = result["version"]["id"]
                    print(f"\n[Terapia salvata nel database - versione #{v_id}]")

            logger.info("[SESSION] Chat interrupted by user (Ctrl+C)")
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            logger.exception(f"[ERROR] Unexpected error: {str(e)}")
            print(f"\nUnexpected error: {str(e)}")
            continue


if __name__ == "__main__":
    main()
