from time import time

import prompts as prompts
import tools as tools
from chat import OllamaChat
from config_loader import MODEL, PATIENT_ID
from database import DatabaseManager
from utils import get_system_info, setup_logger

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
        db.seed_test_data()
        db.load_session(
            PATIENT_ID
        )  # COMMENTA QUESTA LINEA SE HAI BISOGNO DI TESTARE IL SISTEMA MODIFICANDO DIRETTAMENTE IL .JSON
    else:
        logger.warning(
            "[CONFIG] Database not available - session will not be persisted"
        )

    # Chat data
    model_name = MODEL
    system_prompt = prompts.system

    # Chat initialization
    chat = OllamaChat(
        model=model_name,
        system_prompt=system_prompt,
        database_manager=db if db_available else None,
    )

    print("=" * 60)
    print("  OLLAMA CHAT - Local LLM Interface")
    print("=" * 60)
    print(f"Model: {model_name}")
    print("Commands: 'exit' or 'quit' to end session")
    print("=" * 60)
    print()

    # print of first static message
    logger.info(f"[CHAT] ASSISTANT: {chat.conversation_history[-1]['content']}")
    print(f"\nAssistant: {chat.conversation_history[-1]['content']}\n")

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
