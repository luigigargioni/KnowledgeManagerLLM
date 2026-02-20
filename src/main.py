from time import time

import prompts as prompts
import tools as tools
from chat import OllamaChat
from config_loader import DEFAULT_PATIENT_ID, MODEL
from sql_db import DatabaseManager
from utils import get_system_info, setup_logger
from vector_db import VectorDBManager

logger = setup_logger()


def main():
    cpu_info, ram_info, gpu_info = get_system_info()

    logger.info(
        f"[SYS] CPU:{cpu_info['model']} Cores:{cpu_info['cores']}/{cpu_info['threads']} RAM:{ram_info:.0f} GB"
    )

    if len(gpu_info) > 0:
        for info in gpu_info:
            logger.info(f"[SYS] GPU {info['gpu']}: {info['name']} ({info['memory']})")

    # Vector DB initialisation
    vector_db = VectorDBManager()
    vdb_available = vector_db.initialize()
    if vdb_available:
        seeded = vector_db.seed_medicines()
        logger.info(
            f"[CONFIG] Vector DB ready – {seeded} medicine file(s) newly indexed"
        )
        # Seed patient data from files (idempotent)
        vector_db.seed_patient_data(str(DEFAULT_PATIENT_ID))
    else:
        logger.warning("[CONFIG] Vector DB not available – RAG features disabled")
        vector_db = None

    # Database connection
    db = DatabaseManager()
    db_available = db.connect()
    if db_available:
        logger.info("[CONFIG] Database connected")
        db.seed_test_data()
        db.load_session(
            DEFAULT_PATIENT_ID
        )  # COMMENT THIS LINE if you need to test the system by editing the .JSON file directly
    else:
        logger.warning(
            "[CONFIG] Database not available - session will not be persisted"
        )

    # Chat data
    model_name = MODEL
    system_prompt = prompts._THERAPY_MANAGER_PROMPT

    # Chat initialization
    chat = OllamaChat(
        model=model_name,
        system_prompt=system_prompt,
        database_manager=db if db_available else None,
        vector_db=vector_db,
    )

    print("=" * 60)
    print("  KnowledgeManagerLLM - LLM Chat Interface")
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
                result = chat.end_session()
                if result.get("status") == "success":
                    v_id = result.get("version", {}).get("id")
                    if v_id:
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
            result = chat.end_session()
            if result.get("status") == "success":
                v_id = result.get("version", {}).get("id")
                if v_id:
                    print(f"\n[Therapy saved to database – version #{v_id}]")

            logger.info("[SESSION] Chat interrupted by user (Ctrl+C)")
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            logger.exception(f"[ERROR] Unexpected error: {str(e)}")
            print(f"\nUnexpected error: {str(e)}")
            continue


if __name__ == "__main__":
    main()
