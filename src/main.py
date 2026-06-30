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
        db.seed_test_data(patient_id=str(DEFAULT_PATIENT_ID))
        db.load_session(
            int(DEFAULT_PATIENT_ID)
        )  # COMMENT THIS LINE if you need to test the system by editing the .JSON file directly
    else:
        logger.warning(
            "[CONFIG] Database not available - session will not be persisted"
        )

    # Chat data
    # model_name = MODEL
    # system_prompt = prompts._THERAPY_MANAGER_PROMPT

    # Chat initialization
    chat = OllamaChat(
        model=MODEL,
        database_manager=db if db_available else None,
        vector_db=vector_db,
    )

    print("=" * 60)
    print("  KnowledgeManagerLLM - LLM Chat Interface")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print("Commands: 'exit' or 'quit' to end session")
    print("=" * 60)

    # Il main chiede il primo messaggio senza sapere come è generato
    first_message = chat.chat_agent.conversation_history[-1]
    logger.info(f"[CHAT] ASSISTANT: {first_message}")
    print(f"\nAssistant: {first_message}\n")

    while True:
        try:
            user_input = input("You: ").strip()

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

            response = chat.send_message(user_input)

            if response:
                print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            result = chat.end_session()
            if result.get("status") == "success":
                v_id = result.get("version", {}).get("id")
                if v_id:
                    print(f"\n[Therapy saved – version #{v_id}]")
            print("\n\nSession interrupted. Goodbye!")
            break

        except Exception as e:
            logger.exception(f"[ERROR] {e}")
            print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()
