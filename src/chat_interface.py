import json
from time import time

import streamlit as st

import prompts
from chat import OllamaChat
from config_loader import MODEL, PATIENT_ID, THERAPY_FILE
from database import DatabaseManager
from utils import get_system_info, setup_logger

if "logger" not in st.session_state:
    st.session_state.logger = setup_logger()
logger = st.session_state.logger

if "db" not in st.session_state:
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
    st.session_state.db = db
    st.session_state.db_available = db_available

if "chat" not in st.session_state:
    st.session_state.chat = OllamaChat(
        model=MODEL,
        system_prompt=prompts.system,
        database_manager=st.session_state.db if st.session_state.db_available else None,
    )
    st.session_state.first_message = st.session_state.chat.conversation_history[-1]

if "conversation" not in st.session_state:
    st.session_state.conversation = []


st.set_page_config(page_title="KnowledgeManagerLLM", page_icon="🤖", layout="centered")
st.title("KnowledgeManagerLLM")

with st.chat_message(st.session_state.first_message["role"]):
    st.markdown(st.session_state.first_message["content"])

for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

user_message = st.chat_input("Write a message...")
if user_message:
    with st.chat_message("user"):
        st.markdown(user_message)
    st.session_state.conversation.append({"role": "user", "message": user_message})

    with st.chat_message("assistant"):
        start = time()
        with st.spinner("Thinking..."):
            response_gen = st.session_state.chat.send_message(user_message)
        elapsed = time() - start

        st.session_state.conversation.append(
            {"role": "assistant", "message": response_gen}
        )

        logger.debug(f"[TIMING] Total elapsed time: {time() - start:.2f}s")
        logger.info(f"[CHAT] ASSISTANT: {response_gen}")

        with st.container():
            st.markdown(response_gen)
            st.badge(f"🕛{elapsed:.2f}s")


# SIDEBAR

with st.sidebar:
    st.subheader("💾 Session")
    db_status = "✅ Connected" if st.session_state.db_available else "❌ Not available"
    st.caption(f"Database: {db_status}")

    if st.button(
        "Save therapy",
        use_container_width=True,
        disabled=not st.session_state.db_available,
    ):
        result = st.session_state.db.save_session(notes="Saved manually from Streamlit")
        if result["status"] == "success":
            v_id = result["version"]["id"]
            logger.info(f"[SESSION] Therapy saved manually - version #{v_id}")
            st.success(f"Version #{v_id} saved!")
        else:
            logger.error(f"[SESSION] Save failed: {result['message']}")
            st.error(result["message"])

    st.divider()
    st.subheader("📋 Therapy")

    therapy_path = THERAPY_FILE
    if therapy_path.exists():
        try:
            therapy_data = json.loads(therapy_path.read_text(encoding="utf-8"))
            patient_name = therapy_data.get("patient_full_name", "N/A")
            conditions = therapy_data.get("medical_conditions", [])
            activities = therapy_data.get("activities", [])
            expired_activities = therapy_data.get("expired_activities", [])

            days_map = {
                1: "Mon",
                2: "Tue",
                3: "Wed",
                4: "Thu",
                5: "Fri",
                6: "Sat",
                7: "Sun",
            }
            st.markdown(f"**Patient:** {patient_name}")

            if conditions:
                with st.expander(f"Medical conditions ({len(conditions)})"):
                    for c in conditions:
                        st.markdown(f"- {c}")

            if activities:
                with st.expander(f"Activities ({len(activities)})"):
                    for act in activities:
                        days = ", ".join(
                            days_map[d] for d in act.get("day_of_week", [])
                        )

                        st.markdown(
                            f"**{act['name']}**  \n"
                            f"🕐 {act['time']} · ⏱️ {act['duration_minutes']}min · 📅 {days}"
                        )

                        if act.get("dependencies"):
                            st.caption(f"Depends on: {', '.join(act['dependencies'])}")
                        st.write("")

            if expired_activities:
                with st.expander(
                    f"Activities that expired today ({len(expired_activities)})"
                ):
                    for act in expired_activities:
                        days = ", ".join(
                            days_map[d] for d in act.get("day_of_week", [])
                        )

                        st.markdown(
                            f"**{act['name']}**  \nUntil: {act['valid_until']}  \n"
                            f"🕐 {act['time']} · ⏱️ {act['duration_minutes']}min · 📅 {days}"
                        )

                        if act.get("dependencies"):
                            st.caption(f"Depends on: {', '.join(act['dependencies'])}")
                        st.write("")

        except Exception as e:
            logger.error(f"[UI] Error reading therapy.json: {e}")
            st.error(f"Errore lettura terapia: {e}")
    else:
        st.warning("therapy.json not found")

    st.divider()
    with st.expander("⚙️ System info"):
        cpu_info, ram_info, gpu_info = get_system_info()
        st.markdown(
            f"**CPU:** {cpu_info['model']} {cpu_info['cores']}/{cpu_info['threads']}"
        )
        st.markdown(f"🧠 **RAM:** {ram_info:.0f} GB")
        if gpu_info:
            for info in gpu_info:
                st.markdown(
                    f"🎮 **GPU {info['gpu']}:** {info['name']} ({info['memory']})"
                )
