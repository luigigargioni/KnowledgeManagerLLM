import json
from time import time

import streamlit as st

import prompts
from chat import OllamaChat
from config_loader import DEFAULT_PATIENT_ID, MODEL, PATIENTS_DATA_FOLDER, THERAPY_FILE
from sql_db import DatabaseManager
from utils import get_system_info, setup_logger
from vector_db import VectorDBManager

# Must be the first Streamlit command
st.set_page_config(page_title="KnowledgeManagerLLM", page_icon="🤖", layout="centered")

# Prevent Streamlit from dimming the sidebar during script execution
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { opacity: 1 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "logger" not in st.session_state:
    st.session_state.logger = setup_logger()
logger = st.session_state.logger

# ── DB connection (once, not patient-specific) ──────────────────────────────
if "db" not in st.session_state:
    db = DatabaseManager()
    db_available = db.connect()
    if db_available:
        db.seed_test_data()
    else:
        logger.warning(
            "[CONFIG] Database not available - session will not be persisted"
        )
    st.session_state.db = db
    st.session_state.db_available = db_available


# ── Patient list helper ───────────────────────────────────────────────────
def get_available_patients() -> list[tuple[str, str]]:
    """Return (id, name) pairs for every patient that has a data folder."""
    result = []
    if not PATIENTS_DATA_FOLDER.exists():
        return result
    for folder in sorted(PATIENTS_DATA_FOLDER.iterdir()):
        if not folder.is_dir():
            continue
        pid = folder.name
        name = f"Patient {pid}"
        if st.session_state.get("db_available"):
            try:
                r = st.session_state.db.get_patient(int(pid))
                if r.get("status") == "success":
                    name = r["patient"]["name"]
            except Exception:
                pass
        result.append((pid, name))
    return result


available_patients = get_available_patients()
patient_ids = [p[0] for p in available_patients]
patient_labels = {p[0]: p[1] for p in available_patients}

if "selected_patient_id" not in st.session_state:
    # Restore from URL (survives F5), fall back to env default
    st.session_state.selected_patient_id = st.query_params.get(
        "patient", DEFAULT_PATIENT_ID
    )

if "processing" not in st.session_state:
    st.session_state.processing = False

if "pending_message" not in st.session_state:
    st.session_state.pending_message = None

# ── Sidebar – patient selector (top) ───────────────────────────────────────
with st.sidebar:
    st.subheader("👤 Patient")
    if available_patients:
        selectbox_options = [f"{patient_labels[pid]} — ID {pid}" for pid in patient_ids]
        current_idx = (
            patient_ids.index(st.session_state.selected_patient_id)
            if st.session_state.selected_patient_id in patient_ids
            else 0
        )
        chosen = st.selectbox(
            "Select patient",
            options=selectbox_options,
            index=current_idx,
            label_visibility="collapsed",
            disabled=st.session_state.processing,
        )
        chosen_id = patient_ids[selectbox_options.index(chosen)]
        if chosen_id != st.session_state.selected_patient_id:
            # Patient changed: persist in URL and reset all patient-specific state
            for key in [
                "chat",
                "vector_db",
                "conversation",
                "session_ended",
                "first_message",
                "session_loaded_for",
                "processing",
                "pending_message",
            ]:
                st.session_state.pop(key, None)
            st.session_state.selected_patient_id = chosen_id
            st.query_params["patient"] = chosen_id
            logger.info(f"[UI] Patient switched to ID {chosen_id}")
            st.rerun()
    else:
        st.warning("No patient data folders found.")
    st.divider()

# ── Vector DB (once per patient selection) ──────────────────────────────────
if "vector_db" not in st.session_state:
    vdb = VectorDBManager()
    vdb_available = vdb.initialize()
    if vdb_available:
        seeded = vdb.seed_medicines()
        vdb.seed_patient_data(st.session_state.selected_patient_id)
        st.session_state.vector_db = vdb
    else:
        logger.warning("[CONFIG] Vector DB not available – RAG features disabled")
        st.session_state.vector_db = None

# ── Load therapy session for the selected patient ────────────────────────────
if st.session_state.get("session_loaded_for") != st.session_state.selected_patient_id:
    if st.session_state.db_available:
        st.session_state.db.load_session(int(st.session_state.selected_patient_id))

    st.session_state.session_loaded_for = st.session_state.selected_patient_id

# ── Chat (once per patient selection) ────────────────────────────────────────
if "chat" not in st.session_state:
    st.session_state.chat = OllamaChat(
        model=MODEL,
        system_prompt=prompts._THERAPY_MANAGER_PROMPT,
        database_manager=st.session_state.db if st.session_state.db_available else None,
        vector_db=st.session_state.vector_db,
    )
    st.session_state.first_message = st.session_state.chat.conversation_history[-1]

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "session_ended" not in st.session_state:
    st.session_state.session_ended = False

# processing and pending_message are initialized earlier (before sidebar)


# ── Main UI ───────────────────────────────────────────────────────────────
st.title("KnowledgeManagerLLM")

if st.session_state.session_ended:
    st.success(
        "✅ Session saved. The conversation has been closed. "
        "Refresh the page or change patient to start a new session."
    )


with st.chat_message(st.session_state.first_message["role"]):
    st.markdown(st.session_state.first_message["content"])

for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

user_message = st.chat_input(
    "Write a message...",
    disabled=st.session_state.session_ended or st.session_state.processing,
)

# Phase 1: new input from the user → save it and rerun with processing=True
# so the sidebar renders with disabled widgets before the blocking call.
if user_message and not st.session_state.processing:
    st.session_state.pending_message = user_message
    st.session_state.processing = True
    st.rerun()

# Phase 2: actually process the pending message
if st.session_state.processing and st.session_state.pending_message:
    user_message = st.session_state.pending_message
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

        logger.debug(f"[TIMING] Total elapsed time: {elapsed:.2f}s")
        logger.info(f"[CHAT] ASSISTANT: {response_gen}")

        with st.container():
            st.markdown(response_gen)
            st.badge(f"🕛{elapsed:.2f}s")

    st.session_state.pending_message = None
    st.session_state.processing = False

    # Check if the LLM triggered save_session during this turn
    if st.session_state.chat.session_ended and not st.session_state.session_ended:
        logger.info("[UI] Session ended via LLM tool call – locking chat input")
        st.session_state.session_ended = True

    st.rerun()


# ── Sidebar – session, therapy, system info ──────────────────────────────────
with st.sidebar:
    st.subheader("💾 Session")
    db_status = "✅ Connected" if st.session_state.db_available else "❌ Not available"
    st.caption(f"Database: {db_status}")

    if st.button(
        "Save therapy",
        use_container_width=True,
        disabled=not st.session_state.db_available
        or st.session_state.session_ended
        or st.session_state.processing,
    ):
        logger.info("[UI] Save therapy button clicked – running end_session")
        with st.spinner("Saving session and extracting knowledge..."):
            result = st.session_state.chat.end_session()
        if result.get("status") == "success":
            v_id = result.get("version", {}).get("id")
            logger.info(f"[SESSION] Therapy saved manually – version #{v_id}")
            st.session_state.session_ended = True
            st.success(f"Version #{v_id} saved!")
            st.rerun()
        elif result.get("status") == "skipped":
            msg = result.get("message", "Session already ended")
            logger.warning(f"[SESSION] Save skipped: {msg}")
            st.info(msg)
        else:
            msg = result.get("message", "Unknown error")
            logger.error(f"[SESSION] Save failed: {msg}")
            st.error(msg)

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
            f":computer: **CPU:** {cpu_info['model']} {cpu_info['cores']}/{cpu_info['threads']}"
        )
        st.markdown(f":brain: **RAM:** {ram_info:.0f} GB")
        if gpu_info:
            for info in gpu_info:
                st.markdown(
                    f":video_game: **GPU {info['gpu']}:** {info['name']} ({info['memory']})"
                )
        st.markdown(f":space_invader: **LLM Model:** {MODEL}")
