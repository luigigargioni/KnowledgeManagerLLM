"""
Log Debugger - Interactive log viewer for KnowledgeManagerLLM sessions
"""

import json
import re
from datetime import datetime

import streamlit as st

from config_loader import THERAPY_FILE

# Must be first Streamlit command
st.set_page_config(
    page_title="Log Debugger - KnowledgeManagerLLM", page_icon="🐛", layout="wide"
)

st.title("🐛 Log Debugger")
st.caption("Interactive log viewer for chat sessions")


# ─── LOG PARSING ──────────────────────────────────────────────────────────────


def parse_log_file(log_lines: list[str]) -> tuple[list[dict], list[str]]:
    """
    Parse log file handling multi-line messages

    Returns: (parsed_entries, unparsed_lines)
    """
    pattern = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - (\w+) - \[(\w+)\] (.+)$"

    parsed_entries = []
    unparsed_lines = []
    current_entry = None

    for line in log_lines:
        match = re.match(pattern, line)

        if match:
            # Save previous entry if exists
            if current_entry:
                parsed_entries.append(current_entry)

            # Start new entry
            timestamp_str, level, tag, message = match.groups()
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

            current_entry = {
                "timestamp": timestamp,
                "level": level,
                "tag": tag,
                "message": message,
                "raw": line,
            }
        elif current_entry:
            # Continuation of previous message
            current_entry["message"] += "\n" + line
            current_entry["raw"] += "\n" + line
        elif line.strip():
            # Orphaned line (no current entry)
            unparsed_lines.append(line)

    # Don't forget the last entry
    if current_entry:
        parsed_entries.append(current_entry)

    return parsed_entries, unparsed_lines


def extract_chat_conversation(log_entries: list[dict]) -> list[dict]:
    """Extract only [CHAT] entries and structure them as conversation"""
    conversation = []

    for entry in log_entries:
        if entry["tag"] != "CHAT":
            continue

        msg = entry["message"]

        # Parse USER: and ASSISTANT: messages
        if msg.startswith("USER: "):
            conversation.append(
                {
                    "role": "user",
                    "message": msg[6:],  # Remove "USER: " prefix
                    "timestamp": entry["timestamp"],
                }
            )
        elif msg.startswith("ASSISTANT: "):
            conversation.append(
                {
                    "role": "assistant",
                    "message": msg[11:],  # Remove "ASSISTANT: " prefix
                    "timestamp": entry["timestamp"],
                }
            )

    return conversation


def extract_tools_used(log_entries: list[dict]) -> list[dict]:
    """Extract tool execution information"""
    tools = []

    for entry in log_entries:
        if entry["tag"] != "TOOL":
            continue

        msg = entry["message"]
        if msg.startswith("Executing: "):
            tools.append(
                {
                    "timestamp": entry["timestamp"],
                    "execution": msg[11:],  # Remove "Executing: " prefix
                    "raw": entry["raw"],
                }
            )

    return tools


def get_session_stats(log_entries: list[dict]) -> dict:
    """Calculate session statistics"""
    if not log_entries:
        return {}

    tags = {}
    levels = {}

    for entry in log_entries:
        tags[entry["tag"]] = tags.get(entry["tag"], 0) + 1
        levels[entry["level"]] = levels.get(entry["level"], 0) + 1

    # Extract timing information
    timings = []
    for entry in log_entries:
        if entry["tag"] == "TIMING" and "elapsed time:" in entry["message"]:
            match = re.search(r"(\d+\.?\d*)s", entry["message"])
            if match:
                timings.append(float(match.group(1)))

    return {
        "total_lines": len(log_entries),
        "start_time": log_entries[0]["timestamp"] if log_entries else None,
        "end_time": log_entries[-1]["timestamp"] if log_entries else None,
        "tags": tags,
        "levels": levels,
        "timings": timings,
        "avg_response_time": sum(timings) / len(timings) if timings else 0,
    }


# ─── FILE UPLOAD ──────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload a log file",
    type=["log", "txt"],
    help="Select a session log file from the logs/ directory",
)

if uploaded_file is None:
    st.info("👆 Upload a log file to start debugging")
    st.stop()

# Read and parse the log file
log_content = uploaded_file.read().decode("utf-8")
log_lines = log_content.strip().split("\n")

parsed_entries, unparsed_lines = parse_log_file(log_lines)

if not parsed_entries:
    st.error("❌ No valid log entries found in this file")
    st.stop()

st.success(f"✅ Loaded {len(parsed_entries)} log entries")

if unparsed_lines:
    with st.expander(f"⚠️ {len(unparsed_lines)} unparsed lines (click to view)"):
        for line in unparsed_lines[:50]:  # Limit to 50
            st.code(line, language=None)
        if len(unparsed_lines) > 50:
            st.caption(f"... and {len(unparsed_lines) - 50} more")

with st.sidebar:
    st.subheader("👤 Patient")
    chosen = st.selectbox(
        "Select patient",
        options=["Mario Rossi"],
        index=0,
        label_visibility="collapsed",
    )
    therapy_path = THERAPY_FILE
    if therapy_path.exists():
        try:
            therapy_data = json.loads(therapy_path.read_text(encoding="utf-8"))
            patient_name = therapy_data.get("patient_full_name", "N/A")
            birth_date = therapy_data.get("birth_date", None)
            birth_date_desc = ""
            if birth_date:
                try:
                    birth_date_desc = datetime.strptime(
                        birth_date, "%Y-%m-%dT%H:%M:%S"
                    ).strftime("%d/%m/%Y")
                except Exception:
                    birth_date_desc = ""
            age = therapy_data.get("age", "0")

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
            # st.markdown(f"**Patient:** {patient_name}")

            st.markdown(f"📅{birth_date_desc} - {age} yrs.")

            if conditions:
                with st.expander(f"Medical conditions ({len(conditions)})"):
                    for c in conditions:
                        st.markdown(f"- {c}")

            st.divider()

            st.subheader("📋 Therapy")
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
                            dep_names = []
                            for dep in act["dependencies"]:
                                dep_names += [
                                    x["name"]
                                    for x in activities
                                    if x["activity_id"] == dep
                                ]
                            st.caption(f"Depends on: {', '.join(dep_names)}")
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
            st.error(f"Errore lettura terapia: {e}")
    else:
        st.warning("therapy.json not found")

    st.divider()


# ─── TABS ─────────────────────────────────────────────────────────────────────

tab_chat, tab_tools, tab_timeline, tab_stats, tab_raw = st.tabs(
    ["💬 Chat", "🔧 Tools", "📅 Timeline", "📊 Stats", "📄 Raw Logs"]
)


# ─── TAB: CHAT ────────────────────────────────────────────────────────────────

with tab_chat:
    st.header("Your Therapy Assistant")

    conversation = extract_chat_conversation(parsed_entries)

    if not conversation:
        st.info("No chat messages found in this log")
    else:
        for msg in conversation:
            with st.chat_message(msg["role"]):
                st.markdown(msg["message"])
                # st.caption(f"🕐 {msg['timestamp'].strftime('%H:%M:%S')}")


# ─── TAB: TOOLS ───────────────────────────────────────────────────────────────

with tab_tools:
    st.subheader("Tool Executions")

    tools = extract_tools_used(parsed_entries)

    if not tools:
        st.info("No tool executions found in this log")
    else:
        st.caption(f"Total tool calls: {len(tools)}")

        for i, tool in enumerate(tools, 1):
            with st.expander(f"#{i} — {tool['timestamp'].strftime('%H:%M:%S')}"):
                st.code(tool["execution"], language="python")


# ─── TAB: TIMELINE ────────────────────────────────────────────────────────────

with tab_timeline:
    st.subheader("Event Timeline")

    # Filter options
    col1, col2 = st.columns(2)

    with col1:
        selected_tags = st.multiselect(
            "Filter by tag",
            options=sorted(set(e["tag"] for e in parsed_entries)),
            default=None,
            help="Leave empty to show all",
        )

    with col2:
        selected_levels = st.multiselect(
            "Filter by level",
            options=sorted(set(e["level"] for e in parsed_entries)),
            default=None,
            help="Leave empty to show all",
        )

    # Apply filters
    filtered = parsed_entries
    if selected_tags:
        filtered = [e for e in filtered if e["tag"] in selected_tags]
    if selected_levels:
        filtered = [e for e in filtered if e["level"] in selected_levels]

    st.caption(f"Showing {len(filtered)} of {len(parsed_entries)} entries")

    # Display timeline
    for entry in filtered:
        level_emoji = {"DEBUG": "🔍", "INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌"}.get(
            entry["level"], "•"
        )

        col1, col2, col3, col4 = st.columns([1, 1, 1, 7])

        with col1:
            st.caption(entry["timestamp"].strftime("%H:%M:%S"))
        with col2:
            st.caption(f"{level_emoji} {entry['level']}")
        with col3:
            st.code(entry["tag"], language=None)
        with col4:
            # Truncate long messages
            msg = entry["message"]
            if len(msg) > 100:
                st.text(msg[:100] + "...")
                with st.popover("Show full"):
                    st.text(msg)
            else:
                st.text(msg)


# ─── TAB: STATS ───────────────────────────────────────────────────────────────

with tab_stats:
    st.subheader("Session Statistics")

    stats = get_session_stats(parsed_entries)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Entries", stats["total_lines"])

    with col2:
        if stats.get("timings"):
            st.metric("Avg Response", f"{stats['avg_response_time']:.2f}s")
        else:
            st.metric("Avg Response", "N/A")

    with col3:
        duration = None
        if stats["start_time"] and stats["end_time"]:
            duration = stats["end_time"] - stats["start_time"]
            st.metric("Session Duration", f"{duration.total_seconds():.0f}s")
        else:
            st.metric("Session Duration", "N/A")

    with col4:
        error_count = stats["levels"].get("ERROR", 0)
        st.metric("Errors", error_count, delta=None if error_count == 0 else "-")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Entries by Tag")
        sorted_tags = sorted(stats["tags"].items(), key=lambda x: x[1], reverse=True)
        for tag, count in sorted_tags:
            st.progress(count / stats["total_lines"], text=f"{tag}: {count}")

    with col2:
        st.subheader("Entries by Level")
        sorted_levels = sorted(
            stats["levels"].items(), key=lambda x: x[1], reverse=True
        )
        for level, count in sorted_levels:
            st.progress(count / stats["total_lines"], text=f"{level}: {count}")

    if stats.get("timings"):
        st.divider()
        st.subheader("Response Times")
        st.line_chart(stats["timings"])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min", f"{min(stats['timings']):.2f}s")
        with col2:
            st.metric("Max", f"{max(stats['timings']):.2f}s")
        with col3:
            st.metric("Avg", f"{stats['avg_response_time']:.2f}s")


# ─── TAB: RAW LOGS ────────────────────────────────────────────────────────────

with tab_raw:
    st.subheader("Raw Log Output")

    # Search functionality
    search_term = st.text_input("Search logs", placeholder="Enter text to filter...")

    filtered_raw = log_lines
    if search_term:
        filtered_raw = [
            line for line in log_lines if search_term.lower() in line.lower()
        ]
        st.caption(f"Found {len(filtered_raw)} matching lines")

    # Display with line numbers
    code_content = "\n".join(
        f"{i + 1:4d} | {line}" for i, line in enumerate(filtered_raw)
    )
    st.code(code_content, language=None, line_numbers=False)


# ─── FOOTER ───────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"📁 File: {uploaded_file.name} | 📊 {len(parsed_entries)} entries | "
    f"🕐 {stats['start_time'].strftime('%Y-%m-%d %H:%M:%S') if stats.get('start_time') else 'N/A'}"
)
