import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import psutil

import prompts as prompts
from config_loader import (
    CHECK_NVIDIA_GPU,
    FILE_LOG_LEVEL,
    LLM_PROVIDER,
    LOGS_FOLDER,
    MODEL,
    TERMINAL_LOG_LEVEL,
    THERAPY_FILE,
)


def get_system_info():
    cpu_info = {
        "model": platform.processor() or "Unknown CPU",
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
    }

    ram_info = psutil.virtual_memory().total / (1024**3)
    gpu_info = []
    if CHECK_NVIDIA_GPU:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            for i, line in enumerate(result.stdout.strip().splitlines()):
                name, mem = line.split(",")
                gpu_info.append({"gpu": i, "name": name.strip(), "memory": mem.strip()})
        except Exception:
            pass
    return cpu_info, ram_info, gpu_info


def hhmm_to_minutes(hhmm):
    hours, minutes = map(int, hhmm.split(":"))
    return hours * 60 + minutes


def minutes_to_hhmm(total_minutes):
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


class StartWithFilter(logging.Filter):
    def __init__(self, filter_string: str = ""):
        self.filter_string = filter_string

    def filter(self, record):
        return record.getMessage().startswith(self.filter_string)


def setup_logger(
    logs_dir: Path = LOGS_FOLDER,
    session_folder_name=None,
    logger_name: str = "knowledge_manager",
):
    """Configure the logger to write to a session file in the logs folder and to the terminal"""

    logs_dir.mkdir(exist_ok=True)
    if not session_folder_name:
        session_folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    session_dir = logs_dir / session_folder_name
    session_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.session_dir = session_dir

    for hand in logger.handlers:
        logger.removeHandler(hand)

    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File Handler General
    file_handler = logging.FileHandler(f"{session_dir}/full.log", encoding="utf-8")
    file_handler.setLevel(FILE_LOG_LEVEL)
    file_handler.setFormatter(file_formatter)

    # Chat-only log file
    chat_handler = logging.FileHandler(f"{session_dir}/chat.log", encoding="utf-8")
    chat_handler.setLevel(FILE_LOG_LEVEL)
    chat_formatter = logging.Formatter(
        "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    chat_handler.setFormatter(chat_formatter)
    chat_handler.addFilter(StartWithFilter(filter_string="[CHAT]"))

    # Terminal Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(TERMINAL_LOG_LEVEL)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(chat_handler)
    logger.addHandler(console_handler)

    logger.info(f"[SESSION] New chat session started ID:{session_folder_name}")
    cpu_info, ram_info, gpu_info = get_system_info()
    logger.info(
        f"[SESSION] CPU: {cpu_info['model']} {cpu_info['cores']}/{cpu_info['threads']}\tRAM:{ram_info:.0f} GB"
    )
    if len(gpu_info) > 0:
        for info in gpu_info:
            logger.info(
                f"[SESSION] GPU {info['gpu']}: {info['name']} ({info['memory']})"
            )

    logger.info(f"[SESSION] Provider={LLM_PROVIDER} Model={MODEL}")

    return logger


def get_current_logger():
    return logging.getLogger("knowledge_manager")


def addAgentFilterLogger(agent_name):
    logger = get_current_logger()
    agent_handler = logging.FileHandler(
        f"{logger.session_dir}/agent_{agent_name}.log", encoding="utf-8"
    )
    agent_handler.setLevel(FILE_LOG_LEVEL)
    agent_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    agent_handler.setFormatter(agent_formatter)
    agent_handler.addFilter(StartWithFilter(filter_string=f"[{agent_name.upper()}]"))
    logger.addHandler(agent_handler)


_LOG_LINE_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - \[CHAT\] (USER|ASSISTANT): (.*)$"
)

# Map the log marker to the OpenAI-style role
_ROLE_MAP = {
    "USER": "user",
    "ASSISTANT": "assistant",
}


def parse_chat_log(log_path: str | Path) -> list[dict]:
    """
    Parse a log file in the format:
        YYYY-MM-DD HH:MM:SS - [CHAT] USER: <message>
        YYYY-MM-DD HH:MM:SS - [CHAT] ASSISTANT: <multi-line message>

    Reconstruct the conversation_history with correct roles, handling
    messages that span multiple lines (e.g. formatted assistant replies).

    Args:
        log_path: path to the .log file

    Returns:
        List of dicts {"role": "user"|"assistant", "content": "..."}
        ready to populate agent.conversation_history.
    """
    log_path = Path(log_path)
    lines = log_path.read_text(encoding="utf-8").splitlines()

    history: list[dict] = []
    current_role: str | None = None
    current_content: list[str] = []

    def _flush():
        """Close the current message and add it to the history."""
        if current_role is not None:
            content = "\n".join(current_content).strip()
            if content:
                history.append({"role": current_role, "content": content})

    for line in lines:
        match = _LOG_LINE_PATTERN.match(line)

        if match:
            # New log line recognized: close the previous message
            _flush()
            marker, first_line = match.groups()
            current_role = _ROLE_MAP[marker]
            current_content = [first_line]
        else:
            # Continuation line (e.g. bullet point of an ASSISTANT reply)
            # Skip empty lines before the first recognized message
            if current_role is not None:
                current_content.append(line)

    # Flush the last message remaining in the buffer
    _flush()

    return history


def populate_agent_history(
    agent, log_path: str | Path, keep_system_prompt: bool = True
) -> None:
    """
    Populate agent.conversation_history from a log file,
    preserving the initial system prompt if present.

    Args:
        agent: Agent instance (e.g. TherapyManagerAgent) whose history should be populated
        log_path: path to the .log file
        keep_system_prompt: if True, keeps the existing system message
                            at the top of the history before appending the parsed messages
    """
    parsed_messages = parse_chat_log(log_path)

    if keep_system_prompt and agent.conversation_history:
        agent.conversation_history = agent.conversation_history + parsed_messages
    else:
        agent.conversation_history = parsed_messages


def copy_session_therapy():
    logger = get_current_logger()
    session_log_dir = logger.session_dir

    os.makedirs(session_log_dir, exist_ok=True)

    shutil.copy2(THERAPY_FILE, os.path.join(session_log_dir, "therapy.json"))


def load_past_session(
    agent,
    session_log_dir: str | Path,
    keep_system_prompt: bool = True,
) -> list[dict]:
    """
    Load a previous session into the agent's history and
    return the therapy snapshots found.

    Args:
        agent: Agent instance to populate
        session_log_dir: previous session folder (e.g. logs/session_20260630)
        keep_system_prompt: if True, keeps the initial system prompt

    Returns:
        List of therapy snapshots [{"message_idx": int, "therapy": dict}, ...]
        to assign to chat._therapy_snapshots.
        Empty list if the file does not exist.
    """
    logger = get_current_logger()
    session_log_dir = Path(session_log_dir)
    chat_log = session_log_dir / "chat.log"
    snapshots_path = session_log_dir / "therapy_snapshots.json"

    if not chat_log.exists():
        raise FileNotFoundError(f"chat.log not found in {session_log_dir}")

    # Populate the history with messages from the previous session
    populate_agent_history(agent, chat_log, keep_system_prompt=keep_system_prompt)

    # Load snapshots if present
    if snapshots_path.exists():
        snapshots = json.loads(snapshots_path.read_text(encoding="utf-8"))
        logger.info(
            f"[LOAD] Loaded {len(snapshots)} therapy snapshots from {session_log_dir}"
        )
    else:
        logger.warning(
            f"[LOAD] No therapy_snapshots.json found in {session_log_dir} – "
            "starting with empty snapshot list"
        )
        snapshots = []

    return snapshots


def build_transcript(conversation_history: list[dict]) -> str:
    """
    Build a readable transcript for the JudgeAgent
    from the chat_agent's conversation history.
    Filter only user/assistant messages.
    """
    lines = []
    for msg in conversation_history:
        role = msg.get("role")
        if role == "user":
            lines.append(f"CAREGIVER: {msg['content']}")
        elif role == "assistant":
            lines.append(f"CHATBOT: {msg['content']}")
    return "\n\n".join(lines)
