import json
import logging
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import psutil

from config_loader import (
    CHECK_NVIDIA_GPU,
    FILE_LOG_LEVEL,
    LOGS_FOLDER,
    MAIN_LLM,
    SIM_LLM,
    TERMINAL_LOG_LEVEL,
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
        # logging.Filter.__init__ sets .name/.nlen, which the base class and any
        # code inspecting the filter expect to exist. Skipping it happened to
        # work only because filter() is fully overridden below.
        super().__init__()
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

    # Iterate over a copy: removeHandler mutates logger.handlers, so looping over
    # the live list skipped every other handler. The batch runner calls this once
    # per scenario, so the survivors stayed attached and kept writing — each
    # scenario's lines landed in the previous scenarios' full.log and chat.log as
    # well as its own. Closing them also releases the file handles.
    for hand in list(logger.handlers):
        logger.removeHandler(hand)
        hand.close()

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
    chat_formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
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
            logger.info(f"[SESSION] GPU {info['gpu']}: {info['name']} ({info['memory']})")

    for cfg in (MAIN_LLM, SIM_LLM):
        line = f"[SESSION] {cfg.role}: provider={cfg.provider} model={cfg.model}"
        if cfg.rpm or cfg.tpm or cfg.rpd or cfg.tpd:
            line += f" | limits {cfg.rpm} RPM / {cfg.tpm} TPM / {cfg.rpd} RPD / {cfg.tpd} TPD"
        logger.info(line)

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


def populate_agent_history(agent, log_path: str | Path, keep_system_prompt: bool = True) -> None:
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
        logger.info(f"[LOAD] Loaded {len(snapshots)} therapy snapshots from {session_log_dir}")
    else:
        logger.warning(
            f"[LOAD] No therapy_snapshots.json found in {session_log_dir} – "
            "starting with empty snapshot list"
        )
        snapshots = []

    return snapshots


EXIT_WORDS = ("exit", "quit", "esci")

# The caregiver agent is told to end by sending only "exit", but in practice it
# almost always appends it to a closing pleasantry ("No further edits needed on
# my end.exit") or wraps it ("(exit)"). Matching the whole message exactly meant
# those turns were forwarded to the assistant instead and the conversation kept
# going past its intended end.
_EXIT_RE = re.compile(
    r"(?:^|[\s.!?;:,()\[\]*\"'—-])(" + "|".join(EXIT_WORDS) + r")"
    r"[\s.!?;:,()\[\]*\"']*$",
    re.IGNORECASE,
)


def is_exit_message(message: str) -> bool:
    """
    True when the caregiver signalled the end of the conversation.

    Accepts the bare keyword as well as a message whose last word is one of the
    exit keywords, which is how the simulated caregiver actually ends turns.
    """
    text = (message or "").strip()
    if not text:
        return False
    if text.lower().strip("\"'*()[]. !?") in EXIT_WORDS:
        return True
    return bool(_EXIT_RE.search(text))


# Vocabulary the assistant uses when it raises a problem and hands the decision
# back to the caregiver. Two sources: the failure messages built in tools.py
# ("cannot be scheduled at … overlaps with the activity named …", "Dependencies
# not found in schedule", "Temporal ordering violation", "Cannot remove '…'
# because it is a dependency of …"), and the safety findings the checker agent
# returns for medicines and patient history.
#
# This gates the delivery of a scenario's withheld reaction clauses
# (scenario_loader.split_objectives): the caregiver receives them only once the
# assistant has actually said the thing, so it can react instead of predicting.
# Recall matters more than precision here — a false positive puts the caregiver
# back where it was before this gate existed, while a false negative leaves it
# without its instructions for a branch that really was exercised.
_ISSUE_MARKERS = (
    # scheduling
    "conflict",
    "overlap",
    "clash",
    "same time",
    "already scheduled",
    "no available",
    # offering an alternative means a problem was found first
    "alternative",
    # dependencies and temporal ordering
    "dependenc",
    "depends on",
    "ordering",
    "cannot be removed",
    "cannot remove",
    "blocked",
    "does not exist",
    "doesn't exist",
    "not found",
    "no activity",
    # safety: medicines and patient history
    "contraindicat",
    "interaction",
    "warning",
    "warn you",
    "caution",
    "risk",
    "unsafe",
    "not recommended",
    "not advisable",
    "concern",
    "flags",
    "flagged",
    "worsen",
    "adverse",
    "implication",
    # how a history finding gets phrased once it comes back from the RAG lookup
    "past episode",
    "in the past",
    "last time",
    "history shows",
    "previously",
)


# The all-clear is phrased with the very words that name the problem: "I found
# no conflicts", "safety review flagged no direct conflict", "there is no risk".
# Matched naively, a reply saying nothing happened counted as one raising an
# issue — in scenario 17 that delivered the caregiver its reaction instructions
# one turn early, and it answered "yes, add it at the suggested alternative
# time" when no alternative had been suggested. Negated occurrences are dropped
# before the markers are looked for.
#
# Only nouns that make sense negated are listed: "not recommended" and "not
# advisable" are markers themselves and must keep matching.
_NEGATED_ISSUE_RE = re.compile(
    r"\b(?:no|not|without|never|any)\s+(?:\w+\s+){0,2}?"
    r"(?:conflicts?|overlaps?|clash\w*|interactions?|warnings?|cautions?|risks?"
    r"|concerns?|contraindicat\w*|issues?|problems?|adverse\s+\w+)",
    re.IGNORECASE,
)


def assistant_raised_issue(message: str) -> bool:
    """
    True when the assistant's reply raises a problem of its own accord.

    Scheduling conflicts, broken dependencies, ordering violations,
    contraindications and history warnings are exactly what the scenarios exist
    to test, so the caregiver must never be the one to bring them up. It is told
    how to react only from the moment this returns True.
    """
    text = _NEGATED_ISSUE_RE.sub(" ", (message or "").lower())
    return any(marker in text for marker in _ISSUE_MARKERS)


def assistant_handed_back(message: str) -> bool:
    """
    True when the assistant's reply looks like it is waiting for the caregiver.

    Used only in combination with a deterministic signal from `Chat` (a tool
    reported a conflict, a blocked dependency, a history hit): the signal proves
    a problem was found, this only has to rule out the case where the assistant
    swallowed it and answered as if nothing had happened. Asking anything at all
    is enough evidence for that, which is why this is far weaker — and far less
    wording-dependent — than assistant_raised_issue.
    """
    text = (message or "").strip()
    return "?" in text or assistant_raised_issue(text)


def is_visible_turn(msg: dict) -> bool:
    """
    True for the user/assistant turns that were actually exchanged with the
    caregiver.

    Excludes system and tool messages, and the internal assistant turns that
    only carry tool calls: those belong to the agent's working history, not to
    the conversation. Every consumer that walks the history as a dialogue
    (transcript, therapy snapshots, chat UI, knowledge extraction) must agree on
    this definition, otherwise message indexes drift apart.
    """
    role = msg.get("role")
    if role == "user":
        return True
    if role != "assistant":
        return False
    if msg.get("tool_calls"):
        return False
    return bool((msg.get("content") or "").strip())


def visible_turns(conversation_history: list[dict]) -> list[dict]:
    """Return only the turns the caregiver actually saw, in order."""
    return [m for m in conversation_history if is_visible_turn(m)]


def build_transcript(conversation_history: list[dict]) -> str:
    """
    Build a readable transcript for the JudgeAgent
    from the chat_agent's conversation history.
    Filter only user/assistant messages.
    """
    lines = []
    for msg in visible_turns(conversation_history):
        speaker = "CAREGIVER" if msg["role"] == "user" else "CHATBOT"
        lines.append(f"{speaker}: {msg['content']}")
    return "\n\n".join(lines)
