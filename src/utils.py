import logging
import platform
import subprocess
import sys
from datetime import datetime

import psutil

import prompts as prompts
import tools as tools
from config_loader import (
    CHECK_NVIDIA_GPU,
    FILE_LOG_LEVEL,
    LOGS_FOLDER,
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
        self.filter_string = filter_string

    def filter(self, record):
        return record.getMessage().startswith(self.filter_string)


def setup_logger():
    """Configura il logger per scrivere su file di sessione nella cartella logs e su terminale"""

    logs_dir = LOGS_FOLDER
    logs_dir.mkdir(exist_ok=True)

    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = logs_dir / session_timestamp
    session_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("knowledge_manager")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

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

    logger.session_dir = session_dir

    logger.info(f"[SESSION] New chat session started ID:{session_timestamp}")

    return logger


def addAgentFilterLogger(agent_name):
    logger = logging.getLogger("knowledge_manager")
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
