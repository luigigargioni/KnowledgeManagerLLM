import logging
import os
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


def load_markdown_for_llm(filename: str) -> str:
    """
    Legge un file .md dalla cartella data/facts/ e restituisce il contenuto come stringa.

    Args:
        filename: nome del file (con o senza estensione .md)

    Returns:
        Il contenuto del file come stringa
    """
    if not filename.endswith(".md"):
        filename += ".md"

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "data", "facts", filename)

    if not os.path.exists(filepath):
        return f"File non found: {filepath}"

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    return content


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


def setup_logger():
    """Configura il logger per scrivere su file di sessione nella cartella logs e su terminale"""

    logs_dir = LOGS_FOLDER
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
