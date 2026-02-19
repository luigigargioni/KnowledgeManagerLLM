import os
import platform
import subprocess

import psutil

from config_loader import CHECK_NVIDIA_GPU


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
