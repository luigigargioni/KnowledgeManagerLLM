import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# General server settings
FILE_LOG_LEVEL = os.getenv("FILE_LOG_LEVEL", "DEBUG")
TERMINAL_LOG_LEVEL = os.getenv("TERMINAL_LOG_LEVEL", "WARNING")
MODEL = os.getenv("MODEL", "qwen2.5:14b")
CHECK_NVIDIA_GPU = int(os.getenv("CHECK_NVIDIA_GPU", "0")) == 1
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "therapy_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

DB_CONNECTION_STRING = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


THERAPY_FILE = Path(__file__).parent.parent / "data" / "therapy.json"
LOGS_FOLDER = Path(__file__).parent.parent / "logs"

PATIENT_ID = os.getenv("PATIENT_ID", "test")
