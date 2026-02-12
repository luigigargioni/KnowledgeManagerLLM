import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# General server settings
FILE_LOG_LEVEL = os.getenv("FILE_LOG_LEVEL", "DEBUG")
TERMINAL_LOG_LEVEL = os.getenv("TERMINAL_LOG_LEVEL", "WARNING")
MODEL = os.getenv("MODEL", "qwen2.5:14b")
CHECK_NVIDIA_GPU=int(os.getenv("CHECK_NVIDIA_GPU", "0"))==1