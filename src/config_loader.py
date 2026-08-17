import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv()

# General server settings
FILE_LOG_LEVEL = os.getenv("FILE_LOG_LEVEL", "DEBUG")
TERMINAL_LOG_LEVEL = os.getenv("TERMINAL_LOG_LEVEL", "WARNING")
CHECK_NVIDIA_GPU = int(os.getenv("CHECK_NVIDIA_GPU", "0")) == 1

# ── LLM backends ─────────────────────────────────────────────────────────────
# Two roles, configured independently:
#   MAIN_LLM — the system under test: therapy manager, checker, extractors.
#   SIM_LLM  — the simulated caregiver and the judge of the test harness.
# Each has its own provider AND model, so the harness can grade a locally served
# model with a cloud one, or keep the two on separate quotas of the same cloud.
# Every SIM_* setting falls back to its MAIN counterpart when left empty.
#
# Credentials and endpoints belong to the provider, not to the role: both roles
# on Groq share GROQ_API_KEY.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_URL = os.getenv("OPENAI_URL", "")  # empty = the SDK default
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1")

SUPPORTED_PROVIDERS = ("openai", "groq", "ollama")

_DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-5.4-mini",
    "ollama": "gpt-oss:20b",
}

# Client-side rate limits, per provider. The Groq figures are the free tier
# (30 RPM, 8K TPM, 1K RPD, 200K TPD, counted per model); OpenAI and Ollama are
# unthrottled here. 0 disables a limit.
_RATE_LIMIT_DEFAULTS = {
    "groq": {"rpm": 30, "tpm": 8000, "rpd": 1000, "tpd": 200_000},
    "openai": {"rpm": 0, "tpm": 0, "rpd": 0, "tpd": 0},
    "ollama": {"rpm": 0, "tpm": 0, "rpd": 0, "tpd": 0},
}


@dataclass(frozen=True)
class LLMConfig:
    """
    Everything needed to talk to one backend, for one role.

    `temperature` and `seed` are None when unset, and a None is not sent at all —
    the provider's own default applies, which is what this project did for its
    whole life before the two knobs existed. They are separate settings on
    purpose: for reproducibility `seed` is the better lever, because it makes
    draws repeatable while leaving the sampling distribution at the value the
    model card recommends (temperature=1.0 for gpt-oss, see .env.example).

    TO VERIFY on the intended test backend: OpenAI's *hosted* reasoning models
    reject a non-default temperature whenever `reasoning_effort` is also sent
    ("Unsupported value: 'temperature' does not support 0 with this model. Only
    the default (1) value is supported." — measured on gpt-5.4-mini, reproducible
    3/3). Ollama's OpenAI-compatibility docs list `temperature`, `seed` and
    `reasoning_effort` together as supported, so the conflict is expected to be an
    OpenAI API policy rather than a property of gpt-oss — but that has NOT been
    confirmed empirically against gpt-oss:20b on Ollama. Check it before trusting
    a run that sets temperature and reasoning_effort at the same time.
    """

    role: str
    provider: str
    model: str
    api_key: str
    base_url: str | None
    timeout: int
    reasoning_effort: str
    temperature: float | None
    seed: int | None
    rpm: int
    tpm: int
    rpd: int
    tpd: int
    completion_reserve: int
    max_retries: int
    max_retry_wait: float

    @property
    def quota_key(self) -> str:
        """Identity of the quota this config consumes (providers count per model)."""
        return f"{self.provider}:{self.model}"

    def describe(self) -> str:
        return f"{self.role}={self.provider}/{self.model}"


def _env(name: str, prefix: str = "", default: str = "") -> str:
    return os.getenv(f"{prefix}{name}", default).strip()


def _resolve_provider(prefix: str, fallback: str | None) -> str:
    """Explicit PROVIDER wins; otherwise infer from the keys, then fall back."""
    provider = _env("PROVIDER", prefix).lower()
    if provider:
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unknown {prefix}PROVIDER '{provider}' in .env – "
                f"supported: {', '.join(SUPPORTED_PROVIDERS)}"
            )
        return provider
    if fallback:
        return fallback
    if OPENAI_API_KEY:
        return "openai"
    if GROQ_API_KEY:
        return "groq"
    return "ollama"


def _credentials(provider: str) -> tuple[str, str | None]:
    if provider == "openai":
        return OPENAI_API_KEY, (OPENAI_URL or None)
    if provider == "groq":
        return GROQ_API_KEY, GROQ_URL
    # Ollama ignores the key but the OpenAI SDK insists on a non-empty one
    return "ollama", f"{OLLAMA_URL}/v1"


def _int_env(name: str, prefix: str, default: int) -> int:
    raw = _env(name, prefix)
    return int(raw) if raw else default


def _optional_num_env(name: str, prefix: str, base_value, cast):
    """
    An optional numeric setting: None means "do not send the parameter at all".

    Follows the same three-way convention as REASONING_EFFORT, because a role has
    to be able to opt out of a value the other one uses:
      - variable absent          → inherit from the base role (None for MAIN)
      - variable present, empty  → None, i.e. do not send it, no inheritance
      - variable present, valued → that value
    """
    raw = os.getenv(f"{prefix}{name}")
    if raw is None:
        return base_value
    raw = raw.strip()
    if not raw:
        return None
    try:
        return cast(raw)
    except ValueError as e:
        raise ValueError(f"{prefix}{name}='{raw}' in .env is not a valid {cast.__name__}") from e


def _build_config(role: str, prefix: str, base: "LLMConfig | None" = None) -> LLMConfig:
    """
    Build one role's configuration. `base` is the config this role falls back to
    when a setting is left empty (SIM_* inherits from MAIN unless overridden).
    """
    provider = _resolve_provider(prefix, base.provider if base else None)
    model = _env("MODEL", prefix) or (
        base.model if base and base.provider == provider else _DEFAULT_MODELS[provider]
    )
    api_key, base_url = _credentials(provider)
    limits = _RATE_LIMIT_DEFAULTS[provider]

    reasoning = os.getenv(f"{prefix}REASONING_EFFORT")
    if reasoning is None:
        reasoning = base.reasoning_effort if base else "low"

    return LLMConfig(
        role=role,
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=_int_env("LLM_TIMEOUT", prefix, base.timeout if base else 120),
        reasoning_effort=reasoning.strip(),
        temperature=_optional_num_env(
            "TEMPERATURE", prefix, base.temperature if base else None, float
        ),
        seed=_optional_num_env("SEED", prefix, base.seed if base else None, int),
        rpm=_int_env("LLM_RPM", prefix, limits["rpm"]),
        tpm=_int_env("LLM_TPM", prefix, limits["tpm"]),
        rpd=_int_env("LLM_RPD", prefix, limits["rpd"]),
        tpd=_int_env("LLM_TPD", prefix, limits["tpd"]),
        completion_reserve=_int_env("LLM_COMPLETION_RESERVE", prefix, 1200),
        max_retries=_int_env("LLM_MAX_RETRIES", prefix, 5),
        max_retry_wait=float(_int_env("LLM_MAX_RETRY_WAIT", prefix, 300)),
    )


MAIN_LLM = _build_config("main", prefix="")
SIM_LLM = _build_config("sim", prefix="SIM_", base=MAIN_LLM)

# Backwards-compatible aliases: the rest of the code reads these names.
LLM_PROVIDER = MAIN_LLM.provider
MODEL = MAIN_LLM.model
SIM_PROVIDER = SIM_LLM.provider
SIM_MODEL = SIM_LLM.model
LLM_TIMEOUT = MAIN_LLM.timeout
REASONING_EFFORT = MAIN_LLM.reasoning_effort


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "therapy_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

# quote_plus on the credentials: a password containing '@', ':', '/' or '#' —
# ordinary in a generated one — otherwise splits the URL in the wrong place and
# the connection fails with an error that points at the host, not at the password.
DB_CONNECTION_STRING = (
    f"postgresql://{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


THERAPY_FILE = Path(__file__).parent.parent / "data" / "therapy.json"
# Seed for THERAPY_FILE, which is untracked because its content is whatever ran
# last. This one is versioned, so a fresh clone starts from a real patient
# instead of an empty stub. Never written to at runtime.
THERAPY_SEED_FILE = Path(__file__).parent.parent / "data" / "therapy.example.json"
LOGS_FOLDER = Path(__file__).parent.parent / "logs"
MEDICINES_FOLDER = Path(__file__).parent.parent / "data" / "medicines"
PATIENTS_DATA_FOLDER = Path(__file__).parent.parent / "data" / "patients"
CHROMA_DB_PATH = Path(__file__).parent.parent / "chromadb"

DEFAULT_PATIENT_ID = os.getenv("DEFAULT_PATIENT_ID", "1")

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"
RESULTS_DIR = LOGS_FOLDER / "batch_results"
