# llm_client.py
"""
Single entry point for the LLM backends.

Two things live here:

1. `make_client(config)` — builds an OpenAI-compatible client from an
   `LLMConfig` (OpenAI, Groq or Ollama; all three speak the same protocol, only
   the base URL and the key differ). The two roles of the project each get their
   own: `make_main_client()` for the system under test, `make_sim_client()` for
   the simulated caregiver and the judge. They can sit on different providers.

2. A client-side rate limiter. Groq's free tier is tight enough (8K tokens per
   minute per model) that an unthrottled batch run spends most of its time
   bouncing off 429s: a single agent-loop iteration with the therapy tools
   declared is already a few thousand tokens. The wrapper below paces requests
   against a 60-second sliding window *before* sending them, and still retries
   on 429 in case the server disagrees with the local accounting.

The wrapper is transparent: it exposes `.chat.completions.create(...)` with the
same signature as the raw SDK, so `chat.client.chat.completions.create(...)`
keeps working unchanged (`model` may be omitted — the client's own model is
used). Quotas are tracked per `provider:model`, because that is how they are
counted and because two roles pointing at the same model share one budget.
"""

import json
import re
import threading
from collections import deque
from time import monotonic, sleep

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from config_loader import MAIN_LLM, SIM_LLM, LLMConfig
from utils import get_current_logger

logger = get_current_logger()

_WINDOW = 60.0  # seconds; both RPM and TPM are per-minute
# Starting char→token ratio, used only until the first responses come back.
# Deliberately pessimistic: over-estimating costs throughput, under-estimating
# costs a 429. The real ratio for these prompts is far higher (the tool schemas
# and the therapy JSON tokenise very densely), so it is measured per model from
# the reported usage and takes over after a few requests — at 8K TPM, an
# estimate twice the true cost halves the number of requests that fit a minute.
_CHARS_PER_TOKEN = 3.4
_CALIBRATION_MIN_SAMPLES = 3
_PROMPT_SAFETY_MARGIN = 1.10  # applied to the calibrated prompt estimate
_MIN_COMPLETION_RESERVE = 300


class DailyQuotaExceeded(RuntimeError):
    """Raised when the provider's per-day quota is exhausted."""


class RequestTooLarge(RuntimeError):
    """Raised when a single request cannot fit the per-minute token budget."""


def _payload_chars(kwargs: dict) -> int:
    """Serialised size of everything that will be billed as prompt."""
    payload = 0
    for key in ("messages", "tools"):
        value = kwargs.get(key)
        if value:
            payload += len(json.dumps(value, ensure_ascii=False, default=str))
    return payload


def _parse_retry_after(error: Exception) -> float | None:
    """Seconds to wait, from the Retry-After header or the error message."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        raw = headers.get("retry-after")
        if raw:
            try:
                return float(raw)
            except ValueError:
                pass

    # Groq puts the wait in the message: "Please try again in 7m32.5s"
    match = re.search(
        r"try again in\s+(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", str(error), re.IGNORECASE
    )
    if match and any(match.groups()):
        hours, minutes, seconds = match.groups()
        return float(hours or 0) * 3600 + float(minutes or 0) * 60 + float(seconds or 0)
    return None


class _ModelRateLimiter:
    """
    Sliding-window limiter for one `provider:model` quota: RPM/TPM pacing plus
    RPD/TPD counters. Both roles share one instance when they point at the same
    provider and model, because the provider counts them against one budget.
    """

    def __init__(self, quota_key: str, config: LLMConfig):
        self.key = quota_key
        self.model = config.model
        self.rpm = config.rpm
        self.tpm = config.tpm
        self.rpd = config.rpd
        self.tpd = config.tpd
        self.completion_reserve = config.completion_reserve
        self._lock = threading.Lock()
        self._requests: deque[float] = deque()  # timestamps
        self._tokens: deque[list] = deque()  # [timestamp, tokens] – tokens is mutable
        self.day_requests = 0
        self.day_tokens = 0
        # Calibration of the estimator against what the provider actually charges
        self._seen_chars = 0
        self._seen_prompt_tokens = 0
        self._samples = 0
        self._max_completion = 0

    def estimate(self, kwargs: dict) -> int:
        """Estimated prompt + completion tokens, calibrated on past responses."""
        chars = _payload_chars(kwargs)
        explicit = kwargs.get("max_completion_tokens") or kwargs.get("max_tokens")
        with self._lock:
            calibrated = self._samples >= _CALIBRATION_MIN_SAMPLES and self._seen_prompt_tokens
            if calibrated:
                ratio = self._seen_chars / self._seen_prompt_tokens
                prompt = int(chars / ratio * _PROMPT_SAFETY_MARGIN)
                reserve = int(max(self._max_completion * 1.25, _MIN_COMPLETION_RESERVE))
            else:
                prompt = int(chars / _CHARS_PER_TOKEN)
                reserve = self.completion_reserve
        return prompt + int(explicit or reserve)

    def calibrate(self, chars: int, usage) -> None:
        """Feed the reported usage back into the estimator."""
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None) or 0
        if not prompt_tokens or not chars:
            return
        with self._lock:
            self._seen_chars += chars
            self._seen_prompt_tokens += prompt_tokens
            self._samples += 1
            self._max_completion = max(self._max_completion, completion_tokens)

    def _prune(self, now: float) -> None:
        cutoff = now - _WINDOW
        while self._requests and self._requests[0] <= cutoff:
            self._requests.popleft()
        while self._tokens and self._tokens[0][0] <= cutoff:
            self._tokens.popleft()

    def _wait_needed(self, estimated: int) -> float:
        """Seconds to wait before a request of `estimated` tokens may be sent."""
        now = monotonic()
        self._prune(now)
        waits = [0.0]

        if self.rpm and len(self._requests) >= self.rpm:
            waits.append(_WINDOW - (now - self._requests[0]))

        if self.tpm:
            used = sum(entry[1] for entry in self._tokens)
            if used + estimated > self.tpm:
                if estimated > self.tpm:
                    # The request alone cannot fit the per-minute budget. Nothing
                    # to do but drain the window and let the provider decide.
                    if self._tokens:
                        waits.append(_WINDOW - (now - self._tokens[-1][0]))
                else:
                    freed = 0
                    needed = used + estimated - self.tpm
                    for timestamp, tokens in self._tokens:
                        freed += tokens
                        if freed >= needed:
                            waits.append(_WINDOW - (now - timestamp))
                            break
        return max(waits)

    def acquire(self, estimated: int) -> list:
        """Block until the request may be sent; return its mutable ledger entry."""
        if self.tpm and estimated > self.tpm:
            logger.warning(
                f"[RATE_LIMIT][{self.key}] Request estimated at {estimated} tokens exceeds "
                f"the {self.tpm} TPM budget on its own – sending it anyway, expect a 429"
            )
        while True:
            with self._lock:
                wait = self._wait_needed(estimated)
                if wait <= 0:
                    now = monotonic()
                    entry = [now, estimated]
                    self._requests.append(now)
                    self._tokens.append(entry)
                    self.day_requests += 1
                    self.day_tokens += estimated
                    self._check_daily()
                    return entry
            if wait >= 0.5:
                logger.info(
                    f"[RATE_LIMIT][{self.key}] Pausing {wait:.1f}s to stay within "
                    f"{self.rpm} RPM / {self.tpm} TPM"
                )
            sleep(wait + 0.05)

    def commit(self, entry: list, actual: int | None) -> None:
        """Replace the estimate with the usage the provider actually reported."""
        if not actual:
            return
        with self._lock:
            self.day_tokens += actual - entry[1]
            entry[1] = actual

    def release(self, entry: list) -> None:
        """Drop a request that never reached the provider (connection error)."""
        with self._lock:
            self.day_tokens -= entry[1]
            self.day_requests -= 1
            entry[1] = 0
            # Zeroing the ledger entry frees the tokens but left the timestamp in
            # the request window, so an attempt that never reached the provider
            # still held an RPM slot for a whole minute. acquire() appends the
            # same value to both deques.
            try:
                self._requests.remove(entry[0])
            except ValueError:
                pass  # already pruned out of the 60s window

    def _check_daily(self) -> None:
        if self.rpd and self.day_requests > self.rpd:
            raise DailyQuotaExceeded(
                f"Daily request quota reached for '{self.key}': {self.day_requests} "
                f"requests in this run against a limit of {self.rpd} RPD."
            )
        if self.tpd and self.day_tokens > self.tpd:
            raise DailyQuotaExceeded(
                f"Daily token quota reached for '{self.key}': ~{self.day_tokens} tokens "
                f"in this run against a limit of {self.tpd} TPD."
            )

    def usage(self) -> dict:
        return {
            "quota": self.key,
            "model": self.model,
            "requests": self.day_requests,
            "tokens": self.day_tokens,
            "rpd_limit": self.rpd,
            "tpd_limit": self.tpd,
        }


_limiters: dict[str, _ModelRateLimiter] = {}
_limiters_lock = threading.Lock()


def _limiter_for(config: LLMConfig, model: str) -> _ModelRateLimiter:
    key = f"{config.provider}:{model}"
    with _limiters_lock:
        if key not in _limiters:
            _limiters[key] = _ModelRateLimiter(key, config)
        limiter = _limiters[key]
        if (limiter.rpm, limiter.tpm, limiter.rpd, limiter.tpd) != (
            config.rpm,
            config.tpm,
            config.rpd,
            config.tpd,
        ):
            # Both roles landed on the same provider:model, which is one quota at
            # the provider, but were configured with different limits. The first
            # one wins; say so rather than silently pacing against the wrong ones.
            logger.warning(
                f"[RATE_LIMIT][{key}] {config.role} declares different limits "
                f"({config.rpm}/{config.tpm}/{config.rpd}/{config.tpd}) for a quota already "
                f"paced at {limiter.rpm}/{limiter.tpm}/{limiter.rpd}/{limiter.tpd} "
                f"(RPM/TPM/RPD/TPD) – keeping the latter for both roles"
            )
        return limiter


def usage_report() -> list[dict]:
    """
    Per-quota request/token counters accumulated by this process, plus the
    parameters that turned out to be unsupported and were dropped.

    `dropped_params` is here because the session header is written before the
    first request and can only state what was *configured*. When a model then
    refuses a parameter, it is dropped for the rest of the process and the run
    proceeds under settings the log still reports as being in force — measured on
    gpt-5.4-mini, which rejects `reasoning_effort` alongside tool declarations, so
    every batch so far ran with no reasoning effort while the header announced
    `reasoning_effort=low`. A report that misdescribes the configuration it was
    produced under silently contaminates anything concluded from it, so what was
    actually sent has to be recorded at the end, when it is known.
    """
    # Snapshot first, so the two locks are never held at the same time.
    with _unsupported_lock:
        dropped = {key: sorted(names) for key, names in _unsupported.items()}
    with _limiters_lock:
        report = [limiter.usage() for limiter in _limiters.values()]
    for entry in report:
        entry["dropped_params"] = dropped.get(entry["quota"], [])
    return report


# Parameters the project adds by itself and can therefore drop again when a
# model refuses them. `reasoning_effort` is sent on every agent-loop call: the
# gpt-oss family accepts it everywhere, plain chat models answer 400. Rather than
# making the operator match the knob to the model when switching providers, the
# offending parameter is dropped once per quota and the call is retried.
#
# `temperature` and `seed` are deliberately NOT in here. Dropping a rejected
# sampling parameter would let a batch finish while sampling differently from the
# configuration it reports, and the results would be attributed to settings that
# were never in force — the one failure mode a measurement harness cannot absorb
# quietly. A provider that refuses them (OpenAI's hosted reasoning models refuse a
# non-default temperature alongside reasoning_effort) must fail the run instead.
_DROPPABLE_PARAMS = ("reasoning_effort",)
_unsupported: dict[str, set[str]] = {}
_unsupported_lock = threading.Lock()


def _drop_unsupported(quota_key: str, kwargs: dict) -> None:
    with _unsupported_lock:
        for name in _unsupported.get(quota_key, ()):
            kwargs.pop(name, None)


def _record_unsupported(quota_key: str, error: Exception, kwargs: dict) -> str | None:
    """If the 400 blames a droppable parameter, remember it and report which."""
    text = str(error).lower()
    for name in _DROPPABLE_PARAMS:
        if name in kwargs and name in text:
            with _unsupported_lock:
                _unsupported.setdefault(quota_key, set()).add(name)
            return name
    return None


class _Completions:
    def __init__(self, inner, config: LLMConfig):
        self._inner = inner
        self._config = config

    def create(self, **kwargs):
        config = self._config
        # The client knows its own model and reasoning effort; call sites may
        # still override either. Both roles get this, so SIM_REASONING_EFFORT
        # reaches the caregiver and the judge without every call site repeating it.
        kwargs.setdefault("model", config.model)
        if config.reasoning_effort:
            kwargs.setdefault("reasoning_effort", config.reasoning_effort)
        # Both are omitted entirely when unset, so the provider's own default
        # applies and nothing changes for a config that does not mention them.
        if config.temperature is not None:
            kwargs.setdefault("temperature", config.temperature)
        if config.seed is not None:
            kwargs.setdefault("seed", config.seed)
        model = kwargs["model"]
        quota_key = f"{config.provider}:{model}"
        _drop_unsupported(quota_key, kwargs)

        # Every call goes through a limiter, including the unthrottled providers:
        # with all four limits at 0 it paces nothing, but it still counts requests
        # and tokens. Without it a run reports only the throttled half of itself.
        limiter = _limiter_for(config, model)
        chars = _payload_chars(kwargs)
        estimated = limiter.estimate(kwargs)

        for attempt in range(config.max_retries + 1):
            entry = limiter.acquire(estimated)
            try:
                response = self._inner.create(**kwargs)
            except RateLimitError as e:
                limiter.commit(entry, estimated)  # the attempt still consumed quota
                wait = _parse_retry_after(e) or min(2**attempt * 5, 60)
                if wait > config.max_retry_wait:
                    raise DailyQuotaExceeded(
                        f"Provider asked to wait {wait:.0f}s on '{limiter.key}' – that is a "
                        f"daily quota, not a per-minute one. Aborting instead of sleeping.\n{e}"
                    ) from e
                if attempt == config.max_retries:
                    raise
                logger.warning(
                    f"[RATE_LIMIT][{limiter.key}] 429 from provider, retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/{config.max_retries})"
                )
                sleep(wait + 0.5)
            except BadRequestError as e:
                dropped = _record_unsupported(limiter.key, e, kwargs)
                if not dropped or attempt == config.max_retries:
                    limiter.commit(entry, estimated)
                    raise
                limiter.release(entry)
                logger.warning(
                    f"[LLM][{limiter.key}] model rejected '{dropped}' – dropping it and retrying"
                )
                kwargs.pop(dropped, None)
            except (APIConnectionError, APITimeoutError) as e:
                limiter.release(entry)
                if attempt == config.max_retries:
                    raise
                wait = min(2**attempt * 2, 30)
                logger.warning(f"[LLM][{limiter.key}] {type(e).__name__}, retrying in {wait}s")
                sleep(wait)
            except APIStatusError as e:
                if e.status_code == 413:
                    # "Request too large … on tokens per minute": the prompt alone
                    # is over the per-minute budget, so no amount of waiting helps.
                    limiter.release(entry)
                    raise RequestTooLarge(
                        f"A single request to '{limiter.key}' (~{estimated} estimated tokens) "
                        f"does not fit the {config.tpm} tokens-per-minute budget of this "
                        f"account, so it can never succeed: the conversation has simply grown "
                        f"past what the tier allows. Shorten the run (fewer turns) or raise "
                        f"the limit on the provider side.\n{e}"
                    ) from e
                if e.status_code < 500 or attempt == config.max_retries:
                    limiter.commit(entry, estimated)
                    raise
                limiter.release(entry)
                wait = min(2**attempt * 2, 30)
                logger.warning(f"[LLM][{limiter.key}] HTTP {e.status_code}, retrying in {wait}s")
                sleep(wait)
            else:
                usage = getattr(response, "usage", None)
                actual = getattr(usage, "total_tokens", None) if usage else None
                limiter.commit(entry, actual)
                if usage:
                    limiter.calibrate(chars, usage)
                logger.debug(
                    f"[LLM][{limiter.key}] tokens: estimated={estimated} actual={actual} | "
                    f"run total ~{limiter.day_tokens}"
                )
                return response

        raise RuntimeError(f"Request to '{limiter.key}' failed after {config.max_retries} retries")


class _Chat:
    def __init__(self, inner, config: LLMConfig):
        self.completions = _Completions(inner.completions, config)


class RateLimitedClient:
    """Thin proxy around an OpenAI client, adding pacing and retries."""

    def __init__(self, client: OpenAI, config: LLMConfig):
        self._client = client
        self.config = config
        self.model = config.model
        self.chat = _Chat(client.chat, config)

    def __getattr__(self, name):
        # Anything we do not wrap (models, embeddings, …) passes straight through.
        return getattr(self._client, name)


def make_client(config: LLMConfig) -> RateLimitedClient:
    """Return a rate-limited, OpenAI-compatible client for one role's backend."""
    if config.provider in ("openai", "groq") and not config.api_key:
        raise ValueError(
            f"{config.describe()}: {config.provider.upper()}_API_KEY is not set in .env"
        )
    # max_retries=0 because the retry policy (and its quota accounting) lives in
    # _Completions.create above; base_url is None only for OpenAI's own default.
    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout,
        max_retries=0,
    )
    logger.debug(
        f"[LLM] Client ready – {config.describe()} @ {config.base_url or 'openai default'}"
    )
    return RateLimitedClient(client, config)


def make_main_client() -> RateLimitedClient:
    """Client for the system under test: therapy manager, checker, extractors."""
    return make_client(MAIN_LLM)


def make_sim_client() -> RateLimitedClient:
    """Client for the simulation agents of the harness: caregiver and judge."""
    return make_client(SIM_LLM)
