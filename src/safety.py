# safety.py
"""
The checker's verdict, typed — and therefore usable by code.

Why this module exists
----------------------
Before it, the checker answered with `{"activity_name":…,"check_result":[…]}`
where `check_result` was a flat list of prose findings. The format held (64
replies out of 64 parsed), but every finding had the same shape, so an absolute
contraindication and a passing remark about meal timing were indistinguishable.
Two consequences, both measured on the 50-scenario batch of 2026-08-24:

- The delivery gate in `test.py` fires only on a *blocked write*. A
  contraindication does not block anything — `add_therapy_activity` succeeds —
  so of the 18 conditional scenarios whose trigger is a safety finding, the gate
  delivered the caregiver its reaction clause 0 times on the intended cause. The
  6 that did deliver all fired on an unrelated scheduling or dependency block.
- Whether the check happened at all was left to the prompt. On 6 scenarios of
  the same class (add a contraindicated medication) the manager delegated to the
  checker 4 times and skipped it twice, writing the contraindicated drug with no
  check whatsoever (scenarios 32 and 37).

Typing the severity fixes both. `blocking` and `caution` become blocking causes
like a scheduling conflict, which is what the gate already understands; and the
check itself moves from the prompt into `Chat.execute_tool`, the same way
scheduling was moved out of the model's hands into `tools.py`.

The three levels are cut at "who has to decide":

- `blocking` — the activity must not exist as requested (absolute
  contraindication). Nobody in this conversation can authorise it; the assistant
  has to propose something else or ask. This one refuses the write.
- `caution` — a real clinical risk the *caregiver* has to weigh. It does NOT
  refuse the write: reporting it and asking is the assistant's duty, and the
  manager's prompt carries it. It does raise a signal, which is what the delivery
  gate in test.py needs.
- `remark` — an observation with no decision attached ("12:45 is around lunch,
  so it is not fasting"). Never blocks, never signals. This level is the whole
  reason the checker's verdict is usable now and was not before: it is where
  everything that used to produce false positives goes.

`caution` did refuse the write at first, once per activity, releasing on the
caregiver's next turn so the decision demonstrably reached them. It was measured
out on gpt-oss-20b at low reasoning effort: on scenarios 3, 13 and 14 the model
read the refusal, reported it, and never called the tool again — the caregiver
agreed and nothing was written. It had made the branch more *observable* and the
outcome worse, which is the wrong trade for a system whose job is to apply the
change; and refusing a legal write is the assistant taking a decision the prompt
explicitly reserves for the caregiver. Do not reintroduce it: if the assistant
writes without asking, that is a prompt failure to measure, not a lock to add.

Fail-open on format, loudly
---------------------------
A reply whose verdict cannot be parsed, or whose findings carry no severity, is
treated as *checked with no finding* and logged as a warning, and the count is
carried into the report (`Chat.safety_verdicts_unparsed`). Failing closed would
be the right default for a clinical system; here it would deadlock every
scenario the moment a small model dropped the format, and attribute the deadlock
to the behaviour under test. The choice is therefore to keep the run going and
make the degradation visible in the results, never silent.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# Severity levels, ordered from harmless to absolute.
SEVERITY_REMARK = "remark"
SEVERITY_CAUTION = "caution"
SEVERITY_BLOCKING = "blocking"

_SEVERITY_ORDER = {SEVERITY_REMARK: 0, SEVERITY_CAUTION: 1, SEVERITY_BLOCKING: 2}

# Issue names attached to a refused write, in the same namespace as tools.py's
# ISSUE_* constants: they land in Chat.turn_issues and are what the delivery gate
# in test.py keys on.
ISSUE_SAFETY_BLOCKED = "safety_blocked"
ISSUE_SAFETY_CAUTION = "safety_caution"
ISSUE_SAFETY_CHECK_REQUIRED = "safety_check_required"

# Words that carry no identity, stripped before comparing the name the checker
# examined with the name being written. Dosages and units are the common noise:
# the checker is asked about "Aspirin 100 mg", the write says "Aspirin".
_NOISE_TOKENS = {
    "mg",
    "mcg",
    "ml",
    "units",
    "unit",
    "daily",
    "day",
    "every",
    "the",
    "and",
    "for",
    "after",
    "before",
    "dose",
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "activity",
    "medication",
    "medicine",
    "session",
    "check",
    "new",
}


def name_tokens(text: str) -> set[str]:
    """
    Identifying words of an activity name, lowercased, noise and digits out.

    Public because the caution latch in `Chat` has to recognise the same activity
    across turns, and it has to agree with `SafetyVerdict.concerns` on what
    "the same activity" means — two different notions of identity there would let
    a caution be latched under one name and released under another.
    """
    words = re.findall(r"[a-z]+", (text or "").lower())
    return {w for w in words if w not in _NOISE_TOKENS and len(w) > 2}


_tokens = name_tokens


class SafetyVerdict:
    """One checker answer, parsed. `severity` is the worst finding it carries."""

    def __init__(self, activity_name: str, findings: list[dict], raw: str = ""):
        self.activity_name = activity_name or ""
        self.findings = findings
        self.raw = raw

    @property
    def severity(self) -> str:
        if not self.findings:
            return SEVERITY_REMARK
        return max(
            (f.get("severity", SEVERITY_REMARK) for f in self.findings),
            key=lambda s: _SEVERITY_ORDER.get(s, 0),
        )

    def findings_at_least(self, level: str) -> list[dict]:
        floor = _SEVERITY_ORDER.get(level, 0)
        return [
            f
            for f in self.findings
            if _SEVERITY_ORDER.get(f.get("severity", SEVERITY_REMARK), 0) >= floor
        ]

    def summary(self, level: str = SEVERITY_CAUTION) -> str:
        """The findings at or above `level`, as one line per finding."""
        return "; ".join(
            (f.get("finding") or "").strip()
            for f in self.findings_at_least(level)
            if (f.get("finding") or "").strip()
        )

    def concerns(self, activity_name: str) -> bool:
        """
        Whether this verdict is about the activity now being written.

        A verdict authorises the activity it examined and nothing else: without
        this, a clean check on Paracetamol would let the next call write Warfarin.
        Deliberately lenient — the manager renames and re-doses freely between the
        check and the write — so one shared identifying word is enough, and an
        empty name on either side matches, but a verdict about a different drug
        does not.
        """
        mine, theirs = _tokens(self.activity_name), _tokens(activity_name)
        if not mine or not theirs:
            return True
        return bool(mine & theirs)

    def __repr__(self) -> str:
        return (
            f"SafetyVerdict({self.activity_name!r}, {self.severity}, {len(self.findings)} findings)"
        )


def _normalise_findings(raw_findings) -> tuple[list[dict], bool]:
    """
    Bring `check_result` to a list of {severity, finding}.

    Returns (findings, typed): `typed` is False when the checker answered with
    the old untyped shape — a list of bare strings, or objects with no severity.
    Those findings are kept, at `remark`, so they still reach the report and the
    caregiver through the assistant, but they do not block: guessing a severity
    from prose is the false-positive machine this module exists to retire.
    """
    findings: list[dict] = []
    typed = True

    if not isinstance(raw_findings, list):
        return [], False

    for entry in raw_findings:
        if isinstance(entry, str):
            text = entry.strip()
            if text:
                findings.append({"severity": SEVERITY_REMARK, "finding": text})
                typed = False
            continue
        if not isinstance(entry, dict):
            continue
        severity = str(entry.get("severity", "")).strip().lower()
        text = (entry.get("finding") or entry.get("issue") or entry.get("problem") or "").strip()
        if severity not in _SEVERITY_ORDER:
            severity = SEVERITY_REMARK
            typed = False
        findings.append({"severity": severity, "finding": text})

    return findings, typed


def parse_verdict(reply: str) -> tuple["SafetyVerdict | None", bool]:
    """
    Extract the verdict from a checker reply.

    Returns (verdict, typed). `verdict` is None when nothing parsable was found —
    the caller then treats the activity as unchecked-but-not-blocked and counts
    the miss (see the module docstring on failing open).

    The reply is prose with a JSON object in it, so the object is located by
    brace matching from each `{` backwards rather than by a regex over the whole
    text: the checker's own prose routinely contains braces, and the verdict is
    normally the last object in the reply.
    """
    if not isinstance(reply, str) or "check_result" not in reply:
        return None, False

    for match in reversed(list(re.finditer(r"\{", reply))):
        start = match.start()
        depth = 0
        for i in range(start, min(len(reply), start + 8000)):
            if reply[i] == "{":
                depth += 1
            elif reply[i] == "}":
                depth -= 1
                if depth == 0:
                    blob = reply[start : i + 1]
                    try:
                        payload = json.loads(blob)
                    except json.JSONDecodeError:
                        break
                    if not isinstance(payload, dict) or "check_result" not in payload:
                        break
                    findings, typed = _normalise_findings(payload.get("check_result"))
                    verdict = SafetyVerdict(
                        activity_name=str(payload.get("activity_name") or ""),
                        findings=findings,
                        raw=blob,
                    )
                    return verdict, typed

    return None, False
