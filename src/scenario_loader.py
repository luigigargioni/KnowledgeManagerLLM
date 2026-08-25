# scenario_loader.py

import json
import logging
import re

from config_loader import SCENARIOS_DIR, THERAPY_FILE
from tools import find_conflicting_activity
from utils import hhmm_to_minutes, minutes_to_hhmm

logger = logging.getLogger("knowledge_manager")

DAYS_MAP = {
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
    7: "Sunday",
}


def load_scenario(scenario_id: int) -> dict:
    """
    Read the scenario.json file from the scenario folder.

    Reports — but does not raise on — a therapy the scheduler itself would refuse
    to build; see validate_scenario_therapy for why the two are kept apart.
    """
    path = SCENARIOS_DIR / f"{str(scenario_id)}.json"
    if not path.exists():
        raise FileNotFoundError(f"Scenario {scenario_id} not found: {path}")
    scenario = json.loads(path.read_text(encoding="utf-8"))
    for problem in validate_scenario_therapy(scenario):
        logger.warning(f"[SCENARIO {scenario_id}] invalid seed - {problem}")
    return scenario


def validate_scenario_therapy(scenario: dict) -> list[str]:
    """
    Check a scenario's initial therapy against the invariants tools.py enforces on
    every write, and return one line per violation (empty list = clean).

    A scenario's therapy is installed into therapy.json verbatim by
    `install_scenario_therapy`, bypassing the tools — so nothing has ever checked
    that the starting point is a state the tools would accept. On the 2026-08-24
    batch of 100 scenarios, 15 started from one they would not:

        Breakfast 08:00 +20min       -> ends 08:20
        <medication> 08:10 +5min  dependencies=[br_001]

    which both violates the ordering rule (the dependency ends *after* the
    dependent starts) and overlaps it. Harmless while nothing touches that chain —
    11 of the 15 still graded "completed" — and decisive when something does: it
    is where 26, 68, 38 and 48 all failed, the assistant made to repair an
    inconsistency it did not create and then graded on the result.

    The same three rules as the write path, in the same order, so a violation here
    names what would be refused there:
      - every declared dependency exists;
      - every dependency ends at or before the dependent activity starts
        (in tools.py, dep_end > start is a temporal_ordering error);
      - no two activities overlap in time and day-of-week and validity range —
        `tools.find_conflicting_activity` is reused rather than reimplemented, so
        the seed check and the write check cannot drift apart.

    All 15 have since been repaired — 26, 68 and 88 by moving the whole morning
    or evening block, the other 12 by moving the meal earlier so it ends five
    minutes before the medication starts. That direction was chosen because it
    survives either answer to the question still open about the ordering rule
    ("after breakfast": after it starts, or after it ends?): a dependency that
    ends before the dependent begins satisfies both readings.

    Returns problems instead of raising, and the caller decides: `load_scenario`
    logs them, `test.py` aborts on them unless --allow-invalid-scenarios is set.
    Aborting is the right default now that the dataset is clean — the check earns
    its keep by catching the next scenario written by hand, not by re-reporting
    known damage.
    """
    problems: list[str] = []
    activities = scenario.get("activities") or []

    by_id: dict[str, dict] = {}
    for act in activities:
        activity_id = act.get("activity_id")
        if activity_id in by_id:
            problems.append(f"duplicate activity_id '{activity_id}'")
        by_id[activity_id] = act

    for act in activities:
        for dep_id in act.get("dependencies") or []:
            dep = by_id.get(dep_id)
            if dep is None:
                problems.append(
                    f"'{act['name']}' depends on '{dep_id}', which is not in the schedule"
                )
                continue
            dep_end = hhmm_to_minutes(dep["time"]) + dep["duration_minutes"]
            if dep_end > hhmm_to_minutes(act["time"]):
                problems.append(
                    f"temporal ordering: '{act['name']}' starts at {act['time']} but "
                    f"depends on '{dep['name']}', which ends at {minutes_to_hhmm(dep_end)}"
                )

    reported: set[tuple[str, str]] = set()
    for act in activities:
        others = [other for other in activities if other is not act]
        clash = find_conflicting_activity(act, others)
        if clash is None:
            continue
        pair = tuple(sorted((act["name"], clash["name"])))
        if pair in reported:
            continue
        reported.add(pair)
        problems.append(f"overlap: '{pair[0]}' and '{pair[1]}' run at the same time")

    return problems


def install_scenario_therapy(scenario: dict) -> None:
    """
    Overwrite therapy.json with the scenario's therapy.
    """
    THERAPY_FILE.write_text(
        json.dumps(scenario, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# Clauses that tell the caregiver what the assistant is expected to do (or not do).
# "First ask the assistant …" is deliberately NOT included: that one is an
# instruction to act, it has to be known up front and it reveals no answer.
#
# The third alternative catches the same conditional written without naming the
# assistant ("If a conflict is detected", "If an error is reported"). Scenario 10
# was phrased that way, matched nothing, and therefore handed the caregiver the
# whole answer — title, context and all — in its opening message.
_CONDITIONAL_RE = re.compile(
    r"\b(?:"
    r"If\s+the\s+assistant"
    r"|Verify\s+that\s+the\s+assistant"
    r"|If\s+(?:an?|the|any)\s+[\w\s'-]{0,40}?\bis\s+"
    r"(?:detected|reported|found|flagged|raised|returned|triggered)"
    r")\b",
    re.IGNORECASE,
)

# The "# Scenario N – …" heading is harness metadata: no caregiver could know it,
# and it routinely states the expected outcome ("Add *Safe* Metformin Medication",
# "Add Health Checkup - *No History Warning*", "Complex Multi-Step with *Conflict
# Resolution*"). In scenarios with a conditional clause it was already withheld
# together with the rest of the preamble; the 36 scenarios without one were
# handing it over verbatim. The judge still receives the untouched script, so
# nothing is lost for grading.
_TITLE_RE = re.compile(r"^[ \t]*#[ \t]*Scenario\b.*$", re.IGNORECASE | re.MULTILINE)


def _strip_title(text: str) -> str:
    """Remove the scenario heading from text meant for the caregiver."""
    return _TITLE_RE.sub("", text or "").strip()


def split_objectives(objectives: str) -> tuple[str, str]:
    """
    Split a scenario's objectives into what the caregiver may know up front and
    what must stay hidden until the assistant has had its turn.

    Many scenarios exist to test whether the assistant *itself* raises something —
    a history warning, a scheduling conflict, a blocking dependency — and encode
    the expected reaction as "If the assistant warns about X, acknowledge…".
    Handing that to the caregiver together with the initial request tells it the
    answer: it then mentions X in its own first message, or reacts to a warning
    that never came. Either way the behaviour under test is never observed, while
    the objective still looks satisfied.

    Returns (initial, deferred):
      - initial:  the bare imperative requests, with no title and no context
      - deferred: the context and the conditional clauses, to be revealed only
                  once the assistant has actually raised the point (the gate is
                  in test.py: a blocking signal from Chat.turn_issues, confirmed
                  by utils.assistant_handed_back). Wording alone never
                  delivers: a branch whose trigger is a clinical judgement
                  rather than a refused write is not delivered at all, and
                  is recorded as branch_exercised=False.

    Scenarios without a conditional clause keep their context, minus the title,
    and get an empty deferred part.
    """
    if not _CONDITIONAL_RE.search(objectives or ""):
        return _strip_title(objectives), ""

    lines = (objectives or "").splitlines()
    try:
        start = next(
            i for i, line in enumerate(lines) if line.strip().lower().startswith("## objectives")
        )
    except StopIteration:
        # Unexpected layout: withholding nothing is safer than mangling the script.
        return _strip_title(objectives), ""

    preamble = _strip_title("\n".join(lines[:start]))
    objective_lines = lines[start + 1 :]

    kept, withheld = [], []
    for line in objective_lines:
        match = _CONDITIONAL_RE.search(line)
        if match:
            kept.append(line[: match.start()].rstrip())
            withheld.append(line[match.start() :].strip())
        else:
            kept.append(line)

    initial = "## Objectives\n" + "\n".join(kept).strip()

    deferred_parts = []
    if preamble:
        deferred_parts.append(preamble)
    if withheld:
        deferred_parts.append(
            "## How to react if the assistant raises it\n"
            + "\n".join(f"- {clause}" for clause in withheld)
        )
    return initial, "\n\n".join(deferred_parts)


_NUMBERED_OBJECTIVE_RE = re.compile(r"^\s*\d+\.\s", re.MULTILINE)


def count_objectives(objectives: str) -> int:
    """
    How many numbered objectives the script asks the caregiver to carry out.

    Counted from the script rather than from the judge's reply, so that the two
    can be compared: the caregiver is a simulated user and does sometimes drop
    an objective and end the conversation without ever raising it. That is not a
    harsh grade, it is a test that never ran — the system under test was never
    asked — and it is indistinguishable from a real failure unless the expected
    count is recorded next to the outcome.
    """
    body = (objectives or "").split("## Objectives")[-1]
    return len(_NUMBERED_OBJECTIVE_RE.findall(body))


def therapy_to_natural_language(scenario: dict) -> str:
    """
    Convert the scenario's therapy into descriptive text to inject into the CaregiverAgent's context.
    """
    lines = []

    # Patient data
    name = scenario.get("patient_full_name", "Unknown")
    age = scenario.get("age", "N/A")
    birth = scenario.get("birth_date", "")
    if birth:
        try:
            from datetime import datetime

            birth = datetime.fromisoformat(birth).strftime("%d/%m/%Y")
        except ValueError:
            pass

    lines.append("## Patient")
    lines.append(f"- Name: {name}")
    lines.append(f"- Date of birth: {birth} (age {age})")

    conditions = scenario.get("medical_conditions", [])
    if conditions:
        lines.append(f"- Medical conditions: {', '.join(conditions)}")
    else:
        lines.append("- Medical conditions: none reported")

    # Current activities
    activities = scenario.get("activities", [])
    lines.append("\n## Current therapy activities")
    if not activities:
        lines.append("- No activities currently scheduled.")
    else:
        for act in activities:
            days = ", ".join(DAYS_MAP[d] for d in act.get("day_of_week", []))
            line = (
                f"- **{act['name']}** ({act.get('category', 'N/A')}): "
                f"{act['time']}, {act['duration_minutes']} min, {days}"
            )
            if act.get("valid_from") or act.get("valid_until"):
                line += f" — valid: {act.get('valid_from', '…')} → {act.get('valid_until', '…')}"
            if act.get("description"):
                line += f"\n  {act['description']}"
            lines.append(line)

    # Expired activities
    expired = scenario.get("expired_activities", [])
    if expired:
        lines.append("\n## Expired activities")
        for act in expired:
            days = ", ".join(DAYS_MAP[d] for d in act.get("day_of_week", []))
            lines.append(
                f"- **{act['name']}**: {act['time']}, {act['duration_minutes']} min, "
                f"{days} — expired on {act.get('valid_until', 'N/A')}"
            )

    return "\n".join(lines)
