# scenario_loader.py

import json
import re

from config_loader import SCENARIOS_DIR, THERAPY_FILE

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
    """Read the scenario.json file from the scenario folder."""
    path = SCENARIOS_DIR / f"{str(scenario_id)}.json"
    if not path.exists():
        raise FileNotFoundError(f"Scenario {scenario_id} not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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
_CONDITIONAL_RE = re.compile(r"\b(?:If the assistant|Verify that the assistant)\b", re.IGNORECASE)


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
      - deferred: the title, the context and the conditional clauses, to be
                  revealed only after the assistant has replied

    Scenarios without a conditional clause are returned unchanged, with an empty
    deferred part.
    """
    if not _CONDITIONAL_RE.search(objectives or ""):
        return objectives, ""

    lines = (objectives or "").splitlines()
    try:
        start = next(
            i for i, line in enumerate(lines) if line.strip().lower().startswith("## objectives")
        )
    except StopIteration:
        # Unexpected layout: withholding nothing is safer than mangling the script.
        return objectives, ""

    preamble = "\n".join(lines[:start]).strip()
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
