# scenario_loader.py

import json

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
