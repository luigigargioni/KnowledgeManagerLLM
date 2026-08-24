# therapy_diff.py
"""
Deterministic comparison between the therapy a scenario started from and the
therapy it ended with.

The JudgeAgent used to decide whether an objective was met by reading the
conversation transcript. That is not a reliable record of what happened: the
assistant sometimes reports an activity as added, updated or removed when no such
change ever reached therapy.json, and a transcript containing a well-formatted
confirmation is indistinguishable from a real success. Computing the change set
in code gives the judge — and any reviewer — the one account that cannot be
fabricated.
"""

# Fields whose change counts as a modification of an existing activity.
COMPARED_FIELDS = (
    "name",
    "time",
    "duration_minutes",
    "day_of_week",
    "category",
    "dependencies",
    "valid_from",
    "valid_until",
    "description",
)

DAYS_MAP = {
    1: "Mon",
    2: "Tue",
    3: "Wed",
    4: "Thu",
    5: "Fri",
    6: "Sat",
    7: "Sun",
}


def _by_id(activities: list[dict]) -> dict[str, dict]:
    return {act.get("activity_id"): act for act in (activities or [])}


def _format_value(field: str, value) -> str:
    if field == "day_of_week" and isinstance(value, list):
        return "/".join(DAYS_MAP.get(d, str(d)) for d in value) or "none"
    if field == "dependencies":
        return ", ".join(value) if value else "none"
    if value in (None, ""):
        return "none"
    return str(value)


def _describe(activity: dict) -> str:
    # The description is printed because it is the one field the scheduler never
    # reads. An assistant asked to place an activity "after X" sometimes writes
    # that into the description instead of dependencies — the ordering is then
    # recorded nowhere it can be enforced, and the diff used to render such an
    # activity identically to one whose constraint was honestly dropped
    # ("depends on: none"). Showing it lets the judge tell the two apart.
    described = (
        f"'{activity.get('name')}' [{activity.get('activity_id')}] "
        f"at {activity.get('time')} for {activity.get('duration_minutes')}min "
        f"on {_format_value('day_of_week', activity.get('day_of_week'))} "
        f"(category: {activity.get('category')}, "
        f"depends on: {_format_value('dependencies', activity.get('dependencies'))})"
    )
    description = (activity.get("description") or "").strip()
    return f'{described} description: "{description}"' if description else described


def diff_therapies(initial: dict, final: dict) -> dict:
    """
    Compare two therapy states and return the change set.

    Activities are matched on activity_id, which is assigned server-side and
    never reused, so the matching is exact.
    """
    initial_active = _by_id(initial.get("activities"))
    final_active = _by_id(final.get("activities"))
    final_expired = _by_id(final.get("expired_activities"))

    added = [act for act_id, act in final_active.items() if act_id not in initial_active]

    removed, expired = [], []
    for act_id, act in initial_active.items():
        if act_id in final_active:
            continue
        (expired if act_id in final_expired else removed).append(act)

    modified = []
    for act_id, final_act in final_active.items():
        initial_act = initial_active.get(act_id)
        if not initial_act:
            continue
        changes = [
            {
                "field": field,
                "before": initial_act.get(field),
                "after": final_act.get(field),
            }
            for field in COMPARED_FIELDS
            if initial_act.get(field) != final_act.get(field)
        ]
        if changes:
            modified.append({"activity": final_act, "changes": changes})

    return {
        "added": added,
        "removed": removed,
        "expired": expired,
        "modified": modified,
        "unchanged_count": len(final_active) - len(added) - len(modified),
        "has_changes": bool(added or removed or expired or modified),
    }


def summarise_touched(diff: dict) -> str:
    """
    One compact line naming every activity the conversation touched.

    Deliberately *not* a detector of unrequested changes. Matching what changed
    against what the caregiver asked for would mean matching activity names
    against the words of someone who speaks like a person ("the walk after
    lunch", never "Evening walk"), which produces false positives on the one
    signal that has to be trusted. This lists the facts instead and leaves the
    judgement to whoever reads the results — which is how these runs are
    reviewed anyway.

    Reading it is how a silent change gets noticed: an assistant once moved a
    patient's lunch by 45 minutes to fit a medication around it, nobody had
    asked, and the run was graded a full success because every scripted
    objective had also been met.
    """
    parts: list[str] = []
    parts += [f"+ {a.get('name')}" for a in diff.get("added", [])]
    parts += [f"- {a.get('name')}" for a in diff.get("removed", [])]
    parts += [f"~ {a.get('name')} (expired)" for a in diff.get("expired", [])]
    for entry in diff.get("modified", []):
        fields = ", ".join(c["field"] for c in entry["changes"])
        parts.append(f"~ {entry['activity'].get('name')} ({fields})")
    return " | ".join(parts) if parts else "no change"


def render_diff(diff: dict) -> str:
    """Render a change set as the plain-text block handed to the JudgeAgent."""
    lines: list[str] = []

    if not diff["has_changes"]:
        return (
            "NO CHANGE was applied to the therapy during this conversation. "
            "Every activity is exactly as it was before it started."
        )

    if diff["added"]:
        lines.append("ADDED activities:")
        lines += [f"  + {_describe(act)}" for act in diff["added"]]

    if diff["removed"]:
        lines.append("REMOVED activities:")
        lines += [f"  - {_describe(act)}" for act in diff["removed"]]

    if diff["expired"]:
        lines.append("MOVED TO EXPIRED:")
        lines += [f"  ~ {_describe(act)}" for act in diff["expired"]]

    if diff["modified"]:
        lines.append("MODIFIED activities:")
        for entry in diff["modified"]:
            activity = entry["activity"]
            lines.append(f"  * '{activity.get('name')}' [{activity.get('activity_id')}]")
            for change in entry["changes"]:
                field = change["field"]
                lines.append(
                    f"      {field}: "
                    f"{_format_value(field, change['before'])} -> "
                    f"{_format_value(field, change['after'])}"
                )

    lines.append(f"UNCHANGED activities: {diff['unchanged_count']}")
    return "\n".join(lines)
