import copy
import json
import logging
import os
import re
from datetime import datetime

from config_loader import THERAPY_FILE, THERAPY_SEED_FILE
from utils import hhmm_to_minutes, minutes_to_hhmm

CATEGORIES = [
    "medication",
    "outside_activity",
    "meal",
    "health_checkup",
    "therapy",
    "relaxation",
    "social_activity",
]

# activity_id prefix per category. IDs are assigned here rather than by the LLM:
# when the model invented them it produced ids outside the convention
# ("dessert_sugar_free_001", "br_short001" for a bed rest), collided with
# existing ones, and referenced ids for activities it had never created.
CATEGORY_ID_PREFIX = {
    "medication": "med",
    "outside_activity": "out",
    "meal": "ml",
    "health_checkup": "hc",
    "therapy": "thp",
    "relaxation": "rlx",
    "social_activity": "soc",
}


logger = logging.getLogger("knowledge_manager")


# Machine-readable cause attached to the tool results the caregiver is supposed
# to react to: a scheduling conflict, a missing or violated dependency, an
# activity that cannot be removed. Everything else returning status "error" is a
# validation slip the assistant fixes by itself (bad category, malformed time),
# and must NOT be mistaken for a decision handed back to the caregiver.
#
# It exists because `chat.Chat` reads it to know, deterministically, that the
# system raised something during a turn — see Chat._record_issue_signals and
# test.py, which gates the delivery of a scenario's withheld reaction clauses on
# it instead of guessing from the assistant's wording.
ISSUE_SCHEDULE_CONFLICT = "schedule_conflict"
ISSUE_MISSING_DEPENDENCY = "missing_dependency"
ISSUE_TEMPORAL_ORDERING = "temporal_ordering"
ISSUE_DEPENDENCY_BLOCKED = "dependency_blocked"


# Appended to the blocking failures of add_therapy_activity — the ones whose
# message names *another* activity (the dependency it must follow, or the one it
# overlaps with).
#
# Those messages say what is wrong but not what happened, and the difference
# matters: nothing was created, so the activity being added has no id yet. The
# model has twice been observed reading such a failure as "it exists, it just
# needs moving", and then calling update_therapy_activity with the only id in
# scope — the one belonging to the activity named in the message. In fifty
# scenarios that silently moved the patient's lunch by 45 minutes in one run and
# the patient's dinner in another, while the medication was never added at all.
# Both times the assistant reported the wrong outcome truthfully; both times the
# change stayed. Saying "nothing was created, do not call update" closes the gap
# at its source, the same way spelling out the duration rule stopped one-minute
# medications.
_NOTHING_CREATED = (
    " NOTE: no activity was created and no activity_id was assigned, so there is "
    "nothing to update. Call add_therapy_activity again with corrected values. Do "
    "NOT call update_therapy_activity — any activity named above is a different "
    "one, and moving it is not what was asked."
)


def _tool_json(payload) -> str:
    """
    Serialise a tool result for the model.

    Compact on purpose: indentation is whitespace the model does not read but
    the provider still bills, and every tool result stays in the conversation
    for the rest of the session. Measured on a real therapy document, dropping
    `indent=2` cuts the payload by ~35% of its tokens for identical content —
    which on a tight tokens-per-minute budget is throughput, not just cost.
    The file on disk keeps its indentation (see _save_therapy): that one is
    read by people.
    """
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


# ─── Vector DB reference (injected at startup via set_vector_db) ──────────────
_vector_db = None


def set_vector_db(vdb) -> None:
    """Inject the VectorDBManager instance used by all tools."""
    global _vector_db
    _vector_db = vdb
    logger.info("[TOOLS] VectorDBManager injected")


def _get_patient_id() -> str:
    """Read the current patient_id from therapy.json."""
    try:
        return str(_load_therapy().get("patient_id", "unknown"))
    except Exception:
        return "unknown"


def _get_patient_conditions() -> list[str]:
    """Read the current patient's medical conditions from therapy.json."""
    try:
        return list(_load_therapy().get("medical_conditions") or [])
    except Exception:
        return []


def _history_query(activity: dict) -> str:
    """
    Build the text used to search the patient's history for a given activity.

    Searching on the bare activity name has very poor recall: the stored events
    describe an outcome ("Frank became dehydrated after a hot afternoon…") while
    the name is a label ("Outdoor Gardening"), so the two embed far apart.
    Adding the category and the patient's conditions bridges that gap — measured
    on the current dataset it moves the relevant event from distance 1.01 to 0.75.
    """
    parts = [
        str(activity.get("name") or "").strip(),
        str(activity.get("description") or "").strip(),
    ]
    category = activity.get("category")
    if category:
        parts.append(f"({category})")
    conditions = _get_patient_conditions()
    if conditions:
        parts.append(f"for a patient with {', '.join(conditions)}")
    return " ".join(p for p in parts if p)


def _generate_activity_id(category: str, therapy: dict) -> str:
    """
    Build the next free activity_id for a category, as '<prefix>_<NNN>'.

    Considers both active and expired activities so an id is never reused, and
    scans every id sharing the prefix regardless of which category it currently
    belongs to.
    """
    prefix = CATEGORY_ID_PREFIX.get(category, "act")
    taken = {
        act.get("activity_id", "")
        for act in therapy.get("activities", []) + therapy.get("expired_activities", [])
    }

    used_numbers = set()
    for activity_id in taken:
        match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", activity_id or "")
        if match:
            used_numbers.add(int(match.group(1)))

    counter = 1
    while counter in used_numbers or f"{prefix}_{counter:03d}" in taken:
        counter += 1
    return f"{prefix}_{counter:03d}"


def _ensure_data_dir():
    """Create the data directory if it does not exist."""
    THERAPY_FILE.parent.mkdir(exist_ok=True)


def _load_therapy():
    """Load therapy.json and return its contents as a dict."""
    _ensure_data_dir()

    if not THERAPY_FILE.exists():
        # therapy.json is untracked, so this is the normal state of a fresh
        # clone rather than an error. Start from the versioned seed when it is
        # there; the empty stub is the last resort.
        if THERAPY_SEED_FILE.exists():
            logger.info(f"[THERAPY] therapy.json not found – seeding from {THERAPY_SEED_FILE.name}")
            data = json.loads(THERAPY_SEED_FILE.read_text(encoding="utf-8"))
        else:
            logger.warning("[THERAPY] therapy.json not found, creating empty structure")
            data = {"patient_id": "test", "patient_name": "Test", "activities": []}
        _save_therapy(data)
        return data

    try:
        with open(THERAPY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"[THERAPY] Error loading therapy.json: {e}")
        raise


def _save_therapy(data):
    """
    Persist the given data dict to therapy.json.

    Written to a temporary file in the same directory and then moved into place:
    opening THERAPY_FILE in "w" truncates it before the new content is written,
    so an error (or a reader in another thread, which the Streamlit UI has) sees a
    half-written file. os.replace is atomic on the same filesystem, so the file is
    either the previous therapy or the new one, never a truncated mix — and
    _load_therapy raising JSONDecodeError on a corrupted file kills the session.
    """
    _ensure_data_dir()

    tmp_path = THERAPY_FILE.with_suffix(THERAPY_FILE.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        os.replace(tmp_path, THERAPY_FILE)
        logger.debug(f"[THERAPY] Saved therapy data: {len(data.get('activities', []))} activities")
    except Exception as e:
        logger.error(f"[THERAPY] Error saving therapy.json: {e}")
        tmp_path.unlink(missing_ok=True)
        raise


def get_all_activities():
    """
    Retrieve all therapy activities.

    Returns:
        str: JSON-formatted string with all activities
    """
    try:
        data = _load_therapy()

        if not data.get("activities"):
            # therapy.json uses 'patient_full_name'; older/default dicts may use 'patient_name'
            full_name = data.get("patient_full_name") or data.get("patient_name", "")
            return _tool_json(
                {
                    "status": "success",
                    "message": "No activities configured",
                    "patient_id": data.get("patient_id", ""),
                    "patient_full_name": full_name,
                    "activities": [],
                },
            )

        logger.info(f"[THERAPY] Retrieved {len(data['activities'])} activities")
        result = dict(data)
        if "objectives" in result:
            result.pop("objectives")

        result["status"] = "success"
        return _tool_json(result)

    except Exception as e:
        logger.error(f"[THERAPY] Error getting activities: {e}")
        return _tool_json({"status": "error", "message": str(e)})


def _parse_activity_date(d):
    """Parse a date string accepting YYYY-MM-DD or ISO 8601. Returns None if d is None/unparseable."""
    if d is None:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            continue
    logger.warning(f"[TOOLS] Cannot parse date: {d!r} – treating as unbounded")
    return None


def _normalise_optional_date(value):
    """
    Collapse the empty string to None for an optional date field.

    Models routinely send "" for "no end date". Stored verbatim it is neither a
    date nor absent: _parse_activity_date logs a "cannot parse" warning for it on
    every single overlap comparison, and the value reaches the diff and the UI as
    an empty string rather than as null.
    """
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _validate_date_field(value, field_name: str):
    """Return an error message string if *value* is a non-None string in an
    unrecognised date format; return None when the value is acceptable."""
    if value is None:
        return None
    if len(value) == 0:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            datetime.strptime(value, fmt)
            return None
        except ValueError:
            continue
    return f"Invalid date format for '{field_name}': expected YYYY-MM-DD, got '{value}'"


def _validate_time_field(value, field_name: str = "time"):
    """Return an error message if *value* is not a valid HH:MM string; None if acceptable."""
    import re

    if value is None:
        return None
    if not re.fullmatch(r"\d{2}:\d{2}", value):
        return f"Invalid time format for '{field_name}': expected HH:MM (e.g. 08:30), got '{value}'"
    h, m = int(value[:2]), int(value[3:])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return (
            f"Invalid time value for '{field_name}': hours must be 00-23 and minutes 00-59, "
            f"got '{value}'"
        )
    return None


def _validate_duration_field(value):
    """
    Return (coerced_duration, error_message); error_message is None when valid.

    The two ways of getting this wrong need different advice, and handing back the
    wrong one steers the caller into a worse answer than it started with. A single
    "must be a positive integer (e.g. 30, not 30.0)" reads as a complaint about the
    *format*, so a model that sent 0 — a reasonable way to say "taking a pill is
    instantaneous" — fixes the format and retries with 1. Observed in scenario 10:
    the resulting one-minute medication then fit a one-minute gap and was scheduled
    at 12:29 to end exactly as lunch began, which is correct scheduling of a
    nonsense duration.
    """
    # bool is a subclass of int, so it would otherwise sail through as 0/1.
    if isinstance(value, bool):
        return value, "duration_minutes must be a number of minutes, not a boolean"

    if isinstance(value, float) and value.is_integer():
        value = int(value)  # 30.0 → 30, which some models produce

    if not isinstance(value, int):
        return value, (
            f"duration_minutes must be a whole number of minutes, got {value!r}. "
            'Send it as an integer (e.g. 30 — not 30.0 and not "30").'
        )

    if value <= 0:
        return value, (
            f"duration_minutes must be greater than 0, got {value}. It is how long the "
            "activity occupies the schedule, not how long the action takes, so give it a "
            "realistic length: a medication dose is typically 5-15 minutes, a meal or an "
            "outing longer. Do not use 1 to mean 'instantaneous' — an activity that short "
            "gets placed in any one-minute gap of the day."
        )

    return value, None


def _validate_day_of_week_field(value, field_name: str = "day_of_week"):
    """Return an error message if *value* is not a non-empty list of integers in [1, 7];
    return None if acceptable."""
    if value is None:
        return None
    if not isinstance(value, list) or len(value) == 0:
        return f"'{field_name}' must be a non-empty list of day numbers (1=Mon … 7=Sun)"
    invalid = [d for d in value if not isinstance(d, int) or not (1 <= d <= 7)]
    if invalid:
        return (
            f"'{field_name}' contains invalid day value(s) {invalid}; "
            "allowed range is 1 (Monday) to 7 (Sunday)"
        )
    return None


def _date_ranges_overlap(from1, until1, from2, until2) -> bool:
    """Return True if two [from, until] date ranges overlap. None means unbounded."""
    s1 = _parse_activity_date(from1)
    e1 = _parse_activity_date(until1)
    s2 = _parse_activity_date(from2)
    e2 = _parse_activity_date(until2)
    # Range 1 starts strictly after range 2 ends
    if e2 is not None and s1 is not None and s1 > e2:
        return False
    # Range 2 starts strictly after range 1 ends
    if e1 is not None and s2 is not None and s2 > e1:
        return False
    return True


def find_conflicting_activity(new_activity, schedule):
    new_start = hhmm_to_minutes(new_activity["time"])
    new_end = new_start + new_activity["duration_minutes"]
    new_days = set(new_activity["day_of_week"])

    # Sort by start time so the early-exit break is always valid,
    # regardless of whether the caller passes an already-sorted list.
    sorted_schedule = sorted(schedule, key=lambda a: hhmm_to_minutes(a["time"]))

    for act in sorted_schedule:
        act_start = hhmm_to_minutes(act["time"])

        # If we surpass the new activity ending time we can stop
        if act_start >= new_end:
            break

        # Check if the activity takes place on the same day(s)
        if not new_days & set(act["day_of_week"]):
            continue

        # Check if both activities' validity periods overlap
        if not _date_ranges_overlap(
            new_activity.get("valid_from"),
            new_activity.get("valid_until"),
            act.get("valid_from"),
            act.get("valid_until"),
        ):
            continue

        # Check time overlap
        act_end = act_start + act["duration_minutes"]
        if new_start < act_end:
            return act

    return None  # no conflict


def find_earlier_time(activity, schedule):
    """Return the latest valid start time strictly before (or equal to) the
    current start time, fitting the activity in a free gap.

    Instead of recursing one slot at a time this function collects *all*
    candidate end-times (the start of every existing activity that could be
    reached by moving backwards, the conflicting one included) and tries them
    from latest to earliest.

    It replaces a recursive version that walked back one conflicting activity at
    a time. That one lived on here as the caller's choice long after this
    replacement was written and left unused; its counterpart `find_later_time`
    also returned "24:00" — a start time the tools' own validation rejects, so
    the caregiver was offered an alternative that could not be applied.
    """
    duration = activity["duration_minutes"]
    if duration <= 0:
        return activity["time"]

    current_start = hhmm_to_minutes(activity["time"])
    current_end = current_start + duration

    # Build a sorted (descending) list of candidate end-times.
    # Each candidate places the new activity to end exactly when an existing
    # one begins, working from the current position backwards.
    # The bound is on the *resulting* start (it must not be later than the
    # current one), i.e. `act_start <= current_end` — not on the anchor itself:
    # an activity beginning after the current start but before the current end
    # is exactly the one being conflicted with, and anchoring on it gives the
    # latest slot that still ends before it. Bounding by `current_start`
    # dropped that anchor and moved the activity a whole extra `duration`
    # earlier than needed.
    # Only consider activities that share at least one day with the new activity
    # so that cross-day activities don't distort the available time windows.
    new_days = set(activity["day_of_week"])
    candidate_ends = sorted(
        {
            hhmm_to_minutes(act["time"])
            for act in schedule
            if hhmm_to_minutes(act["time"]) <= current_end and new_days & set(act["day_of_week"])
        }
        # `current_end`, not `current_start`: these are *end* times, so the
        # anchor standing for "the requested slot is already fine" is the end
        # of that slot. Anchoring on `current_start` made the best reachable
        # answer `current_start - duration`, i.e. a whole duration earlier
        # than asked even when nothing was in the way.
        | {current_end},
        reverse=True,
    )

    for cand_end in candidate_ends:
        cand_start = cand_end - duration
        if cand_start < 0:
            continue
        test_activity = {**activity, "time": minutes_to_hhmm(cand_start)}
        if find_conflicting_activity(test_activity, schedule) is None:
            return minutes_to_hhmm(cand_start)

    return None


def find_later_time(activity, schedule):
    """Return the earliest valid start time strictly after the current one.

    Collects candidate start-times (the end of every existing activity that is
    still running at the current start time or begins later — the conflicting
    one included) and returns the first one that fits within the day without
    conflicts — `None` when the activity would have to cross midnight to fit,
    which is the case the recursive version it replaces answered with "24:00".
    """
    duration = activity["duration_minutes"]
    if duration <= 0:
        return activity["time"]

    current_start = hhmm_to_minutes(activity["time"])

    # Candidate start-times: begin right after each activity that is still
    # running when the current slot starts, or that starts later.
    # The bound is `act_end > current_start`, not `act_end >= current_end`:
    # an activity longer than the overlap ends *before* the current slot does,
    # so the latter dropped the conflicting activity itself from the anchors
    # and picked the next unrelated one — a lunch conflict was answered with
    # "postpone to after dinner" instead of "postpone to when lunch ends".
    # Only consider activities that share at least one day with the new activity.
    new_days = set(activity["day_of_week"])
    candidate_starts = sorted(
        {
            hhmm_to_minutes(act["time"]) + act["duration_minutes"]
            for act in schedule
            if hhmm_to_minutes(act["time"]) + act["duration_minutes"] > current_start
            and new_days & set(act["day_of_week"])
        }
    )

    for cand_start in candidate_starts:
        if cand_start + duration > 24 * 60:
            break
        test_activity = {**activity, "time": minutes_to_hhmm(cand_start)}
        if find_conflicting_activity(test_activity, schedule) is None:
            return minutes_to_hhmm(cand_start)

    return None


def find_scheduling_conflicts(new_activity, schedule, patient_id: str = None):
    conflicting_activity = find_conflicting_activity(new_activity, schedule)
    if conflicting_activity:
        anticipate_time = find_earlier_time(copy.deepcopy(new_activity), schedule)
        postpone_time = find_later_time(copy.deepcopy(new_activity), schedule)
        suggestion_string = ""

        if not anticipate_time and not postpone_time:
            suggestion_string = f"""There are not possible alternative time for the '{new_activity["name"]}'
            with its current duration. Please either change the duration or the conflicting activity"""

        else:
            suggestion_string = f"""Suggestions to solve the problem:
                    {f"- Anticipate the activity at {anticipate_time}" if anticipate_time else ""}
                    {f"- Postpone the activity at {postpone_time}" if postpone_time else ""}
                    - Modify the conflicting activity
                """

        # Query past conflict resolution hints from the vector DB
        past_hints: list[dict] = []
        if _vector_db is not None:
            conflict_query = (
                f"Scheduling conflict between {new_activity['name']} "
                f"and {conflicting_activity['name']}"
            )
            past_hints = _vector_db.query_conflict_resolutions(
                conflict_query, patient_id=patient_id
            )

        result: dict = {
            "status": "failure",
            "issue": ISSUE_SCHEDULE_CONFLICT,
            "message": (
                f"The activity '{new_activity['name']}' cannot be scheduled at "
                f"{new_activity['time']} because it overlaps with the activity "
                f"named '{conflicting_activity['name']}'.\n{suggestion_string}"
            ),
        }
        if past_hints:
            result["past_resolution_hints"] = past_hints

        return result
    else:
        return None


def add_therapy_activity(activity_data):
    """
    Add a new therapy activity.

    Args:
        activity_data: dict with activity fields (activity_id, name, description,
                       day_of_week, time, duration_minutes, dependencies, valid_from, valid_until)

    Returns:
        str: JSON-formatted confirmation or error message
    """
    try:
        data = _load_therapy()

        # Check of mandatory fields. activity_id is deliberately absent: it is
        # assigned by _generate_activity_id below, never taken from the caller.
        required_fields = [
            "name",
            "day_of_week",
            "time",
            "duration_minutes",
            "category",
        ]
        for field in required_fields:
            if field not in activity_data:
                return _tool_json(
                    {
                        "status": "error",
                        "message": f"Mandatory field missing: {field}",
                    },
                )

        # Validate category
        act_category = activity_data.get("category")
        if act_category not in CATEGORIES:
            return _tool_json(
                {
                    "status": "error",
                    "message": f"Category {act_category} is not allowed. Admitted values {','.join(CATEGORIES)}",
                },
            )

        # Validate time format (must be HH:MM)
        time_err = _validate_time_field(activity_data.get("time"), "time")
        if time_err:
            return _tool_json({"status": "error", "message": time_err})

        # Normalise "" to None *in activity_data*, so the value that is stored
        # below is the normalised one. Doing it on local copies only left the
        # empty string in the saved activity.
        for date_field in ("valid_from", "valid_until"):
            if date_field in activity_data:
                activity_data[date_field] = _normalise_optional_date(activity_data[date_field])

        # Validate optional date fields (must be YYYY-MM-DD or ISO 8601)
        for date_field in ("valid_from", "valid_until"):
            date_err = _validate_date_field(activity_data.get(date_field), date_field)
            if date_err:
                return _tool_json({"status": "error", "message": date_err})

        # Validate that valid_from precedes valid_until when both are provided
        vf = activity_data.get("valid_from")
        vu = activity_data.get("valid_until")
        if vf and vu:
            vf_dt = _parse_activity_date(vf)
            vu_dt = _parse_activity_date(vu)
            if vf_dt and vu_dt and vf_dt > vu_dt:
                return _tool_json(
                    {
                        "status": "error",
                        "message": f"'valid_from' ({vf}) must be before or equal to 'valid_until' ({vu})",
                    },
                )

        # Validate day_of_week (non-empty, values 1–7)
        dow_err = _validate_day_of_week_field(activity_data.get("day_of_week"), "day_of_week")
        if dow_err:
            return _tool_json({"status": "error", "message": dow_err})

        # Validate duration_minutes (positive whole number of minutes)
        duration, duration_err = _validate_duration_field(activity_data.get("duration_minutes", 0))
        if duration_err:
            return _tool_json({"status": "error", "message": duration_err})
        activity_data["duration_minutes"] = duration

        # Assign the activity_id here so it always follows the '<prefix>_<NNN>'
        # convention and can never collide with an existing or expired activity.
        activity_id = _generate_activity_id(act_category, data)
        proposed_id = activity_data.get("activity_id")
        if proposed_id and proposed_id != activity_id:
            logger.info(
                f"[THERAPY] Ignoring caller-provided activity_id "
                f"'{proposed_id}', assigned '{activity_id}'"
            )

        new_activity = {
            "activity_id": activity_id,
            "name": activity_data["name"],
            "description": activity_data.get("description", ""),
            "day_of_week": activity_data["day_of_week"],
            "time": activity_data["time"],
            "duration_minutes": activity_data["duration_minutes"],
            "dependencies": activity_data.get("dependencies", []),
            "valid_from": activity_data.get("valid_from"),
            "valid_until": activity_data.get("valid_until"),
            "category": activity_data["category"],
        }

        patient_id = _get_patient_id()

        # Validate that all declared dependencies exist in the current schedule
        if new_activity.get("dependencies"):
            existing_ids = {act["activity_id"] for act in data["activities"]}
            missing_deps = [dep for dep in new_activity["dependencies"] if dep not in existing_ids]
            if missing_deps:
                return _tool_json(
                    {
                        "status": "error",
                        "issue": ISSUE_MISSING_DEPENDENCY,
                        "message": (
                            f"Dependencies not found in schedule: {', '.join(missing_deps)}. "
                            "Use the exact activity_id." + _NOTHING_CREATED
                        ),
                    },
                )

            # Validate temporal ordering: every dependency must end before this activity starts
            new_start = hhmm_to_minutes(new_activity["time"])
            activity_map = {act["activity_id"]: act for act in data["activities"]}
            for dep_id in new_activity["dependencies"]:
                dep = activity_map.get(dep_id)
                if dep:
                    dep_end = hhmm_to_minutes(dep["time"]) + dep["duration_minutes"]
                    if dep_end > new_start:
                        return _tool_json(
                            {
                                "status": "error",
                                "issue": ISSUE_TEMPORAL_ORDERING,
                                "message": (
                                    f"Temporal ordering violation: '{new_activity['name']}' "
                                    f"would start at {new_activity['time']}, but it depends on "
                                    f"'{dep['name']}', which ends at {minutes_to_hhmm(dep_end)}. "
                                    f"Re-add it at {minutes_to_hhmm(dep_end)} or later."
                                    + _NOTHING_CREATED
                                ),
                            },
                        )

        # ── Patient history check (RAG) ─────────────────────────────────────
        history_warnings: list[dict] = []
        if _vector_db is not None:
            history_warnings = _vector_db.query_patient_history(
                patient_id, _history_query(new_activity)
            )

        # ── Scheduling conflict check ────────────────────────────────────────
        conflict = find_scheduling_conflicts(
            new_activity, data["activities"], patient_id=patient_id
        )

        if conflict:
            result = dict(conflict)
            # find_scheduling_conflicts is shared with the update path, where the
            # activity does exist; only here does the refusal mean nothing was
            # created. The suggested alternative times read as "move it", which is
            # exactly the reading that has to be steered away from update_*.
            result["message"] = result.get("message", "") + _NOTHING_CREATED
            if history_warnings:
                result["patient_history_warnings"] = history_warnings
            return _tool_json(result)
        else:
            data["activities"].append(new_activity)
            data["activities"].sort(
                key=lambda act: int(act["time"][:2]) * 60 + int(act["time"][3:])
            )
            _save_therapy(data)

            logger.info(
                f"[THERAPY] Added activity: {new_activity['activity_id']} - {new_activity['name']}"
            )

            result = {
                "status": "success",
                "message": f"Activity '{new_activity['name']}' successfully added",
                "activity": new_activity,
            }
            if history_warnings:
                result["patient_history_warnings"] = history_warnings

            return _tool_json(result)

    except Exception as e:
        logger.error(f"[THERAPY] Error adding activity: {e}")
        return _tool_json({"status": "error", "message": str(e)})


def update_therapy_activity(activity_id, updates):
    """
    Update an existing therapy activity.

    Args:
        activity_id: ID of the activity to update
        updates: dict with the fields to update

    Returns:
        str: JSON-formatted confirmation or error message
    """
    try:
        data = _load_therapy()

        activity_index = None
        for i, act in enumerate(data["activities"]):
            if act["activity_id"] == activity_id:
                activity_index = i
                break

        if activity_index is None:
            return _tool_json(
                {
                    "status": "error",
                    "message": f"Couldn't find activity with id '{activity_id}'",
                },
            )

        # An update carrying no field to change is not a successful update: it is
        # a malformed call. Reporting "successfully updated" for it produced the
        # one thing the manager's prompt forbids above everything else — a tool
        # result confirming a change that never happened.
        effective_updates = {k: v for k, v in (updates or {}).items() if k != "activity_id"}
        if not effective_updates:
            return _tool_json(
                {
                    "status": "error",
                    "message": (
                        f"No field to update was provided for activity '{activity_id}'. "
                        "Pass the fields that must change (e.g. time, duration_minutes, "
                        "day_of_week); nothing was modified."
                    ),
                },
            )
        updates = effective_updates

        # Normalise "" to None for the optional date fields before anything reads them.
        for date_field in ("valid_from", "valid_until"):
            if date_field in updates:
                updates[date_field] = _normalise_optional_date(updates[date_field])

        old_activity = copy.deepcopy(data["activities"][activity_index])

        # Build the validated/updated activity on a *copy* so that the stored
        # data is never mutated before all checks pass.
        updated_activity = copy.deepcopy(old_activity)
        for key, value in updates.items():
            updated_activity[key] = value

        # Schedule without this activity for conflict/dependency checks
        temp_activities = [a for i, a in enumerate(data["activities"]) if i != activity_index]

        patient_id = _get_patient_id()

        # Validate category
        if "category" in updates:
            if updates["category"] not in CATEGORIES:
                return _tool_json(
                    {
                        "status": "error",
                        "message": f"Category {updates['category']} is not allowed. Admitted values {','.join(CATEGORIES)}",
                    },
                )

        # Validate time format if it is being updated (must be HH:MM)
        if "time" in updates:
            time_err = _validate_time_field(updates["time"], "time")
            if time_err:
                return _tool_json({"status": "error", "message": time_err})

        # Validate optional date fields (must be YYYY-MM-DD or ISO 8601)
        for date_field in ("valid_from", "valid_until"):
            if date_field in updates:
                date_err = _validate_date_field(updates[date_field], date_field)
                if date_err:
                    return _tool_json({"status": "error", "message": date_err})

        # Validate that valid_from precedes valid_until (consider merged values from both
        # the update and the existing activity so partial updates are handled correctly)
        merged_vf = updates.get("valid_from", updated_activity.get("valid_from"))
        merged_vu = updates.get("valid_until", updated_activity.get("valid_until"))
        if merged_vf and merged_vu:
            vf_dt = _parse_activity_date(merged_vf)
            vu_dt = _parse_activity_date(merged_vu)
            if vf_dt and vu_dt and vf_dt > vu_dt:
                return _tool_json(
                    {
                        "status": "error",
                        "message": f"'valid_from' ({merged_vf}) must be before or equal to 'valid_until' ({merged_vu})",
                    },
                )

        # Validate day_of_week if being updated (non-empty, values 1–7)
        if "day_of_week" in updates:
            dow_err = _validate_day_of_week_field(updates["day_of_week"], "day_of_week")
            if dow_err:
                return _tool_json({"status": "error", "message": dow_err})

        # Validate duration_minutes if being updated (positive whole number of minutes)
        if "duration_minutes" in updates:
            duration, duration_err = _validate_duration_field(updates["duration_minutes"])
            if duration_err:
                return _tool_json({"status": "error", "message": duration_err})
            updates["duration_minutes"] = duration
            updated_activity["duration_minutes"] = duration

        # Validate that updated dependencies exist in the schedule
        new_deps = updates.get("dependencies")
        if new_deps is not None:
            existing_ids = {act["activity_id"] for act in temp_activities}
            missing_deps = [dep for dep in new_deps if dep not in existing_ids]
            if missing_deps:
                return _tool_json(
                    {
                        "status": "error",
                        "issue": ISSUE_MISSING_DEPENDENCY,
                        "message": (
                            f"Dependencies not found in schedule: {', '.join(missing_deps)}. "
                            "Use the exact activity_id."
                        ),
                    },
                )

        # Validate temporal ordering: every dependency must end before this activity
        # starts. This is checked against the *effective* dependency list — the one
        # the activity ends up with — and whenever anything that can break the
        # ordering moves, not only when the dependencies themselves are restated.
        # Gating it on `dependencies in updates` meant that changing just the time
        # of an activity walked it in front of its own dependency and the write was
        # accepted: exactly the constraint these tools exist to enforce, silently
        # skipped in the most common way of expressing the move.
        if new_deps is not None or "time" in updates or "duration_minutes" in updates:
            effective_deps = updated_activity.get("dependencies") or []
            updated_start = hhmm_to_minutes(updated_activity["time"])
            activity_map = {act["activity_id"]: act for act in temp_activities}
            for dep_id in effective_deps:
                dep = activity_map.get(dep_id)
                if dep:
                    dep_end = hhmm_to_minutes(dep["time"]) + dep["duration_minutes"]
                    if dep_end > updated_start:
                        return _tool_json(
                            {
                                "status": "error",
                                "issue": ISSUE_TEMPORAL_ORDERING,
                                "message": (
                                    f"Temporal ordering violation: '{updated_activity['name']}' "
                                    f"would start at {updated_activity['time']}, but it depends on "
                                    f"'{dep['name']}', which ends at {minutes_to_hhmm(dep_end)}. "
                                    f"It was NOT modified and still stands as it was; retry at "
                                    f"{minutes_to_hhmm(dep_end)} or later."
                                ),
                            },
                        )

        # ── Patient history check (RAG) ─────────────────────────────────────
        history_warnings: list[dict] = []
        if _vector_db is not None:
            history_warnings = _vector_db.query_patient_history(
                patient_id, _history_query(updated_activity)
            )

        # ── Scheduling conflict check ────────────────────────────────────────
        conflict = find_scheduling_conflicts(
            updated_activity, temp_activities, patient_id=patient_id
        )
        if conflict:
            result = dict(conflict)
            if history_warnings:
                result["patient_history_warnings"] = history_warnings
            return _tool_json(result)
        else:
            # ── Dependent ordering check ──────────────────────────────────────
            # If time or duration changed, verify that every activity that lists
            # this one as a dependency still starts after it ends.
            if "time" in updates or "duration_minutes" in updates:
                updated_end = (
                    hhmm_to_minutes(updated_activity["time"]) + updated_activity["duration_minutes"]
                )
                violations = []
                for act in temp_activities:
                    if activity_id in act.get("dependencies", []):
                        if hhmm_to_minutes(act["time"]) < updated_end:
                            violations.append(
                                f"'{act['name']}' starts at {act['time']} but "
                                f"'{updated_activity['name']}' would end at "
                                f"{minutes_to_hhmm(updated_end)}"
                            )
                if violations:
                    return _tool_json(
                        {
                            "status": "error",
                            "issue": ISSUE_TEMPORAL_ORDERING,
                            "message": (
                                "Temporal ordering violation: the update would cause "
                                "this activity to end after the start of its "
                                f"dependent(s): {'; '.join(violations)}"
                            ),
                        },
                    )

            # All checks passed – commit the update
            data["activities"][activity_index] = updated_activity
            data["activities"].sort(
                key=lambda act: int(act["time"][:2]) * 60 + int(act["time"][3:])
            )
            _save_therapy(data)

            logger.info(f"[THERAPY] Updated activity: {activity_id}")
            logger.debug(f"[THERAPY] Old: {old_activity}")
            logger.debug(f"[THERAPY] New: {updated_activity}")

            result = {
                "status": "success",
                "message": f"The activity '{updated_activity['name']}' has been updated successfully",
                "activity": updated_activity,
            }
            if history_warnings:
                result["patient_history_warnings"] = history_warnings

            return _tool_json(result)

    except Exception as e:
        logger.error(f"[THERAPY] Error updating activity: {e}")
        return _tool_json({"status": "error", "message": str(e)})


def remove_therapy_activity(activity_id):
    """
    Remove a therapy activity.

    Args:
        activity_id: ID of the activity to remove

    Returns:
        str: JSON-formatted confirmation or error message
    """
    try:
        data = _load_therapy()

        # First pass: locate the activity to remove
        activity_index = None
        removed_activity = {}
        for i, act in enumerate(data["activities"]):
            if act["activity_id"] == activity_id:
                activity_index = i
                removed_activity = act
                break

        if activity_index is None:
            return _tool_json(
                {
                    "status": "error",
                    "message": f"Couldn't find activity with id '{activity_id}'",
                },
            )

        # Second pass: check dependents using the activity's ID.
        # Dependencies are stored as lists of activity IDs, but this message is
        # the one the caregiver gets to hear: it is a blocking `issue`, so the
        # assistant relays it. Naming the activities instead of their ids keeps
        # it relayable — the previous wording ("Cannot remove 'ml_001' because it
        # is a dependency of: med_001") gave the model nothing but the ids its
        # own prompt forbids it to show, and no names to substitute for them.
        dependent_names = [
            act.get("name", act["activity_id"])
            for act in data["activities"]
            if activity_id in act.get("dependencies", [])
        ]

        if dependent_names:
            return _tool_json(
                {
                    "status": "error",
                    "issue": ISSUE_DEPENDENCY_BLOCKED,
                    "message": (
                        f"Cannot remove '{removed_activity.get('name', activity_id)}' because "
                        f"the following activities depend on it: {', '.join(dependent_names)}."
                    ),
                },
            )

        data["activities"].pop(activity_index)
        _save_therapy(data)

        logger.info(f"[THERAPY] Removed activity: {activity_id} - {removed_activity['name']}")

        return _tool_json(
            {
                "status": "success",
                "message": f"The activity '{removed_activity['name']}' has been removed successfully",
                "removed_activity": removed_activity,
            },
        )

    except Exception as e:
        logger.error(f"[THERAPY] Error removing activity: {e}")
        return _tool_json({"status": "error", "message": str(e)})


def get_medicine_data(medicine_name: str) -> str:
    """
    RAG-based medicine data retrieval.
    Queries the ChromaDB medicines collection for the most relevant document(s).
    Falls back to a 'not available' message if the vector DB is not initialised.
    """
    if _vector_db is not None:
        return _vector_db.query_medicines(medicine_name)
    logger.warning("[TOOLS] Vector DB not available – cannot retrieve medicine data")
    return f"Medicine data for '{medicine_name}' is not available (vector DB not initialised)."


def get_patient_preferences(query: str = "") -> str:
    """
    Retrieve known preferences for the current patient from ChromaDB.
    If *query* is provided, performs a semantic search and returns the closest matches.
    If *query* is empty, returns all preferences for the patient.
    Returns a JSON-formatted string for the LLM to consume.
    """
    if _vector_db is None:
        logger.warning("[TOOLS] Vector DB not available – cannot retrieve preferences")
        return _tool_json(
            {"status": "error", "message": "Vector DB not available"},
        )
    patient_id = _get_patient_id()
    prefs = _vector_db.query_patient_preferences(patient_id, query=query)
    if not prefs:
        return _tool_json(
            {
                "status": "success",
                "patient_id": patient_id,
                "preferences": [],
                "message": "No preferences recorded for this patient.",
            },
        )
    return _tool_json(
        {"status": "success", "patient_id": patient_id, "preferences": prefs},
    )


def get_patient_history_events(query: str) -> str:
    """
    Proactive RAG lookup against the patient history collection.
    Returns past danger/warning events semantically similar to *query*.
    Call this before proposing or adding any non-medicine activity so that
    hazardous patterns are surfaced to the caregiver immediately.
    """
    if _vector_db is None:
        logger.warning("[TOOLS] Vector DB not available – cannot retrieve patient history")
        return _tool_json(
            {"status": "error", "message": "Vector DB not available"},
        )
    patient_id = _get_patient_id()
    # Enrich the caller's query the same way the add/update paths do, so recall
    # does not depend on how precisely the agent happened to word the lookup.
    events = _vector_db.query_patient_history(patient_id, _history_query({"name": query}))
    if not events:
        return _tool_json(
            {
                "status": "success",
                "patient_id": patient_id,
                "events": [],
                "message": "No relevant history events found for this activity.",
            },
        )
    return _tool_json(
        {"status": "success", "patient_id": patient_id, "events": events},
    )


def get_conflict_resolution_hints(query: str) -> str:
    """
    Proactive RAG lookup against the conflict_resolutions collection.
    Returns past resolutions (rejections, modifications, alternatives) that are
    semantically similar to *query*.
    Call this whenever the caregiver requests an activity that might conflict with
    safety rules, medical conditions, or past decisions, to surface prior context
    before asking the caregiver what to do.
    """
    if _vector_db is None:
        logger.warning("[TOOLS] Vector DB not available – cannot retrieve conflict hints")
        return _tool_json(
            {"status": "error", "message": "Vector DB not available"},
        )
    patient_id = _get_patient_id()
    hints = _vector_db.query_conflict_resolutions(query, patient_id=patient_id)
    if not hints:
        return _tool_json(
            {
                "status": "success",
                "patient_id": patient_id,
                "hints": [],
                "message": "No past resolution hints found for this activity.",
            },
        )
    return _tool_json(
        {"status": "success", "patient_id": patient_id, "hints": hints},
    )


# region Tools declaration

tools_decl = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current date and time in a human readable format",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_therapy_activities",
            "description": "Get the entire therapy of the patient",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_therapy_activity",
            "description": """
                Adds a new activity to the therapy of the current patient..
                Requires:
                - name
                - category
                - day_of_week: list of integers representing days of the week with 1=Monday and 7=Sunday
                - time (format HH:MM)
                - duration_minutes.
                Optional: description, dependencies (list of activities names), valid_from, valid_until
                Do NOT provide an activity_id: it is assigned automatically and
                returned in the response. Never state an id before you have read it there.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the activity"},
                    "description": {
                        "type": "string",
                        "description": "Detailed desciption of the activity",
                    },
                    "day_of_week": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Days of the week the activity takes place (1=Monday, 7=Sunday)",
                    },
                    "time": {
                        "type": "string",
                        "description": "Time the activity takes place (format HH:MM)",
                    },
                    "category": {
                        "type": "string",
                        "description": "Category of the activity. Value in ['medication','outside_activity','meal','health_checkup','therapy','relaxation','social_activity'] ",
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Duration of the activity in minutes",
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of activity_ids that must be completed before the current activity (use activity_id values, not names)",
                    },
                    "valid_from": {
                        "type": "string",
                        "description": "Date the activity starts to be valid (YYYY-MM-DD)",
                    },
                    "valid_until": {
                        "type": "string",
                        "description": "Date in which the activity ends to be valid (YYYY-MM-DD)",
                    },
                },
                "required": [
                    "name",
                    "category",
                    "day_of_week",
                    "time",
                    "duration_minutes",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_therapy_activity",
            "description": "Updates an existing activity in the therapy of the current patient. Specify the activity_id and only the fields that need to change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "activity_id": {
                        "type": "string",
                        "description": "ID of the activity to update",
                    },
                    "name": {
                        "type": "string",
                        "description": "New name of the activity",
                    },
                    "description": {
                        "type": "string",
                        "description": "New description of the activity",
                    },
                    "day_of_week": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "New days of the week (1=Monday … 7=Sunday)",
                    },
                    "time": {
                        "type": "string",
                        "description": "New time (HH:MM)",
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "New duration in minutes",
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New list of dependency activity_ids (use activity_id values, not names)",
                    },
                    "valid_from": {
                        "type": "string",
                        "description": "New valid_from date (YYYY-MM-DD)",
                    },
                    "valid_until": {
                        "type": "string",
                        "description": "New valid_until date (YYYY-MM-DD)",
                    },
                },
                "required": ["activity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_therapy_activity",
            "description": "Remove a activity in the therapy of the current patient",
            "parameters": {
                "type": "object",
                "properties": {
                    "activity_id": {
                        "type": "string",
                        "description": "ID of the activity to remove",
                    }
                },
                "required": ["activity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_medicine_data",
            "description": (
                "Get pharmacological data for a medicine via semantic search "
                "(uses RAG on the medicines vector database). "
                "Always call this before adding any medicine-related activity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "medicine_name": {
                        "type": "string",
                        "description": "Name of the medicine (e.g. aulin, tachipirina, aspirina)",
                    }
                },
                "required": ["medicine_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_preferences",
            "description": (
                "Retrieve known preferences and habits of the current patient. "
                "Provide an optional 'query' to narrow results to a specific topic "
                "(e.g. 'food', 'morning routine', 'medication timing'); "
                "omit it to retrieve all preferences."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional topic to search within the patient's preferences (e.g. 'food', 'exercise', 'sleep'). Leave empty to retrieve all preferences.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_history_events",
            "description": (
                "Retrieve past danger and warning events for the current patient "
                "that are semantically related to a given activity or topic. "
                "Call this BEFORE proposing or adding any activity to check whether "
                "a similar activity has caused harm in the past."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Description of the activity or topic to look up in the patient's safety history (e.g. 'potassium-rich snack', 'brisk walk', 'NSAID pain relief').",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_conflict_resolution_hints",
            "description": (
                "Retrieve past conflict resolutions, rejected activities, and prior "
                "caregiver decisions that are semantically related to a given activity or topic. "
                "Call this BEFORE proposing options to the caregiver so that previous decisions "
                "are taken into account and surfaced."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Description of the activity or conflict to look up in past resolution records (e.g. 'potassium snack renal failure', 'NSAID analgesic', 'evening aerobic exercise diabetes').",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_session",
            "description": "Saves the current therapy configuration session inside the database. Run this when the user tells you they finished.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_activity",
            "description": """
                Check the activity in respect with the current patient therapy, health conditions and medications.
                Requires:
                - name
                - day_of_week: list of integers representing days of the week with 1=Monday and 7=Sunday
                - time (format HH:MM)
                - duration_minutes.
                Optional: description, dependencies (list of activities names), valid_from, valid_until,
                activity_id (only for an activity that already exists)""",
            "parameters": {
                "type": "object",
                "properties": {
                    "activity_id": {
                        "type": "string",
                        "description": (
                            "ID of the activity, only when checking one that already "
                            "exists. Omit it for an activity not yet created."
                        ),
                    },
                    "name": {"type": "string", "description": "Name of the activity"},
                    "description": {
                        "type": "string",
                        "description": "Detailed desciption of the activity",
                    },
                    "day_of_week": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Days of the week the activity takes place (1=Monday, 7=Sunday)",
                    },
                    "time": {
                        "type": "string",
                        "description": "Time the activity takes place (format HH:MM)",
                    },
                    "category": {
                        "type": "string",
                        "description": "Category of the activity. Value in ['medication','outside_activity','meal','health_checkup','therapy','relaxation','social_activity'] ",
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Duration of the activity in minutes",
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of activity_ids that must be completed before the current activity (use activity_id values, not names)",
                    },
                    "valid_from": {
                        "type": "string",
                        "description": "Date the activity starts to be valid (YYYY-MM-DD)",
                    },
                    "valid_until": {
                        "type": "string",
                        "description": "Date in which the activity ends to be valid (YYYY-MM-DD)",
                    },
                },
                "required": [
                    "name",
                    "day_of_week",
                    "time",
                    "duration_minutes",
                ],
            },
        },
    },
]

# endregion
