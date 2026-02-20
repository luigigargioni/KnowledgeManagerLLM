import copy
import json
import logging

from config_loader import THERAPY_FILE
from utils import hhmm_to_minutes, minutes_to_hhmm

logger = logging.getLogger("knowledge_manager")

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


def clear_conversation_history(self, keep_system=True):
    if (
        keep_system
        and self.conversation_history
        and self.conversation_history[0]["role"] == "system"
    ):
        system_msg = self.conversation_history[0]
        self.conversation_history = [system_msg]
        return "Conversation history has been cleared. System prompt kept."
    else:
        self.conversation_history = []
        return "Conversation history has been completely cleared."


def _ensure_data_dir():
    """Create the data directory if it does not exist."""
    THERAPY_FILE.parent.mkdir(exist_ok=True)


def _load_therapy():
    """Load therapy.json and return its contents as a dict."""
    _ensure_data_dir()

    if not THERAPY_FILE.exists():
        logger.warning("[THERAPY] therapy.json not found, creating empty structure")
        default_data = {"patient_id": "test", "patient_name": "Test", "activities": []}
        _save_therapy(default_data)
        return default_data

    try:
        with open(THERAPY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"[THERAPY] Error loading therapy.json: {e}")
        raise


def _save_therapy(data):
    """Persist the given data dict to therapy.json."""
    _ensure_data_dir()

    try:
        with open(THERAPY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.debug(
            f"[THERAPY] Saved therapy data: {len(data.get('activities', []))} activities"
        )
    except Exception as e:
        logger.error(f"[THERAPY] Error saving therapy.json: {e}")
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
            return json.dumps(
                {
                    "status": "success",
                    "message": "No activities configured",
                    "patient_id": data.get("patient_id", ""),
                    "patient_full_name": full_name,
                    "activities": [],
                },
                indent=2,
                ensure_ascii=False,
            )

        logger.info(f"[THERAPY] Retrieved {len(data['activities'])} activities")
        data.update(
            {
                "status": "success",
            }
        )
        return json.dumps(data, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"[THERAPY] Error getting activities: {e}")
        return json.dumps({"status": "error", "message": str(e)}, indent=2)


def find_conflicting_activity(new_activity, schedule):
    new_start = hhmm_to_minutes(new_activity["time"])
    new_end = new_start + new_activity["duration_minutes"]
    new_days = set(new_activity["day_of_week"])

    for act in schedule:
        act_start = hhmm_to_minutes(act["time"])

        # If we surpass the new activity ending date we exit the loop
        if act_start >= new_end:
            break

        # Check if the activity take place in the same day
        if not new_days & set(act["day_of_week"]):
            continue

        # Check overlap
        act_end = act_start + act["duration_minutes"]
        if new_start < act_end:
            return act  # the loop stops at the first conflict (which is technically the only possible)

    return None  # no conflict


def find_earlier_time(activity, schedule):
    conflicting_activity = find_conflicting_activity(activity, schedule)
    if conflicting_activity:
        # loop to search for a new possible time
        conflict_time = hhmm_to_minutes(conflicting_activity["time"])

        # anticipate
        new_time = conflict_time - activity["duration_minutes"]

        if new_time < 0:
            return None

        activity["time"] = minutes_to_hhmm(new_time)
        return find_earlier_time(activity, schedule)
    else:
        return activity["time"]


def find_later_time(activity, schedule):
    conflicting_activity = find_conflicting_activity(activity, schedule)
    if conflicting_activity:
        # loop to search for a new possible time
        conflict_time = hhmm_to_minutes(conflicting_activity["time"])

        # anticipate
        new_time = conflict_time + conflicting_activity["duration_minutes"]

        if new_time > 24 * 60:
            return None

        activity["time"] = minutes_to_hhmm(new_time)
        return find_later_time(activity, schedule)
    else:
        return activity["time"]


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

        # Check of mandatory fields
        required_fields = [
            "activity_id",
            "name",
            "day_of_week",
            "time",
            "duration_minutes",
        ]
        for field in required_fields:
            if field not in activity_data:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Mandatory field missing: {field}",
                    },
                    indent=2,
                )

        # Verify unique actvitiy_id
        if any(
            act["activity_id"] == activity_data["activity_id"]
            for act in data["activities"]
        ):
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Activity with id '{activity_data['activity_id']}' already exists.",
                },
                indent=2,
            )

        new_activity = {
            "activity_id": activity_data["activity_id"],
            "name": activity_data["name"],
            "description": activity_data.get("description", ""),
            "day_of_week": activity_data["day_of_week"],
            "time": activity_data["time"],
            "duration_minutes": activity_data["duration_minutes"],
            "dependencies": activity_data.get("dependencies", []),
            "valid_from": activity_data.get("valid_from"),
            "valid_until": activity_data.get("valid_until"),
        }

        # ── Patient history check (RAG) ─────────────────────────────────────
        history_warnings: list[dict] = []
        if _vector_db is not None:
            patient_id = _get_patient_id()
            activity_query = (
                f"{new_activity['name']} {new_activity.get('description', '')}"
            )
            history_warnings = _vector_db.query_patient_history(
                patient_id, activity_query
            )

        # ── Scheduling conflict check ────────────────────────────────────────
        conflict = find_scheduling_conflicts(
            new_activity, data["activities"], patient_id=patient_id
        )

        if conflict:
            result = dict(conflict)
            if history_warnings:
                result["patient_history_warnings"] = history_warnings
            return json.dumps(result, indent=2, ensure_ascii=False)
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

            return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"[THERAPY] Error adding activity: {e}")
        return json.dumps({"status": "error", "message": str(e)}, indent=2)


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
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Couldn't find activity with id '{activity_id}'",
                },
                indent=2,
            )

        activity = data["activities"][activity_index]
        old_activity = activity.copy()

        for key, value in updates.items():
            if key != "activity_id":
                activity[key] = value

        temp_activities = copy.deepcopy(data["activities"])
        temp_activities.pop(activity_index)

        # ── Patient history check (RAG) ─────────────────────────────────────
        history_warnings: list[dict] = []
        if _vector_db is not None:
            patient_id = _get_patient_id()
            activity_query = f"{activity['name']} {activity.get('description', '')}"
            history_warnings = _vector_db.query_patient_history(
                patient_id, activity_query
            )

        # ── Scheduling conflict check ────────────────────────────────────────
        conflict = find_scheduling_conflicts(
            activity, temp_activities, patient_id=patient_id
        )
        if conflict:
            result = dict(conflict)
            if history_warnings:
                result["patient_history_warnings"] = history_warnings
            return json.dumps(result, indent=2, ensure_ascii=False)
        else:
            data["activities"].sort(
                key=lambda act: int(act["time"][:2]) * 60 + int(act["time"][3:])
            )
            _save_therapy(data)

            logger.info(f"[THERAPY] Updated activity: {activity_id}")
            logger.debug(f"[THERAPY] Old: {old_activity}")
            logger.debug(f"[THERAPY] New: {activity}")

            result = {
                "status": "success",
                "message": f"The activity '{activity['name']}' has been updated successfully",
                "activity": activity,
            }
            if history_warnings:
                result["patient_history_warnings"] = history_warnings

            return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"[THERAPY] Error updating activity: {e}")
        return json.dumps({"status": "error", "message": str(e)}, indent=2)


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

        # Find and remove the activity
        activity_index = None
        removed_activity = {}
        dependent_acivities = []

        for i, act in enumerate(data["activities"]):
            if activity_id in act["dependencies"]:
                dependent_acivities.append(act["name"])

            if act["activity_id"] == activity_id:
                activity_index = i
                removed_activity = act

        if activity_index is None:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Couldn't find activity with id '{activity_id}'",
                },
                indent=2,
            )

        if len(dependent_acivities) > 0:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"""Couldn't remove the activity because it is a dependency 
                    of the following activities {",".join(dependent_acivities)}""",
                },
                indent=2,
            )

        data["activities"].pop(activity_index)
        _save_therapy(data)

        logger.info(
            f"[THERAPY] Removed activity: {activity_id} - {removed_activity['name']}"
        )

        return json.dumps(
            {
                "status": "success",
                "message": f"The activity '{removed_activity['name']}' has been removed successfully",
                "removed_activity": removed_activity,
            },
            indent=2,
            ensure_ascii=False,
        )

    except Exception as e:
        logger.error(f"[THERAPY] Error removing activity: {e}")
        return json.dumps({"status": "error", "message": str(e)}, indent=2)


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


def get_patient_preferences() -> str:
    """
    Retrieve all known preferences for the current patient from ChromaDB.
    Returns a JSON-formatted string for the LLM to consume.
    """
    if _vector_db is None:
        logger.warning("[TOOLS] Vector DB not available – cannot retrieve preferences")
        return json.dumps(
            {"status": "error", "message": "Vector DB not available"},
            indent=2,
        )
    patient_id = _get_patient_id()
    prefs = _vector_db.query_patient_preferences(patient_id)
    if not prefs:
        return json.dumps(
            {
                "status": "success",
                "patient_id": patient_id,
                "preferences": [],
                "message": "No preferences recorded for this patient.",
            },
            indent=2,
        )
    return json.dumps(
        {"status": "success", "patient_id": patient_id, "preferences": prefs},
        indent=2,
        ensure_ascii=False,
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
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "clear_conversation_history",
    #         "description": "Clears the conversation history. Use this function when the user wants to reset the session",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "keep_system_prompt": {
    #                     "type": "boolean",
    #                     "description": "If true, keeps the system prompt. Always set it to true unless the users explicitly say otherwise",
    #                 }
    #             },
    #             "required": [],
    #         },
    #     },
    # },
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
                - activity_id (unique)
                - name
                - day_of_week: list of integers representing days of the week with 1=Monday and 7=Sunday
                - time (format HH:MM)
                - duration_minutes. 
                Optional: description, dependencies (list of activities names), valid_from, valid_until""",
            "parameters": {
                "type": "object",
                "properties": {
                    "activity_id": {
                        "type": "string",
                        "description": "Unique ID of the activity (e.g.: 'lb_001')",
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
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Duration of the activity in minutes",
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of activities names that need to be completed before the current activity",
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
                    "activity_id",
                    "name",
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
            "description": "Updates a activity in the therapy of the current patient. You need to specify the id of the activity to updated and the modified fields",
            "parameters": {
                "type": "object",
                "properties": {
                    "activity_id": {
                        "type": "string",
                        "description": "ID of the activity to update",
                    },
                    "updates": {
                        "type": "object",
                        "description": "Object withe the fields to update (name, description, day_of_week, time, duration_minutes, dependencies, valid_from, valid_until)",
                    },
                },
                "required": ["activity_id", "updates"],
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
                "Retrieve all known preferences and habits of the current patient. "
                "Use this to personalise therapy suggestions "
                "(e.g. preferred activity time, food preferences, medication habits)."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
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
]

# endregion
