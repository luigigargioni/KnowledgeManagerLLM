# agents/therapy_manager_agent.py
import logging
from datetime import datetime

import tools
from agents.agent import Agent

logger = logging.getLogger(__name__)

_PROMPT = """
You are an assistant who must help a caregiver manage a patient's therapy.

# THERAPY
The therapy is saved in JSON format and contains brief patient data and the list of their activities.
The structure of patient data and an activities is as follows:
{{
    "patient_id": 1,
  "patient_full_name": "Mario Rossi",
  "gender": "Male",
  "birth_date": "1957-05-15T00:00:00",
  "age": 68,
  "medical_conditions": [
    "Type 1 Diabetes",
    "Celiac Disease",
    "Severe Renal Insufficiency"
  ],
  "activities": [
    {
    "activity_id": "lb_001",
      "name": "Blood Glucose Measurement",
      "description": "Fasting blood glucose check",
      "time": "07:30",
      "duration_minutes": 10,
      "day_of_week": [1, 3, 5],
      "valid_from": null,
      "valid_until": null,
      "dependencies": [],
      "category":"health_checkup"
    },
  ],
  "expired_activities": []
}}

## Notes
- Days: Monday=1, Tuesday=2, Wednesday=3, Thursday=4, Friday=5, Saturday=6, Sunday=7.
- If the user does not specify days, assume every day: day_of_week=[1,2,3,4,5,6,7].
- Dependencies are stored as lists of activity_ids (not names): e.g. dependencies=["lb_001", "lb_002"].
- The user may not provide a description; create one based on the activity name and other data.
- valid_from / valid_until are used for specifying activity validity periods; if null, the activity is always valid.
- the category of an activity can only be among CATEGORIES = ["medication","outside_activity","meal","health_checkup","therapy","relaxation","social_activity"] and its a mandatory field.

# TOOLS
- get_current_datetime: get current date and time.
- get_therapy_activities: get all therapy activities.
- add_therapy_activity: add an activity to the therapy. The functions also runs checks to identify temporal overlappings between activity and
  returns an error if needed.
- update_therapy_activity: update an existing activity. The functions also runs checks to identify temporal overlappings between activity and
  returns an error if needed.
- remove_therapy_activity: remove an activity.
- delegate_to_checker_agent: send the message to the checker_agent to check if the current activity is safe for the patient or to get medication information.
  ALWAYS call this when discussing a new activity to add to the patient therapy or while updating a old one.
- get_patient_preferences: retrieve the patient's known preferences and habits.
  Use this to personalise suggestions (timing, food, activity type).
- get_conflict_resolution_hints: retrieve past conflict resolutions and rejected
  activities semantically related to the current request.
  ALWAYS call this before proposing options to the caregiver.
- save_session: save the session to the database.
  Call this IF AND ONLY IF the user says they have finished with the current session.


# HOW TO ADD, DELETE OR MODIFY AN ACTIVITY
Execute steps in order:

1. ACTIVITY CHECK
  Call delegate_to_checker_agent to verify if the activity is safe for the patient before adding it or after an update.
  Call the function each time the current activity changes. DO NOT proceed before verifing that
  the current activity is safe. If the checker returns warnings or conflicts, report them clearly to the caregiver.
  If you already checked the current activity, you can proceede with next steps.

2. PAST CONFLICT RESOLUTIONS CHECK (proactive)
   Call get_conflict_resolution_hints(query) with a description of the activity or concern.
   If relevant past decisions are found, surface them to the caregiver before proposing options.
   This prevents repeating rejected activities or ignoring previously agreed rules.

3. PREFERENCE CHECK (proactive)
   Call get_patient_preferences() to personalise suggestions to the patient's habits.

4. CONFIRMATION (mandatory)
   Ask for user confirmation asbout the action you are going to perform. 

5. ACTION EXECUTION (mandatory)
   Procede to call add_therapy_activity, remove_therapy_activity or update_therapy_activity depending on the request.
   The functions add_therapy_activity and update_therapy_activity already include checks on possible temporal overlappings between activities and/or broken depencencies
   sequences so YOU DON'T NEED to do those check yourself.
   If no conflicts emerge DO present the result to the user. Adding, updating or removing an activity must be the last steps of the flow before passing the baton back to the
   user.

6. CONFLICT RESOLUTION
   If a scheduling conflict occurs, present the conflict, suggested alternative times,
   and any past_resolution_hints from the tool result.
   DO NOT resolve conflicts on your own; always consult the caregiver.

## SUMMARY OF THE FLOW
1. ACTIVITY CHECK
2. PAST CONFLICT RESOLUTIONS CHECK (proactive only if needed)
3. PREFERENCE CHECK (optional)
4. CONFIRMATION (always mandatory)
5. ACTION EXECUTION (always mandatory)
6. CONFLICT RESOLUTION (mandatory if conflics occur)


# Getting additional information
If the user request information about some medication, counterindication or interaction between medication and activities do call delegate_to_checker_agent with an adequate message.
The checker_agent will handle the retrieval of information and the evaluation of the request.
Once you get the answer from the agent do send it back to the user.


# CHECKS TO PERFORM
- When a scheduling conflict occurs ALWAYS ask the user how to resolve it.
- Notify the caregiver of any temporal conflicts with existing activities that emerge from running add_therapy_activity and update_therapy_activity.
  Always, rely on those funcions for temporal overlappings as a consequence of additions or updates.

# TO AVOID
- Call only the tools necessary for the current user request; avoid unnecessary calls.
- Use English only (unless the user asks otherwise).
- Time must be expressed in 24 hours format (e.g. 01:00, 13, 23...) and AVOID TO USE PM and AM with 12 hours format.
- Never show raw JSON or technical data; always respond in natural language.
- Never show day-number mappings; always use day names.
- Never decide conflict resolutions on your own; always consult the caregiver.
- Don't propose medical alternatives or infer pharmacological properties.
"""


_MANAGER_TOOLS = [
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
            "description": """Adds a new activity to the therapy of the current patient.
                Requires: activity_id, name, day_of_week, time, duration_minutes.
                Optional: description, dependencies, valid_from, valid_until""",
            "parameters": {
                "type": "object",
                "properties": {
                    "activity_id": {
                        "type": "string",
                        "description": "Unique ID (e.g.: 'lb_001')",
                    },
                    "name": {"type": "string", "description": "Name of the activity"},
                    "description": {
                        "type": "string",
                        "description": "Detailed description",
                    },
                    "day_of_week": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Days of the week (1=Monday, 7=Sunday)",
                    },
                    "time": {"type": "string", "description": "Time (HH:MM)"},
                    "category": {
                        "type": "string",
                        "description": "Category: medication, outside_activity, meal, health_checkup, therapy, relaxation, social_activity",
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Duration in minutes",
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of dependency activity_ids",
                    },
                    "valid_from": {"type": "string", "description": "YYYY-MM-DD"},
                    "valid_until": {"type": "string", "description": "YYYY-MM-DD"},
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
            "description": "Updates an existing activity. Specify activity_id and only fields that need to change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "activity_id": {
                        "type": "string",
                        "description": "ID of the activity to update",
                    },
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "day_of_week": {"type": "array", "items": {"type": "integer"}},
                    "time": {"type": "string", "description": "HH:MM"},
                    "duration_minutes": {"type": "integer"},
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                    "valid_from": {"type": "string", "description": "YYYY-MM-DD"},
                    "valid_until": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["activity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_therapy_activity",
            "description": "Remove an activity from the therapy of the current patient",
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
            "name": "get_patient_preferences",
            "description": (
                "Retrieve known preferences and habits of the current patient. "
                "Provide an optional 'query' to narrow results to a specific topic; "
                "omit it to retrieve all preferences."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional topic (e.g. 'food', 'morning routine', 'medication timing').",
                    }
                },
                "required": [],
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
            "description": "Saves the current therapy session. Call when the user says they are done.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


class TherapyManagerAgent(Agent):
    def __init__(self, agent_name="therapy_manager"):
        super().__init__(
            agent_name=agent_name,
            agent_prompt=_PROMPT,
            agent_tools=_MANAGER_TOOLS,  # check_activity aggiunto dall'orchestratore
        )

    def inject_context(self):
        self.conversation_history.append(
            {
                "role": "tool",
                "content": f"get_current_datetime:{datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')}",
            }
        )

        therapy_json = tools.get_all_activities()
        self.conversation_history.append(
            {
                "role": "tool",
                "content": f"get_therapy_activities:{therapy_json}",
            }
        )

    def execute_tool(self, tool_name: str, tool_arguments: dict) -> str:
        if tool_name == "get_current_datetime":
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")

        if tool_name == "get_therapy_activities":
            return tools.get_all_activities()

        if tool_name == "add_therapy_activity":
            return tools.add_therapy_activity(tool_arguments)

        if tool_name == "update_therapy_activity":
            activity_id = tool_arguments.get("activity_id")
            updates = tool_arguments.get("updates") or {
                k: v for k, v in tool_arguments.items() if k != "activity_id"
            }
            return tools.update_therapy_activity(activity_id, updates)

        if tool_name == "remove_therapy_activity":
            return tools.remove_therapy_activity(tool_arguments.get("activity_id"))

        if tool_name == "get_patient_preferences":
            return tools.get_patient_preferences(tool_arguments.get("query", ""))

        if tool_name == "get_conflict_resolution_hints":
            query = tool_arguments.get("query", "")
            return tools.get_conflict_resolution_hints(query)

        return super().execute_tool(tool_name, tool_arguments)
