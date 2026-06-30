# agents/therapy_check_agent.py
import json

import tools
from agents.agent import Agent

_PROMPT = """You are a specialist in analysing interactions between activites, medications and patients' health conditions.
Your task: checking if an activity in a therapy plan is safe for the patients according to their medical conditions and already present activities.
You will receive a therapy action in the following form:
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
}
and you need to find all the possible conflicts with the current therapy.

# TOOLS
- get_therapy_activities: get all therapy activities.
- get_medicine_data: get pharmacological data for a medicine via semantic search.
  ALWAYS call this before any medicine-related activity.
- get_patient_preferences: retrieve the patient's known preferences and habits.
  Use this to personalise suggestions (timing, food, activity type).
- get_patient_history_events: retrieve past danger/warning events for the patient
  semantically related to the activity being considered.
  ALWAYS call this before proposing or adding any activity.

# WHAT TO DO
1. MEDICINE CHECK
   If the activity involves a medicine call get_medicine_data(medicine_name) first.
   - If data IS returned: verify compatibility with the patient's medical_conditions
   (contraindications, interactions, dosage restrictions).
   - If NO data is returned or the medicine is not found in the local database:
   DO NOT proceed. Inform the caregiver that the medicine is not in the local
   knowledge base and ask them to verify contraindications manually before continuing.
   NEVER infer or hypothesise pharmacological properties for medicines not found
   in the database.

2. ACTIVITY CHECK
  If the activity does not include medicines do check if the currently taken medications have some counterindications with the activity. (e.g. running vs a medication that leaves the patient fatigued).
  Inform the caregiver about all the possible negative interactions

 3. PATIENT HISTORY CHECK (proactive)
   Call get_patient_history_events(query) with a description of the activity.
   - event_type "danger": clearly present to the caregiver and ask for explicit confirmation before proceeding.
   - event_type "warning": mention but not blocking.
   calling it here ensures the caregiver is informed BEFORE you ask for confirmation.

4. RESULT COMPUTATION (mandatory)
  Analize all the data retrieved and check for possible conflicts. Produce an answer in the following format:
  {
  "activity_name":[NAME],
  "check_result":[List of problems or "NO_CONFLICTS"]
  }


# TO AVOID
- Call only the tools necessary for the current user request; avoid unnecessary calls.
- Use English only (unless the user asks otherwise).
- Never show raw JSON or technical data; always respond in natural language.
"""


_CHECK_TOOLS = [
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
                "Provide an optional 'query' to narrow results to a specific topic; "
                "omit it to retrieve all preferences."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional topic to search within the patient's preferences.",
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
                        "description": "Description of the activity or topic to look up in the patient's safety history.",
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
                        "description": "Description of the activity or conflict to look up in past resolution records.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

_CHECK_ACTIVITY_PARAMETERS = {
    "type": "object",
    "properties": {
        "activity_id": {
            "type": "string",
            "description": "Unique ID of the activity (e.g.: 'lb_001')",
        },
        "name": {"type": "string", "description": "Name of the activity"},
        "description": {
            "type": "string",
            "description": "Detailed description of the activity",
        },
        "day_of_week": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Days of the week (1=Monday, 7=Sunday)",
        },
        "time": {
            "type": "string",
            "description": "Time the activity takes place (HH:MM)",
        },
        "category": {
            "type": "string",
            "description": "Category of the activity.",
        },
        "duration_minutes": {
            "type": "integer",
            "description": "Duration of the activity in minutes",
        },
        "dependencies": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of dependency activity_ids",
        },
        "valid_from": {"type": "string", "description": "Valid from date (YYYY-MM-DD)"},
        "valid_until": {
            "type": "string",
            "description": "Valid until date (YYYY-MM-DD)",
        },
    },
    "required": ["activity_id", "name", "day_of_week", "time", "duration_minutes"],
}


class TherapyCheckAgent(Agent):
    def __init__(self, agent_name="checker_agent", zero_shot=False):
        # I tool sono fissi per questo agente, non passati dall'esterno
        super().__init__(
            agent_name=agent_name,
            agent_prompt=_PROMPT,
            agent_tools=_CHECK_TOOLS,
            zero_shot=zero_shot,
        )

    def inject_context(self):
        therapy_json = tools.get_all_activities()
        self.conversation_history.append(
            {
                "role": "tool",
                "content": f"get_therapy_activities:{therapy_json}",
            }
        )

        medications = []
        for activity in json.loads(therapy_json).get("activities", []):
            if activity.get("category") == "medication":
                med = tools.get_medicine_data(activity["name"])
                if med not in medications:
                    medications.append(med)

        self.conversation_history.append(
            {
                "role": "tool",
                "content": f"current_medicine_data:{json.dumps(medications)}",
            }
        )

    def reset_agent(self):
        super().reset_agent()
        self.inject_context()

    def execute_tool(self, tool_name: str, tool_arguments: dict) -> str:
        if tool_name == "get_therapy_activities":
            return tools.get_all_activities()

        if tool_name == "get_medicine_data":
            return tools.get_medicine_data(tool_arguments.get("medicine_name", ""))

        if tool_name == "get_patient_preferences":
            return tools.get_patient_preferences(tool_arguments.get("query", ""))

        if tool_name == "get_patient_history_events":
            return tools.get_patient_history_events(tool_arguments.get("query", ""))

        if tool_name == "get_conflict_resolution_hints":
            return tools.get_conflict_resolution_hints(tool_arguments.get("query", ""))

        return super().execute_tool(tool_name, tool_arguments)
