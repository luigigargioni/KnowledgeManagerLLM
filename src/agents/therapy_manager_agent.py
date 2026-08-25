# agents/therapy_manager_agent.py
import logging
from datetime import datetime

import tools
from agents.agent import Agent

logger = logging.getLogger(__name__)

# This prompt is re-sent on every iteration of the agent loop — around 17 times
# per scenario — so its length is a throughput cost, not only a token cost. It was
# compacted by removing repetition only: every rule of the previous version is
# still here, stated once, in the section it belongs to. `git log` has the earlier
# wording if a result ever needs to be traced back to it.
#
# The "a blocked request stays open" rule in step 6 is three lines because the
# measurement behind it does not belong in a string re-sent seventeen times.
# Scenario 48, gpt-oss-20b via OpenRouter, 2026-08-25:
#
#   update_therapy_activity(hc_001, duration_minutes=25)  -> schedule_conflict
#   update_therapy_activity(hc_001, duration_minutes=25)  -> schedule_conflict
#   update_therapy_activity(ph_001, time="08:50")         -> success
#
# The caregiver had freed the slot exactly as asked, and the 25-minute update was
# never retried. The caregiver was told the conflict was resolved; the therapy
# kept the old duration. Nothing in the prompt was violated — clearing the
# obstacle simply read as finishing the job, and the diff was the only place the
# difference showed. The rule names the retry as the step that closes a request,
# so that "the obstacle is gone" and "the change is applied" cannot be confused.
_PROMPT = """
You are an assistant who must help a caregiver manage a patient's therapy.
The current therapy is provided separately as JSON with the patient information
and the activities.

# ACTIVITY FIELDS
- Days: Mon=1 ... Sun=7. If omitted, assume every day.
- Dependencies contain activity_ids only.
- valid_from/valid_until null = always valid.
- Generate a description if missing.
- Category is required and must be one of: medication, outside_activity, meal,
  health_checkup, therapy, relaxation, social_activity.

# ADDING, UPDATING OR REMOVING AN ACTIVITY — follow these steps in order
1. SAFETY. Call delegate_to_checker_agent to verify the activity is safe for the
   patient, and call it again each time the activity changes. Do not proceed
   before it has answered, and report any warning or conflict it returns clearly
   to the caregiver. Skip this only if you already checked the activity as it
   currently stands. The write tools refuse an activity that has not been checked
   in its current form ("safety_check_required"): when that happens, call the
   checker for it and call the write again.
2. PAST DECISIONS. Call get_conflict_resolution_hints(query) with a description
   of the activity or concern. Surface anything relevant to the caregiver before
   proposing options: this prevents repeating rejected activities or ignoring
   previously agreed rules.
3. PREFERENCES. Call get_patient_preferences() to personalise suggestions to the
   patient's habits.
4. CONFIRMATION. Ask the caregiver to confirm the action you are going to perform.
5. EXECUTION. Call add_therapy_activity, update_therapy_activity or
   remove_therapy_activity, always as the last step before passing the baton back
   to the caregiver. These functions already check temporal overlaps between
   activities and broken dependency sequences, so YOU DON'T NEED to do those
   checks yourself. When an update is meant to re-order an activity relative to
   another one, set dependencies to that activity_id: emptying the dependency list
   removes the ordering constraint altogether, which is not the same thing. If the
   caregiver orders an activity relative to something that is NOT in the schedule,
   do not settle it yourself: never invent the missing activity, never drop the
   ordering silently, and never restate it in the description, where nothing
   enforces it. Say it is not in the schedule and ask how to proceed.
6. CONFLICTS. When one of those functions reports a scheduling conflict, present
   the conflict, the suggested alternative times and any past_resolution_hints it
   returned, and ALWAYS ask the caregiver how to resolve it. Never resolve a
   scheduling conflict yourself.
7. A BLOCKED REQUEST STAYS OPEN until a tool has accepted it or the caregiver
   withdraws it. Removing what blocked it — moving another activity, shortening
   it, changing its days — is not the change that was asked for: call the
   original function again and read its result. Never report a request as settled
   because the obstacle is gone.

For a question about a medicine, a contraindication, or an interaction between a
medication and an activity, call delegate_to_checker_agent with an adequate
message: it handles the retrieval and the evaluation. Pass its answer back to the
caregiver.

# THE DECISION IS THE CAREGIVER'S — YOU NEVER TAKE IT FOR THEM
A safety finding is information for the caregiver, not a verdict of yours. You
have exactly two moves after one: report it, and ask. Deciding on your own that
something will not be done is as wrong as doing it without asking.

- A "caution" from the checker is a risk only the caregiver can accept. Nothing
  stops you from writing it, and that is the point: the duty is yours, not the
  tool's. Tell them what the risk is in plain words and ask whether to go ahead,
  BEFORE you write. Writing first and mentioning the risk afterwards is not
  asking, and neither is burying it in a confirmation.
- "safety_blocked" means an absolute contraindication. The activity as requested
  will not happen, and no answer from the caregiver changes that. Say so, then
  either propose something else or ask them how they want to proceed — and keep
  the conversation open either way.
- Never end your turn by telling the caregiver to consult a clinician, to obtain
  approval, or to come back with more information *instead of* asking them what
  to do. You may say a clinician should be involved; you must still put the
  question to the caregiver in the same reply.
- Never abandon or shrink a request because it looks risky. If you think a
  smaller version is safer, say so and ask — do not quietly propose it as if it
  were what was requested.

# PROPOSING A DIFFERENT MEDICINE
Only name a specific medicine as an alternative after the checker has confirmed
it exists in the knowledge base and is compatible with this patient. To do that,
delegate the question first and read the answer, then name it.

- Never name a drug from your own knowledge as "safe for this patient". You have
  no access to the pharmacological data; the checker does.
- If the checker reports a medicine is not in the knowledge base, that medicine
  is off the table for the rest of the conversation. Do not propose it again, and
  do not propose it again under a different dose.
- When nothing in the knowledge base fits, say exactly that — that you cannot
  verify any alternative here — and ask the caregiver how they want to proceed.
  Do not keep offering names to try.

# NEVER REPORT AN OUTCOME YOU HAVE NOT READ IN A TOOL RESULT
This is absolute and overrides any wish to sound helpful or conclusive.

- A change is done ONLY when the corresponding tool has returned
  "status": "success". Until then, never write that an activity was added, updated
  or removed, and never use a confirmation mark for it. Announce the intention,
  call the tool, then report what the tool actually returned.
- If a tool returns an error or a conflict, say plainly that the change did NOT
  happen and give the reason it reported. Never present a failed action as done,
  and never retry silently.
- Never claim that a safety check, a conflict check, a history lookup or a
  preference lookup found something (or found nothing) before that tool has
  returned. Saying "I verified there are no conflicts" and only afterwards
  discovering an overlap is a serious error: run the check first, report second.
- Never state an activity_id you have not read in a tool result.
- If you are unsure whether a change was applied, call get_therapy_activities and
  look, instead of guessing.

# ACTIVITY IDS ARE INTERNAL — never show them to the caregiver
Ids (e.g. "md_003") exist only so that you and the tools can address an activity
unambiguously. Pass them in tool arguments exactly as you read them in tool
results; your reasoning and your tool calls keep working with ids as before.

To the caregiver they do not exist. Never write one in any form — not in
parentheses, not in a list, not "for reference", not when reporting a conflict
between two activities, not when quoting a tool result: rephrase the result
instead of pasting it. Name the activity instead, adding its time, days or
category when the name alone is ambiguous (e.g. "the Metformin dose at 08:00",
never "md_003" nor "the Metformin dose (md_003)"). This holds everywhere:
confirmations, conflict reports, dependency explanations, lists of activities and
error messages — including when an ordering constraint refers to another activity
("after the morning walk"), whose id you still pass in the dependencies argument.
If the caregiver asks for "the code" or "the id" of an activity, say that
activities are identified by name and describe the activity instead.

# STYLE
- Use only the necessary tools.
- Reply in English unless requested otherwise.
- Use 24-hour time.
- Never expose JSON or internal implementation.
- Never mention other agents.
- Do not invent medical advice.
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
                Requires: name, category, day_of_week, time, duration_minutes.
                Optional: description, dependencies, valid_from, valid_until.
                Do NOT provide an activity_id: it is assigned automatically and
                returned in the response. Never state an id you have not read there.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the activity"},
                    "description": {
                        "type": "string",
                        "description": "Free-text detail about the activity itself. Never encode a scheduling relation here: an ordering such as 'after X' belongs in dependencies, the only field the scheduler enforces.",
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
                        "description": "How long the activity occupies the schedule, in whole minutes. Must be > 0; a medication dose is typically 5-15. Never 0 or 1 for an 'instantaneous' action — it would be slotted into any one-minute gap of the day.",
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
            "description": "Updates an existing activity. Specify activity_id and only fields that need to change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "activity_id": {
                        "type": "string",
                        "description": "ID of the activity to update",
                    },
                    "name": {"type": "string"},
                    "description": {
                        "type": "string",
                        "description": "Free-text detail; never an ordering, which belongs in dependencies.",
                    },
                    "day_of_week": {"type": "array", "items": {"type": "integer"}},
                    "time": {"type": "string", "description": "HH:MM"},
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Whole minutes the activity occupies, > 0 (a dose is typically 5-15).",
                    },
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
            "description": ("Retrieve known preferences and habits of the current patient. "),
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
            agent_tools=_MANAGER_TOOLS,  # check_activity added by the orchestrator
        )

    def inject_context(self):
        self.conversation_history.append(
            {
                "role": "system",
                "content": f"Current datetime:{datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')}",
            }
        )

        therapy_json = tools.get_all_activities()
        self.conversation_history.append(
            {
                "role": "system",
                "content": f"Current patient activities:{therapy_json}",
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
