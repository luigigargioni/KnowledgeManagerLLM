import json
import logging
from datetime import datetime

import requests

import prompts as prompts
import tools as tools
from config_loader import (
    OLLAMA_URL,
)
from session_extractor import (
    extract_and_save_conflict_resolutions,
    extract_and_save_patient_preferences,
)
from sql_db import DatabaseManager

logger = logging.getLogger("knowledge_manager")


def build_first_message(therapy_json):
    therapy = json.loads(therapy_json)
    # Support both key names for robustness
    patient_name = therapy.get("patient_full_name") or therapy.get(
        "patient_name", "Unknown"
    )
    first_message = (
        f"Hi I'm your therapy management assistant!  \n"
        f"The current patient is **{patient_name}**. "
        f"The activities of {patient_name}'s therapy are:  \n"
    )

    days_map = {
        1: "Mon",
        2: "Tue",
        3: "Wed",
        4: "Thu",
        5: "Fri",
        6: "Sat",
        7: "Sun",
    }

    if therapy.get("activities") is None or len(therapy.get("activities", [])) == 0:
        first_message += "\n *No activities found for this patient*.  \n\n"
    else:
        for act in therapy["activities"]:
            first_message += f"- {act['name']}   -  {act['time']}  -  {', '.join(days_map[d] for d in act.get('day_of_week', []))} \n"
        first_message += "\n"

    if len(therapy.get("expired_activities", [])) > 0:
        first_message += "The activities that are **not valid anymore** are:  \n"
        for inv_act in therapy.get("expired_activities", []):
            first_message += f"- {inv_act['time']} {inv_act['name']}  -  Valid until: {inv_act['valid_until']}  \n"
        first_message += "\n"
    first_message += "I can help you add new activity, change the the current activities or remove the one that are not necessary. What do you want to do?"

    return first_message


class OllamaChat:
    def __init__(
        self,
        model="qwen2.5:14b",
        base_url=OLLAMA_URL,
        system_prompt=None,
        database_manager: DatabaseManager = None,
        vector_db=None,
    ):
        """
        Initialise the Ollama client.

        Args:
            model: Model name to use (default: qwen2.5:14b)
            base_url: Base URL of the Ollama API (default: http://localhost:11434)
            system_prompt: System prompt to configure the model behaviour
            database_manager: DatabaseManager instance for session persistence
            vector_db: VectorDBManager instance for RAG features
        """
        self.model = model
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/chat"
        self.conversation_history = []
        self.session_ended = False

        self.database_manager = database_manager
        self.vector_db = vector_db

        # Inject the vector DB into the tools module so all tool functions can use it
        if vector_db is not None:
            tools.set_vector_db(vector_db)
            logger.debug("[INIT] Vector DB injected into tools module")

        logger.info(f"[INIT] Model {model}")

        self.tools = tools.tools_decl
        logger.info(f"[INIT] Loaded {len(self.tools)} tools")

        if system_prompt:
            self.conversation_history.append(
                {"role": "system", "content": system_prompt}
            )
            logger.info(
                f"[INIT] System prompt configured: {len(system_prompt)} characters"
            )
            logger.debug(f"[SYSTEM_PROMPT] {system_prompt}")
        else:
            logger.info("[INIT] No system prompt provided")

        # Initialization of useful data
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
                "content": f"get_all_activities:{therapy_json}",
            }
        )

        # Inject patient preferences as initial context
        if vector_db is not None:
            patient_id = tools._get_patient_id()
            prefs = vector_db.query_patient_preferences(patient_id)
            if prefs:
                self.conversation_history.append(
                    {
                        "role": "tool",
                        "content": (
                            "get_patient_preferences:"
                            + json.dumps(
                                {
                                    "status": "success",
                                    "patient_id": patient_id,
                                    "preferences": prefs,
                                },
                                ensure_ascii=False,
                            )
                        ),
                    }
                )
                logger.info(
                    f"[INIT] Injected {len(prefs)} patient preference(s) into initial context"
                )

        first_message = build_first_message(therapy_json)
        self.conversation_history.append(
            {"role": "assistant", "content": first_message}
        )

    def execute_tool(self, tool_name, tool_arguments):
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            tool_arguments: Tool arguments (dict)

        Returns:
            str: Result of the tool execution
        """

        if tool_name == "get_current_datetime":
            now = datetime.now()
            result = now.strftime("%Y-%m-%d %H:%M:%S %A")
            return result

        elif tool_name == "get_devices":
            result = tools.get_devices()
            return result

        elif tool_name == "clear_conversation_history":
            keep_system = tool_arguments.get("keep_system_prompt", True)
            return tools.clear_conversation_history(self, keep_system)

        # THERAPY MANAGEMENT TOOLS
        elif tool_name == "get_therapy_activities":
            return tools.get_all_activities()

        elif tool_name == "add_therapy_activity":
            return tools.add_therapy_activity(tool_arguments)

        elif tool_name == "update_therapy_activity":
            activity_id = tool_arguments.get("activity_id")
            updates = tool_arguments.get("updates", {})
            return tools.update_therapy_activity(activity_id, updates)

        elif tool_name == "remove_therapy_activity":
            activity_id = tool_arguments.get("activity_id")
            return tools.remove_therapy_activity(activity_id)

        elif tool_name == "get_medicine_data":
            medicine_name = tool_arguments.get("medicine_name")
            return tools.get_medicine_data(medicine_name)

        elif tool_name == "get_patient_preferences":
            return tools.get_patient_preferences()

        elif tool_name == "save_session":
            logger.info(
                "[TOOL] save_session triggered by LLM – running full end_session flow"
            )
            result = self.end_session()
            return json.dumps(result, ensure_ascii=False)

        else:
            logger.warning(f"[TOOL] Tool not found: {tool_name}")
            return f"Tool '{tool_name}' not found"

    def send_message(self, user_message):
        """
        Send a message to the model, maintaining conversation context.
        Automatically handles tool calls.

        Args:
            user_message: The user message to send

        Returns:
            str: The model response, or None on error
        """
        logger.info(f"[CHAT] USER: {user_message}")

        self.conversation_history.append({"role": "user", "content": user_message})

        # LLM can call at most 5 tools in a row
        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            payload = {
                "model": self.model,
                "messages": self.conversation_history,
                "tools": self.tools,
                "stream": False,
            }

            try:
                response = requests.post(self.api_endpoint, json=payload, timeout=120)

                response.raise_for_status()

                result = response.json()
                message = result.get("message", {})

                tool_calls = message.get("tool_calls")

                if tool_calls:
                    # Model requires tool calling
                    logger.debug(f"[TOOL] {len(tool_calls)} tool call(s) requested")
                    self.conversation_history.append(message)

                    for tool_call in tool_calls:
                        function = tool_call.get("function", {})
                        tool_name = function.get("name")
                        tool_arguments = function.get("arguments", {})

                        logger.debug(f"[TOOL] Executing: {tool_name}({tool_arguments})")

                        # Running the tool
                        tool_result = self.execute_tool(tool_name, tool_arguments)
                        result_preview = (
                            str(tool_result)[:100] + "..."
                            if len(str(tool_result)) > 100
                            else str(tool_result)
                        )
                        logger.debug(f"[TOOL] Result: {result_preview}")

                        self.conversation_history.append(
                            {"role": "tool", "content": tool_result}
                        )

                    # continue #this continue can be used if the llm needs to call multiple tools in sequence. In the case in which it needs a result first to call another  one

                else:
                    assistant_message = message.get("content", "No response received")
                    self.conversation_history.append(
                        {"role": "assistant", "content": assistant_message}
                    )

                    return assistant_message

            except requests.exceptions.ConnectionError:
                logger.error("[ERROR] Connection failed: Cannot connect to Ollama")
                return None
            except requests.exceptions.Timeout:
                logger.error("[ERROR] Request timeout")
                return None
            except requests.exceptions.RequestException as e:
                logger.error(f"[ERROR] Request failed: {str(e)}")
                return None
            except json.JSONDecodeError:
                logger.error("[ERROR] Invalid JSON response from server")
                return None

        logger.warning("[WARNING] Max iterations reached for tool calling")
        return "Error: too many tool calls"

    def get_history(self):
        return self.conversation_history

    def end_session(self) -> dict:
        """
        Perform full end-of-session processing:
        1. Extract conflict resolutions from the conversation and persist to ChromaDB.
        2. Extract patient preferences from the conversation and upsert to ChromaDB.
        3. Save the therapy session to the PostgreSQL database.
        4. Mark the session as ended (self.session_ended = True).

        Idempotent: if already ended, returns immediately.
        Returns the save_session result dict.
        """
        if self.session_ended:
            logger.warning(
                "[SESSION] end_session called but session is already ended – skipping"
            )
            return {"status": "skipped", "message": "Session already ended"}

        logger.info("[SESSION] Starting end-of-session processing")

        # ── Vector DB extraction ────────────────────────────────────────────
        if self.vector_db is not None:
            patient_id = tools._get_patient_id()
            logger.info(
                f"[SESSION] Running vector DB extraction for patient {patient_id}"
            )

            logger.debug(
                "[SESSION] Extracting conflict resolutions from conversation history"
            )
            n_conflicts = extract_and_save_conflict_resolutions(
                self.conversation_history, self.vector_db, patient_id
            )
            logger.info(f"[SESSION] Conflict resolutions saved: {n_conflicts}")

            logger.debug(
                "[SESSION] Extracting patient preferences from conversation history"
            )
            n_prefs = extract_and_save_patient_preferences(
                self.conversation_history, self.vector_db, patient_id
            )
            logger.info(f"[SESSION] Patient preferences saved/updated: {n_prefs}")
        else:
            logger.warning(
                "[SESSION] Vector DB not available – skipping knowledge extraction"
            )

        # ── PostgreSQL save ────────────────────────────────────────────────
        if self.database_manager:
            logger.debug("[SESSION] Persisting therapy to PostgreSQL")
            result = self.database_manager.save_session()
            if result.get("status") == "success":
                v_id = result.get("version", {}).get("id")
                logger.info(
                    f"[SESSION] Therapy persisted to PostgreSQL – version #{v_id}"
                )
            else:
                logger.error(
                    f"[SESSION] PostgreSQL save failed: {result.get('message')}"
                )
        else:
            logger.warning(
                "[SESSION] No database manager – therapy not persisted to PostgreSQL"
            )
            result = {"status": "skipped", "message": "No database manager available"}

        # ── Mark session as ended ────────────────────────────────────────────
        self.session_ended = True
        logger.info("[SESSION] Session marked as ended")

        return result
