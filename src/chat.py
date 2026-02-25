import json
import logging
from datetime import datetime

from openai import APIConnectionError, APITimeoutError, OpenAI

import prompts as prompts
import tools as tools
from config_loader import (
    LLM_PROVIDER,
    LLM_TIMEOUT,
    MODEL,
    OLLAMA_URL,
    OPENAI_API_KEY,
)
from session_extractor import (
    extract_and_save_conflict_resolutions,
    extract_and_save_patient_preferences,
)
from sql_db import DatabaseManager

logger = logging.getLogger("knowledge_manager")


def _make_client() -> OpenAI:
    """Return an OpenAI-compatible client for the configured provider."""
    if LLM_PROVIDER == "openai":
        return OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT)
    # Ollama exposes an OpenAI-compatible API at /v1
    return OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama", timeout=LLM_TIMEOUT)


def build_first_message(therapy_json):
    therapy = json.loads(therapy_json)
    # Support both key names for robustness
    patient_name = therapy.get("patient_full_name") or therapy.get(
        "patient_name", "Unknown"
    )
    first_message = (
        f"Hi I'm your therapy management assistant!  \n"
        f"The current patient is **{patient_name}**. "
        # f"The activities of {patient_name}'s therapy are:  \n"
        f"The activities of {patient_name}'s therapy are reported in left panel.\n\n"
    )

    # Commented out to avoid confusion with the therapy activities that are now displayed in the left panel.
    """
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
            line = f"- {act['name']}   -  {act['time']}  -  {', '.join(days_map[d] for d in act.get('day_of_week', []))}"
            valid_from = act.get("valid_from")
            valid_until = act.get("valid_until")
            if valid_from or valid_until:
                line += f"  (valid: {valid_from or '…'} → {valid_until or '…'})"
            first_message += line + "  \n"
        first_message += "\n"

    if len(therapy.get("expired_activities", [])) > 0:
        first_message += "The activities that are **not valid anymore** are:  \n"
        for inv_act in therapy.get("expired_activities", []):
            first_message += f"- {inv_act['time']} {inv_act['name']}  -  Valid until: {inv_act['valid_until']}  \n"
        first_message += "\n"
    """

    first_message += "I can help you add new activity, change the the current activities or remove the one that are not necessary. What do you want to do?"

    return first_message


class Chat:
    def __init__(
        self,
        model=MODEL,
        system_prompt=None,
        database_manager: DatabaseManager = None,
        vector_db=None,
    ):
        """
        Initialise the LLM client.
        Supports both OpenAI cloud and Ollama (auto-detected from OPENAI_API_KEY).

        Args:
            model: Model name to use
            system_prompt: System prompt to configure the model behaviour
            database_manager: DatabaseManager instance for session persistence
            vector_db: VectorDBManager instance for RAG features
        """
        self.model = model
        self.client = _make_client()
        self.conversation_history = []
        self.session_ended = False

        self.database_manager = database_manager
        self.vector_db = vector_db

        # Inject the vector DB into the tools module so all tool functions can use it
        if vector_db is not None:
            tools.set_vector_db(vector_db)
            logger.debug("[INIT] Vector DB injected into tools module")

        logger.info(f"[INIT] Provider={LLM_PROVIDER} Model={model}")

        self.tools = tools.tools_decl
        logger.info(f"[INIT] Loaded {len(self.tools)} tools")

        if system_prompt:
            self.conversation_history.append(
                {"role": "system", "content": system_prompt}
            )
            logger.info(
                f"[INIT] System prompt configured: {len(system_prompt)} characters"
            )
        else:
            logger.info("[INIT] No system prompt provided")

        # Initialization of useful data
        self.conversation_history.append(
            {
                "role": "system",
                "content": f"get_current_datetime:{datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')}",
            }
        )
        therapy_json = tools.get_all_activities()
        self.conversation_history.append(
            {
                "role": "system",
                "content": f"get_all_activities:{therapy_json}",
            }
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

        # THERAPY MANAGEMENT TOOLS
        elif tool_name == "get_therapy_activities":
            return tools.get_all_activities()

        elif tool_name == "add_therapy_activity":
            return tools.add_therapy_activity(tool_arguments)

        elif tool_name == "update_therapy_activity":
            activity_id = tool_arguments.get("activity_id")
            # Support both flat args (OpenAI style) and nested {updates: {...}} (legacy)
            updates = tool_arguments.get("updates") or {
                k: v for k, v in tool_arguments.items() if k != "activity_id"
            }
            return tools.update_therapy_activity(activity_id, updates)

        elif tool_name == "remove_therapy_activity":
            activity_id = tool_arguments.get("activity_id")
            return tools.remove_therapy_activity(activity_id)

        elif tool_name == "get_medicine_data":
            medicine_name = tool_arguments.get("medicine_name")
            return tools.get_medicine_data(medicine_name)

        elif tool_name == "get_patient_preferences":
            query = tool_arguments.get("query", "")
            return tools.get_patient_preferences(query)

        elif tool_name == "get_patient_history_events":
            query = tool_arguments.get("query", "")
            return tools.get_patient_history_events(query)

        elif tool_name == "get_conflict_resolution_hints":
            query = tool_arguments.get("query", "")
            return tools.get_conflict_resolution_hints(query)

        elif tool_name == "save_session":
            logger.info(
                "[TOOL] save_session triggered by LLM – running full end_session flow"
            )
            result = self.end_session()
            return json.dumps(result, ensure_ascii=False)

        else:
            logger.warning(f"[TOOL] Tool not found: {tool_name}")
            return f"Tool '{tool_name}' not found"

    def _normalize_messages(self) -> list[dict]:
        """
        Return a copy of the conversation history suitable for the OpenAI API.

        Rules applied:
        1. role=tool messages WITHOUT tool_call_id (init context injections) are
           converted to role=system so OpenAI accepts them.
        2. Any role=assistant message that appears before the first role=user is
           also converted to role=system (OpenAI requires conversations to start
           with a user turn; Ollama is more lenient but OpenAI is not).
        """
        # Determine the index of the first user message
        first_user_idx = next(
            (
                i
                for i, m in enumerate(self.conversation_history)
                if m.get("role") == "user"
            ),
            len(self.conversation_history),
        )

        normalized = []
        for i, msg in enumerate(self.conversation_history):
            role = msg.get("role")
            # Pre-conversation context: tool msgs without tool_call_id → system
            if role == "tool" and "tool_call_id" not in msg:
                normalized.append(
                    {"role": "system", "content": f"[Context] {msg['content']}"}
                )
            # Pre-conversation assistant msg (e.g. the welcome message) → system
            elif role == "assistant" and i < first_user_idx:
                normalized.append(
                    {"role": "system", "content": f"[Assistant intro] {msg['content']}"}
                )
            else:
                normalized.append(msg)
        return normalized

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

        # LLM can call at most n tools in a row
        max_iterations = 10

        for _ in range(max_iterations):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self._normalize_messages(),
                    tools=self.tools,
                )

                message = response.choices[0].message
                tool_calls = message.tool_calls

                if tool_calls:
                    logger.debug(f"[TOOL] {len(tool_calls)} tool call(s) requested")

                    # Record the assistant turn with its tool_calls
                    self.conversation_history.append(
                        {
                            "role": "assistant",
                            "content": message.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in tool_calls
                            ],
                        }
                    )

                    for tool_call in tool_calls:
                        tool_name = tool_call.function.name
                        # OpenAI (and Ollama /v1) always returns arguments as a JSON string
                        tool_arguments = json.loads(tool_call.function.arguments)

                        logger.debug(f"[TOOL] Executing: {tool_name}({tool_arguments})")

                        tool_result = self.execute_tool(tool_name, tool_arguments)
                        result_preview = (
                            str(tool_result)[:100] + "..."
                            if len(str(tool_result)) > 100
                            else str(tool_result)
                        )
                        logger.debug(f"[TOOL] Result: {result_preview}")

                        self.conversation_history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": str(tool_result),
                            }
                        )

                else:
                    assistant_message = message.content or "No response received"
                    self.conversation_history.append(
                        {"role": "assistant", "content": assistant_message}
                    )
                    return assistant_message

            except APIConnectionError:
                logger.error(
                    f"[ERROR] Connection failed: Cannot connect to {LLM_PROVIDER} API"
                )
                return None
            except APITimeoutError:
                logger.error("[ERROR] Request timeout")
                return None
            except Exception as e:
                logger.error(f"[ERROR] Request failed: {e}")
                return None

        logger.warning("[WARNING] Max iterations reached for tool calling")
        error_msg = "I'm sorry, I could not complete the request because too many operations were needed. Please try rephrasing or breaking the request into smaller steps."
        self.conversation_history.append({"role": "assistant", "content": error_msg})
        return error_msg

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

            n_conflicts = extract_and_save_conflict_resolutions(
                self.conversation_history, self.vector_db, patient_id
            )
            n_prefs = extract_and_save_patient_preferences(
                self.conversation_history, self.vector_db, patient_id
            )
        else:
            logger.warning(
                "[SESSION] Vector DB not available – skipping knowledge extraction"
            )

        # ── PostgreSQL save ────────────────────────────────────────────────
        if self.database_manager:
            logger.info("[SESSION] Persisting therapy to PostgreSQL")
            result = self.database_manager.save_session()
            if result.get("status") == "success":
                v_id = result.get("version", {}).get("id")
                # logger.info(
                #    f"[SESSION] Therapy persisted to PostgreSQL – version #{v_id}"
                # ) REDUNDANT LOG
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


# Backward-compatible alias
OllamaChat = Chat
