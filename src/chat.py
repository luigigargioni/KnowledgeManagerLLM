import json
import logging
from time import time

from openai import OpenAI

import prompts as prompts
import tools as tools
from agents.agent import Agent
from agents.check_agent import TherapyCheckAgent
from agents.therapy_manager_agent import TherapyManagerAgent
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
from utils import addAgentFilterLogger

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
        self.session_ended = False
        self.database_manager = database_manager
        self.vector_db = vector_db
        self.conversation_history = []

        # Inject the vector DB into the tools module so all tool functions can use it
        if vector_db is not None:
            tools.set_vector_db(vector_db)
            logger.debug("[INIT] Vector DB injected into tools module")

        # Agents creation
        self.check_agent = TherapyCheckAgent(zero_shot=False)
        addAgentFilterLogger(self.check_agent.name)

        self.chat_agent = TherapyManagerAgent()
        addAgentFilterLogger(self.chat_agent.name)

        # Agents association to the supervisor, needed for delegation
        self._agent_registry: dict[str, Agent] = {
            f"delegate_to_{self.check_agent.name}": self.check_agent,
        }

        # By adding the check_agent to the tools of chat_agente the second can delegate requests
        self.chat_agent.tools.append(
            self.check_agent.as_tool_declaration(
                description=(
                    "Delegate the action to the checker_agent to check it against the patient therapy or get medication information "
                )
            )
        )

        self.tools = self.chat_agent.tools

        logger.info(f"[INIT] Provider={LLM_PROVIDER} Model={model}")

        first_message = build_first_message("{}")
        self.chat_agent.conversation_history.append(
            {"role": "assistant", "content": first_message}
        )

    def execute_tool(self, agent: Agent, tool_name: str, tool_arguments: dict) -> str:
        """
        L'orchestratore gestisce solo:
        1. Delegation ai worker agents
        2. save_session (richiede db_manager e vector_db)
        Tutto il resto è delegato al chat_agent.
        """
        logger.debug(
            f"[{agent.name.upper()}][TOOL] Executing: {tool_name}({tool_arguments})"
        )

        # 1. Delegation
        if tool_name in self._agent_registry:
            agent_delegate = self._agent_registry[tool_name]
            result = self._send_to_agent(agent_delegate, tool_arguments)

        # 2. save_session: richiede dipendenze dell'orchestratore
        elif tool_name == "save_session":
            result = json.dumps(self.end_session(), ensure_ascii=False)

        # 3. Tool del supervisor
        else:
            result = agent.execute_tool(tool_name, json.loads(tool_arguments))

        logger.debug(f"[{agent.name.upper()}][TOOL] Results of {tool_name}: {result}")
        return result

    def _run_agent_loop(self, agent: Agent, user_message: str) -> str:
        """
        Loop tool-calling generico per qualsiasi agente.
        Usato sia dal supervisor (send_message) che per la delegation (_send_to_agent).
        """

        logger.debug(f"[{agent.name.upper()}][REQUEST] {user_message}")
        agent.conversation_history.append({"role": "user", "content": user_message})

        for _ in range(10):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=agent.conversation_history,
                tools=agent.tools,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                reply = msg.content or ""
                if agent.zero_shot:
                    agent.reset_agent()
                else:
                    agent.conversation_history.append(
                        {"role": "assistant", "content": reply}
                    )
                logger.debug(f"[{agent.name.upper()}][REPLY] {reply}")
                return reply

            for tc in msg.tool_calls:
                # Here the chat supervisor decide which agent to call or to close the session
                result = self.execute_tool(
                    agent, tc.function.name, tc.function.arguments
                )
                agent.conversation_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    }
                )

        return "Max iterations reached."

    def send_message(self, user_message: str) -> str:
        """Function used to send a message to the supervisor."""
        logger.info(f"[CHAT] USER: {user_message}")
        start = time()
        res = self._run_agent_loop(self.chat_agent, user_message)
        logger.debug(f"[TIMING] {time() - start:.2f}s")
        logger.info(f"[CHAT] ASSISTANT: {res}")
        return res

    def _send_to_agent(self, agent: Agent, tool_arguments: dict) -> str:
        """Delegation to a worker agent"""
        return self._run_agent_loop(agent, json.dumps(tool_arguments))

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
