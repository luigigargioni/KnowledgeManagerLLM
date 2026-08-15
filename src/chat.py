import json
from pathlib import Path
from time import time

from openai import OpenAI

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
    THERAPY_FILE,
)
from session_extractor import (
    extract_and_save_conflict_resolutions,
    extract_and_save_patient_preferences,
)
from sql_db import DatabaseManager
from utils import addAgentFilterLogger, get_current_logger, visible_turns
from utils import load_past_session as _load

logger = get_current_logger()


def _make_client() -> OpenAI:
    """Return an OpenAI-compatible client for the configured provider."""
    if LLM_PROVIDER == "openai":
        return OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=LLM_TIMEOUT,
        )
    # Ollama exposes an OpenAI-compatible API at /v1
    return OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama", timeout=LLM_TIMEOUT)


def build_first_message(therapy_json):
    therapy = json.loads(therapy_json)
    # Support both key names for robustness
    patient_name = therapy.get("patient_full_name") or therapy.get("patient_name", "Unknown")
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
        self.check_agent = TherapyCheckAgent(zero_shot=True)
        addAgentFilterLogger(self.check_agent.name)

        self.chat_agent = TherapyManagerAgent()
        addAgentFilterLogger(self.chat_agent.name)

        # Agents association to the supervisor, needed for delegation
        self._agent_registry: dict[str, Agent] = {
            f"delegate_to_{self.check_agent.name}": self.check_agent,
        }

        # By adding the check_agent to the tools of chat_agent the latter can delegate requests
        self.chat_agent.tools = self.chat_agent.tools + [
            self.check_agent.as_tool_declaration(
                description=(
                    "Delegate the action to the checker_agent to check it against "
                    "the patient therapy or get medication information"
                )
            )
        ]
        self.tools = self.chat_agent.tools

        first_message = build_first_message(THERAPY_FILE.read_text(encoding="utf-8"))
        self.chat_agent.conversation_history.append({"role": "assistant", "content": first_message})

        self._therapy_snapshots: list[dict] = []
        self._save_therapy_snapshot()

    def _save_therapy_snapshot(self):
        """
        Saves a snapshot of the therapy associated with the chat index.
        Used later for rewind feature.
        """

        current_idx = len(visible_turns(self.chat_agent.conversation_history)) - 1
        therapy = json.loads(THERAPY_FILE.read_text(encoding="utf-8"))

        self._therapy_snapshots.append(
            {
                "message_idx": current_idx,
                "therapy": therapy,
            }
        )

        # Saving of snapshot on memory
        if logger.session_dir:
            snapshots_path = logger.session_dir / "therapy_snapshots.json"
            snapshots_path.write_text(
                json.dumps(self._therapy_snapshots, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        logger.debug(
            f"[SNAPSHOT] Saved therapy snapshot at message_idx={current_idx} "
            f"(total snapshots: {len(self._therapy_snapshots)})"
        )

    def restore_therapy_snapshot(self, message_idx: int):
        """
        Restored the first snapshot with message_idx<= given_message_idx.
        Overwrites therapy.json with the new snapshot.
        """

        # Find the last snapshot with idx <= message_idx
        candidates = [s for s in self._therapy_snapshots if s["message_idx"] <= message_idx]
        if not candidates:
            logger.warning(
                f"[SNAPSHOT] No snapshot found for message_idx<={message_idx}, keeping current therapy"
            )
            return

        snapshot = candidates[-1]  # the last (most recent) among the candidates

        # Also truncate the snapshot list: subsequent ones are no longer valid
        self._therapy_snapshots = candidates

        THERAPY_FILE.write_text(
            json.dumps(snapshot["therapy"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            f"[SNAPSHOT] Restored therapy snapshot from message_idx={snapshot['message_idx']}"
        )

    # chat.py

    def load_past_session(self, session_log_dir: Path) -> None:
        """
        Load the context of a previous session:
        - Populate the chat_agent history from chat.log
        - Restore the therapy snapshots
        - Restore therapy.json to the last snapshot found
        - Log the loading in the current session
        """

        logger.info(f"[LOAD] Loading past session from {session_log_dir}")

        snapshots = _load(
            self.chat_agent,
            session_log_dir,
            keep_system_prompt=True,
        )

        if snapshots:
            self._therapy_snapshots = snapshots
            # Restore therapy.json to the last snapshot of the loaded session
            self.restore_therapy_snapshot(snapshots[-1]["message_idx"])
        else:
            # No snapshot: at least save the current state of therapy.json
            self._therapy_snapshots = []
            self._save_therapy_snapshot()

        logger.info(
            f"[LOAD] Past session loaded from: {session_log_dir} – "
            f"{len(self.chat_agent.conversation_history)} messages, "
            f"{len(self._therapy_snapshots)} snapshots"
        )

    def execute_tool(self, agent: Agent, tool_name: str, tool_arguments: dict) -> str:
        """
        The orchestrator only handles:
        1. Delegation to worker agents
        2. save_session (requires db_manager and vector_db)
        Everything else is delegated to the chat_agent.
        """
        logger.debug(f"[{agent.name.upper()}][TOOL] Executing: {tool_name}({tool_arguments})")

        # 1. Delegation
        if tool_name in self._agent_registry:
            agent_delegate = self._agent_registry[tool_name]
            result = self._send_to_agent(agent_delegate, tool_arguments)

        # 2. save_session: requires orchestrator dependencies
        elif tool_name == "save_session":
            result = json.dumps(self.end_session(), ensure_ascii=False)

        # 3. Supervisor tools
        else:
            result = agent.execute_tool(tool_name, json.loads(tool_arguments))

        # If is an action that changes the therapy i save a new snapshot of it
        if tool_name in (
            "add_therapy_activity",
            "update_therapy_activity",
            "remove_therapy_activity",
        ):
            self._save_therapy_snapshot()

        logger.debug(f"[{agent.name.upper()}][TOOL] Results of {tool_name}: {result}")
        return result

    def _run_agent_loop(self, agent: Agent, user_message: str) -> str:
        """
        Generic tool-calling loop for any agent.
        Used by both the supervisor (send_message) and for delegation (_send_to_agent).
        """

        logger.debug(f"[{agent.name.upper()}][REQUEST] {user_message}")
        agent.conversation_history.append({"role": "user", "content": user_message})

        for _ in range(10):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=agent.conversation_history,
                tools=agent.tools,
                reasoning_effort="low",
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                reply = msg.content or ""
                if agent.zero_shot:
                    agent.reset_agent()
                else:
                    agent.conversation_history.append({"role": "assistant", "content": reply})
                logger.debug(f"[{agent.name.upper()}][REPLY] {reply}")
                return reply

            logger.debug(f"[{agent.name.upper()}][TOOL] Requested {len(msg.tool_calls)} tools")

            # Record the assistant turn that requested the tools BEFORE appending
            # their results. Without it the next iteration sees role=tool messages
            # with no matching tool_calls, so the agent loses track of the actions
            # it just requested and can only guess their outcome.
            agent.conversation_history.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            for tc in msg.tool_calls:
                # Here the chat supervisor decide which agent to call or to close the session
                result = self.execute_tool(agent, tc.function.name, tc.function.arguments)
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
            (i for i, m in enumerate(self.conversation_history) if m.get("role") == "user"),
            len(self.conversation_history),
        )

        normalized = []
        for i, msg in enumerate(self.conversation_history):
            role = msg.get("role")
            # Pre-conversation context: tool msgs without tool_call_id → system
            if role == "tool" and "tool_call_id" not in msg:
                normalized.append({"role": "system", "content": f"[Context] {msg['content']}"})
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
            logger.warning("[SESSION] end_session called but session is already ended – skipping")
            return {"status": "skipped", "message": "Session already ended"}

        logger.info("[SESSION] Starting end-of-session processing")

        # ── Vector DB extraction ────────────────────────────────────────────
        if self.vector_db is not None:
            patient_id = tools._get_patient_id()
            logger.info(f"[SESSION] Running vector DB extraction for patient {patient_id}")

            extract_and_save_conflict_resolutions(
                self.conversation_history, self.vector_db, patient_id
            )
            extract_and_save_patient_preferences(
                self.conversation_history, self.vector_db, patient_id
            )
        else:
            logger.warning("[SESSION] Vector DB not available – skipping knowledge extraction")

        # ── PostgreSQL save ────────────────────────────────────────────────
        if self.database_manager:
            logger.info("[SESSION] Persisting therapy to PostgreSQL")
            result = self.database_manager.save_session()
            if result.get("status") != "success":
                logger.error(f"[SESSION] PostgreSQL save failed: {result.get('message')}")
        else:
            logger.warning("[SESSION] No database manager – therapy not persisted to PostgreSQL")
            result = {"status": "skipped", "message": "No database manager available"}

        # ── Mark session as ended ────────────────────────────────────────────
        self.session_ended = True
        logger.info("[SESSION] Session marked as ended")

        return result


# Backward-compatible alias
OllamaChat = Chat
