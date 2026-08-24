import json
from pathlib import Path
from time import time

import tools as tools
from agents.agent import Agent
from agents.check_agent import TherapyCheckAgent
from agents.therapy_manager_agent import TherapyManagerAgent
from config_loader import THERAPY_FILE
from llm_client import make_main_client
from session_extractor import (
    extract_and_save_conflict_resolutions,
    extract_and_save_patient_preferences,
)
from sql_db import DatabaseManager
from utils import addAgentFilterLogger, get_current_logger, visible_turns
from utils import load_past_session as _load
from vector_db import MEDICINE_NOT_FOUND_MARKER

logger = get_current_logger()

# The one blocking cause no tool reports in an "issue" field: query_medicines
# answers with document text, so a miss is only recognisable by its marker. It
# belongs here because the checker's prompt forbids proceeding without the data.
SIGNAL_MEDICINE_NOT_FOUND = "medicine_not_found"


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
        model: str | None = None,
        database_manager: DatabaseManager = None,
        vector_db=None,
    ):
        """
        Initialise the LLM client of the system under test (MAIN_LLM).
        Supports OpenAI cloud, Groq and Ollama (see llm_client.make_client).

        Args:
            model: Model name; defaults to the one configured for MAIN_LLM
            system_prompt: System prompt to configure the model behaviour
            database_manager: DatabaseManager instance for session persistence
            vector_db: VectorDBManager instance for RAG features
        """
        self.client = make_main_client()
        # The client carries its own model; an explicit argument still wins.
        self.model = model or self.client.model
        self.session_ended = False
        self.database_manager = database_manager
        self.vector_db = vector_db

        # Causes the system itself detected while producing the current reply
        # (reset by send_message) and over the whole session. See
        # _record_issue_signals.
        self._turn_issues: list[str] = []
        self.issue_signals_seen: list[str] = []
        # Patient-history events the RAG surfaced over the run — see
        # _record_history_warnings.
        self.history_warnings_seen: list[dict] = []

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

        # A therapy tool that refused the write (a scheduling conflict, a blocked
        # dependency) leaves the file untouched, so snapshotting it again only
        # added an entry identical to the previous one.
        if self._therapy_snapshots and self._therapy_snapshots[-1]["therapy"] == therapy:
            logger.debug(f"[SNAPSHOT] Therapy unchanged at message_idx={current_idx} – not stored")
            return

        self._therapy_snapshots.append(
            {
                "message_idx": current_idx,
                "therapy": therapy,
            }
        )

        # Saving of snapshot on memory. session_dir is attached to the logger by
        # setup_logger, so it is absent on a plain logging.getLogger — getattr
        # keeps a Chat usable (in-memory snapshots only) when it was not called.
        session_dir = getattr(logger, "session_dir", None)
        if session_dir:
            snapshots_path = session_dir / "therapy_snapshots.json"
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

    def execute_tool(self, agent: Agent, tool_name: str, tool_arguments: str) -> str:
        """
        The orchestrator only handles:
        1. Delegation to worker agents
        2. save_session (requires db_manager and vector_db)
        Everything else is delegated to the chat_agent.

        `tool_arguments` is what the SDK hands over: the raw JSON *string* of the
        call. It is decoded once here and every branch below receives the dict,
        which is what they all expect — the delegation branch used to forward the
        undecoded string to json.dumps and hand the worker a quoted, escaped
        blob ('"{\\"message\\": …}"') as its user message.
        """
        logger.debug(f"[{agent.name.upper()}][TOOL] Executing: {tool_name}({tool_arguments})")

        if isinstance(tool_arguments, str):
            try:
                arguments = json.loads(tool_arguments or "{}")
            except json.JSONDecodeError as e:
                # A model emitting malformed arguments is a bad call, not a crash:
                # returning the error as the tool result lets it correct itself on
                # the next iteration instead of taking the whole turn down.
                logger.warning(
                    f"[{agent.name.upper()}][TOOL] {tool_name} called with invalid JSON "
                    f"arguments: {e} – {tool_arguments!r}"
                )
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Arguments were not valid JSON ({e}). Call the tool again.",
                    },
                    ensure_ascii=False,
                )
        else:
            arguments = tool_arguments
        if not isinstance(arguments, dict):
            arguments = {}

        # 1. Delegation
        if tool_name in self._agent_registry:
            agent_delegate = self._agent_registry[tool_name]
            result = self._send_to_agent(agent_delegate, arguments)

        # 2. save_session: requires orchestrator dependencies
        elif tool_name == "save_session":
            result = json.dumps(self.end_session(), ensure_ascii=False)

        # 3. Supervisor tools
        else:
            result = agent.execute_tool(tool_name, arguments)

        # If is an action that changes the therapy i save a new snapshot of it.
        # _save_therapy_snapshot is a no-op when nothing actually changed, so a
        # call blocked by a conflict no longer appends a duplicate snapshot.
        if tool_name in (
            "add_therapy_activity",
            "update_therapy_activity",
            "remove_therapy_activity",
        ):
            self._save_therapy_snapshot()

        self._record_issue_signals(tool_name, result)
        self._record_history_warnings(result)

        logger.debug(f"[{agent.name.upper()}][TOOL] Results of {tool_name}: {result}")
        return result

    def _record_history_warnings(self, result) -> None:
        """
        Collect the patient-history events the RAG actually surfaced this run.

        Recorded, never judged. Whether the assistant then passed a warning on to
        the caregiver is not decidable from the text, and it was worth measuring
        before assuming otherwise: on this dataset neither distinctive-token
        overlap nor embedding similarity separates a relayed warning from an
        unrelated reply about the same condition (verified-relayed cases score
        0.45-0.55 and 0.49-0.81 respectively, both squarely inside the range of
        everything else). The assistant paraphrases, and it discusses the
        patient's asthma whether or not it ever read the asthma event — what is
        missing is provenance, which similarity cannot supply.

        What *is* decidable is what the system put in front of it. That goes in
        the report next to the transcript so a person settles it at a glance,
        the same way changed_activities lists what changed instead of guessing
        which change was unwanted.

        `info` events are skipped: they carry nothing to warn about.
        """
        if not isinstance(result, str) or not result.lstrip().startswith("{"):
            return
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return

        # add/update attach them as patient_history_warnings; the checker's
        # get_patient_history_events returns them under events.
        found = payload.get("patient_history_warnings") or payload.get("events") or []
        if not isinstance(found, list):
            return
        seen = {event.get("description") for event in self.history_warnings_seen}
        for event in found:
            if not isinstance(event, dict) or event.get("event_type") == "info":
                continue
            description = (event.get("description") or "").strip()
            if description and description not in seen:
                self.history_warnings_seen.append(event)
                seen.add(description)

    def _record_issue_signals(self, tool_name: str, result) -> None:
        """
        Note that the system blocked something the caregiver has to decide on.

        Every tool call of every agent goes through execute_tool, so this sees
        the whole turn: the conflicts and dependency errors raised by tools.py
        and a medicine missing from the knowledge base. It records the cause,
        not the reaction: whether the assistant then passed it on to the
        caregiver is a separate question, and the one the caller has to answer.

        Only *blocking* causes count — the requested action did not happen and
        cannot happen until the caregiver chooses. Three kinds are deliberately
        left out, each after being tried and measured:

        - validation errors (bad category, malformed time, unknown activity_id):
          the assistant's own slips, retried without involving the caregiver;
        - the checker's own verdict. Its prompt has it answer with
          `check_result: [problems]` (empty array when it found none), which
          looks like a declared finding, and the format holds (64 replies out
          of 64 parsed). It is not usable as a signal: the checker comments on
          everything, so a non-empty check_result also covers quality remarks.
          In scenario 17 it observed that "12:45 is around lunch, so it is not
          fasting" — accurate, and nothing for the caregiver to decide — which
          fired the gate on turn 2 of four scenarios out of seven and recorded
          branch_exercised=True for a scheduling conflict that never happened;
        - patient-history hits, whether from get_patient_history_events or the
          `patient_history_warnings` that add/update return alongside a
          successful write. Measured on scenarios 14/17/18/20, one fires on
          essentially every request — the history threshold is permissive on
          purpose (see vector_db) and the checker queries it every time — while
          the activity is still written. Treating that as a signal made the gate
          fire on the second turn of every scenario: in 17 the caregiver was
          handed its reaction instructions after "I found no conflicts" and
          answered "yes, add it at the suggested alternative time" before any
          alternative had been suggested.

        What the last two have in common is that the activity still gets
        written: the system observed something, it did not stop anything.
        Whether such an observation deserves to be raised is the assistant's
        judgement, and the harness does not try to second-guess it — a scenario
        whose trigger is one of those records branch_exercised=False and is
        graded on the transcript instead.
        """
        if not isinstance(result, str):
            return

        signals: list[str] = []

        payload = None
        if result.lstrip().startswith("{"):
            try:
                payload = json.loads(result)
            except json.JSONDecodeError:
                payload = None

        if isinstance(payload, dict):
            issue = payload.get("issue")
            if issue:
                signals.append(issue)
        elif tool_name == "get_medicine_data" and MEDICINE_NOT_FOUND_MARKER in result:
            signals.append(SIGNAL_MEDICINE_NOT_FOUND)

        for signal in signals:
            if signal not in self._turn_issues:
                self._turn_issues.append(signal)
            if signal not in self.issue_signals_seen:
                self.issue_signals_seen.append(signal)
            logger.info(f"[CHAT][ISSUE] {tool_name} raised '{signal}'")

    @property
    def turn_issues(self) -> list[str]:
        """Causes detected while producing the last reply of send_message."""
        return list(self._turn_issues)

    def _run_agent_loop(self, agent: Agent, user_message: str) -> str:
        """
        Generic tool-calling loop for any agent.
        Used by both the supervisor (send_message) and for delegation (_send_to_agent).
        """

        logger.debug(f"[{agent.name.upper()}][REQUEST] {user_message}")
        agent.conversation_history.append({"role": "user", "content": user_message})

        for _ in range(10):
            # model and reasoning_effort come from the client's own config
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

        # The loop gave up with tool calls still pending. Close the turn the same
        # way a normal reply does: a zero_shot agent that is not reset here keeps
        # the exhausted history — including its dangling tool results — for every
        # later delegation, and a supervisor whose reply never reaches its own
        # history loses the turn from the transcript.
        reply = "Max iterations reached."
        logger.warning(f"[{agent.name.upper()}] Agent loop exhausted after 10 iterations")
        if agent.zero_shot:
            agent.reset_agent()
        else:
            agent.conversation_history.append({"role": "assistant", "content": reply})
        return reply

    def send_message(self, user_message: str) -> str:
        """Function used to send a message to the supervisor."""
        logger.info(f"[CHAT] USER: {user_message}")
        start = time()
        self._turn_issues = []
        res = self._run_agent_loop(self.chat_agent, user_message)
        logger.debug(f"[TIMING] {time() - start:.2f}s")
        logger.info(f"[CHAT] ASSISTANT: {res}")
        return res

    def _send_to_agent(self, agent: Agent, tool_arguments: dict) -> str:
        """Delegation to a worker agent"""
        return self._run_agent_loop(agent, json.dumps(tool_arguments, ensure_ascii=False))

    def get_history(self) -> list[dict]:
        """The conversation as exchanged with the caregiver."""
        return self.chat_agent.conversation_history

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

            # The real conversation lives on the supervisor. This used to pass
            # self.conversation_history, which no code path ever appends to, so
            # both extractors received an empty list, found nothing to extract
            # and wrote nothing to ChromaDB for the whole life of the feature.
            history = self.chat_agent.conversation_history
            extract_and_save_conflict_resolutions(history, self.vector_db, patient_id)
            extract_and_save_patient_preferences(history, self.vector_db, patient_id)
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
