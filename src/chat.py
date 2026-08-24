import json
from pathlib import Path
from time import time

import tools as tools
from agents.agent import Agent
from agents.check_agent import TherapyCheckAgent
from agents.therapy_manager_agent import TherapyManagerAgent
from config_loader import THERAPY_FILE
from llm_client import make_main_client
from safety import (
    ISSUE_SAFETY_BLOCKED,
    ISSUE_SAFETY_CAUTION,
    ISSUE_SAFETY_CHECK_REQUIRED,
    SEVERITY_BLOCKING,
    SEVERITY_CAUTION,
    parse_verdict,
)
from session_extractor import (
    extract_and_save_conflict_resolutions,
    extract_and_save_patient_preferences,
)
from sql_db import DatabaseManager
from utils import (
    addAgentFilterLogger,
    claims_applied_change,
    clean_tool_name,
    get_current_logger,
    visible_turns,
)
from utils import load_past_session as _load
from vector_db import MEDICINE_NOT_FOUND_MARKER

logger = get_current_logger()

# The one blocking cause no tool reports in an "issue" field: query_medicines
# answers with document text, so a miss is only recognisable by its marker. It
# belongs here because the checker's prompt forbids proceeding without the data.
SIGNAL_MEDICINE_NOT_FOUND = "medicine_not_found"

# The tools that write to the therapy. Every one of them goes through the safety
# gate below, because "is this safe for the patient" is a question about the
# activity, not about the direction of the change: removing the daily walk of a
# patient whose history records decline when he stops walking is exactly the kind
# of decision the caregiver has to be given.
_WRITE_TOOLS = (
    "add_therapy_activity",
    "update_therapy_activity",
    "remove_therapy_activity",
)

# Update fields that change what the activity is or when it happens, and
# therefore invalidate an earlier check. A pure `description` edit does not.
_SAFETY_RELEVANT_FIELDS = {
    "name",
    "category",
    "time",
    "duration_minutes",
    "day_of_week",
    "dependencies",
    "valid_from",
    "valid_until",
}


def _tool_succeeded(result) -> bool:
    """True when a tool result reports `status: success`."""
    if not isinstance(result, str) or not result.lstrip().startswith("{"):
        return False
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and payload.get("status") == "success"


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

        # ── Safety gate state (see safety.py) ────────────────────────────────
        # Which caregiver turn we are on. The caution latch needs it: a risk the
        # caregiver has to weigh is only cleared by a turn of theirs, never by the
        # model retrying inside the same agent loop.
        self._turn_index = 0
        # Every verdict the checker has returned this session, most recent first.
        # Session-scoped, not per-turn: the gate asks "has this activity been
        # checked", which is what the two scenarios it exists for got wrong (they
        # wrote a contraindicated drug with no check at all). Scoping it to the
        # turn instead forced a re-check before every single write — measured on a
        # live run of scenario 32: 50 requests and 172K tokens for one scenario,
        # the conversation spent entirely re-checking the same aspirin. Whether a
        # verdict has gone stale because the activity changed is a judgement the
        # manager's prompt already carries; the report prints every verdict with
        # its turn number so a stale one is visible.
        self._verdicts: list = []
        # Every verdict of the run, for the report: what the checker actually
        # said, at which severity, on which activity.
        self.safety_verdicts_seen: list[dict] = []
        # Checker replies whose verdict could not be parsed or carried no
        # severity. Fail-open is deliberate (safety.py) — this is what keeps it
        # from being silent.
        self.safety_verdicts_unparsed = 0
        # How often a write was attempted before the activity had been checked.
        # Not a signal (see _record_issue_signals) but worth counting: it is the
        # rate at which the prompt rule is ignored and the gate has to catch it.
        self.safety_checks_skipped = 0
        # Successful writes performed while producing the current reply, and the
        # replies that claimed a change with no write behind them at all.
        self._turn_writes: list[str] = []
        self.unsupported_claims: list[dict] = []

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

        # A rewind undoes the turns these verdicts were collected over, and a
        # verdict about an activity that has been rolled back describes something
        # else now. The cheap and correct answer is to require a fresh check.
        self._verdicts = []

        logger.info(
            f"[SNAPSHOT] Restored therapy snapshot from message_idx={snapshot['message_idx']} "
            "– safety verdicts cleared, activities must be re-checked"
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

        # 0. Safety gate. A write the checker has not cleared does not reach
        # tools.py at all — the same move that took scheduling away from the model
        # and gave it to code. The refusal is a normal tool result, so the model
        # can call the checker and retry within this same agent loop; what it
        # cannot do is skip the check or overrule a finding on its own.
        if tool_name in _WRITE_TOOLS:
            refusal = self._enforce_safety_gate(tool_name, arguments)
            if refusal is not None:
                self._record_issue_signals(tool_name, refusal)
                logger.debug(f"[{agent.name.upper()}][TOOL] {tool_name} refused: {refusal}")
                return refusal

        # 1. Delegation
        if tool_name in self._agent_registry:
            agent_delegate = self._agent_registry[tool_name]
            result = self._send_to_agent(agent_delegate, arguments)
            if agent_delegate is self.check_agent:
                self._record_safety_verdict(result)

        # 2. save_session: requires orchestrator dependencies
        elif tool_name == "save_session":
            result = json.dumps(self.end_session(), ensure_ascii=False)

        # 3. Supervisor tools
        else:
            result = agent.execute_tool(tool_name, arguments)

        if tool_name in _WRITE_TOOLS and _tool_succeeded(result):
            self._turn_writes.append(tool_name)

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

    # ── Safety gate ──────────────────────────────────────────────────────────

    def _record_safety_verdict(self, reply) -> None:
        """
        Parse the checker's reply into a typed verdict and note what it found.

        `blocking` and `caution` are recorded as issue signals here rather than in
        _record_issue_signals, because they are not carried by a tool result: they
        are the checker's judgement, and until it was severity-typed there was no
        way to tell one apart from a passing remark. That is the whole reason the
        earlier attempt at using this verdict as a signal was measured out — it
        fired on "12:45 is around lunch, so it is not fasting" — and the reason it
        is usable now. See safety.py.

        A reply with no parsable verdict counts as checked-with-no-finding and is
        logged: failing closed here would deadlock a scenario over a formatting
        slip and charge it to the behaviour under test.
        """
        verdict, typed = parse_verdict(reply if isinstance(reply, str) else "")
        if verdict is None:
            self.safety_verdicts_unparsed += 1
            logger.warning(
                "[CHAT][SAFETY] Checker reply carried no parsable verdict – treated "
                "as no finding (fail-open). Reply starts: "
                f"{str(reply)[:160]!r}"
            )
            return

        if not typed:
            self.safety_verdicts_unparsed += 1
            logger.warning(
                f"[CHAT][SAFETY] Checker verdict for '{verdict.activity_name}' is not "
                "severity-typed – its findings are treated as remarks and do not block"
            )

        # Most recent first: the gate reads the checker's *current* judgement on an
        # activity, not the first one it ever gave.
        self._verdicts.insert(0, verdict)
        self.safety_verdicts_seen.append(
            {
                "turn": self._turn_index,
                "activity_name": verdict.activity_name,
                "severity": verdict.severity,
                "typed": typed,
                "findings": verdict.findings,
            }
        )
        logger.info(
            f"[CHAT][SAFETY] Verdict on '{verdict.activity_name}': {verdict.severity} "
            f"({len(verdict.findings)} finding(s))"
        )

        if verdict.severity == SEVERITY_BLOCKING:
            self._note_signal(ISSUE_SAFETY_BLOCKED)
        elif verdict.severity == SEVERITY_CAUTION:
            self._note_signal(ISSUE_SAFETY_CAUTION)

    def _target_of_write(self, tool_name: str, arguments: dict) -> tuple[str, bool]:
        """
        The activity name a write is about, and whether the gate applies.

        For an add it is in the arguments; for an update or a remove it has to be
        read back from the therapy, since the model only passes an id. An update
        that touches nothing but the description leaves the activity unchanged as
        far as safety goes, so it is let through.
        """
        if tool_name == "add_therapy_activity":
            return str(arguments.get("name") or ""), True

        activity_id = arguments.get("activity_id")
        name = ""
        try:
            for activity in json.loads(tools.get_all_activities()).get("activities", []):
                if activity.get("activity_id") == activity_id:
                    name = activity.get("name") or ""
                    break
        except json.JSONDecodeError, TypeError:
            pass

        if tool_name == "update_therapy_activity":
            updates = arguments.get("updates") or {
                k: v for k, v in arguments.items() if k != "activity_id"
            }
            if isinstance(updates, dict):
                if not (set(updates) & _SAFETY_RELEVANT_FIELDS):
                    return name, False
                name = str(updates.get("name") or name)

        return name, True

    def _enforce_safety_gate(self, tool_name: str, arguments: dict) -> str | None:
        """
        Refuse a write the checker has not cleared. None means "let it through".

        Three refusals, in order of precedence:

        - `safety_check_required` — the checker has never been asked about this
          activity. The model can fix it without costing a caregiver turn: call
          the checker, then call the write again. It is here rather than in the
          prompt because on the 2026-08-24 batch the prompt rule was skipped on 2
          of 6 scenarios of the same class, and both wrote a contraindicated drug
          with no check at all.
        - `safety_blocked` — the checker's current judgement is an absolute
          contraindication. The caregiver cannot consent past it; the way forward
          is a different activity, which is what the scenarios of this class
          expect.
        - `safety_caution` — a risk only the caregiver can accept. Refused once,
          then allowed from the *next* turn on. The turn boundary is the point:
          releasing it on a retry inside the same loop would let the model wave
          its own warning through, which is the behaviour these scenarios exist to
          catch.

        The authority is always the checker's *latest* verdict on the activity,
        never a historical one. A block used to be latched permanently, on the
        reasoning that an absolute contraindication is not something a later turn
        dissolves — and on the first live run that banned Paracetamol for the rest
        of scenario 32 off a finding reading "contraindicated only if severe
        hepatic insufficiency", a condition this patient does not have. Paracetamol
        was the correct answer to that scenario. A misclassification is more likely
        than a contraindication changing mid-conversation, so the newest judgement
        wins; every verdict and its severity go to the report, which is what makes
        a model shopping for a softer answer visible.
        """
        name, applies = self._target_of_write(tool_name, arguments)
        if not applies:
            return None

        verb = {
            "add_therapy_activity": "added",
            "update_therapy_activity": "updated",
            "remove_therapy_activity": "removed",
        }[tool_name]

        verdict = next((v for v in self._verdicts if v.concerns(name)), None)
        if verdict is None:
            return json.dumps(
                {
                    "status": "failure",
                    "issue": ISSUE_SAFETY_CHECK_REQUIRED,
                    "message": (
                        f"'{name or 'this activity'}' has not been safety-checked. "
                        "Call delegate_to_checker_agent for this activity, read its "
                        "answer, then call this tool again. Nothing was written."
                    ),
                },
                ensure_ascii=False,
            )

        if verdict.severity == SEVERITY_BLOCKING:
            finding = verdict.summary(SEVERITY_BLOCKING) or "absolute contraindication"
            return json.dumps(
                {
                    "status": "failure",
                    "issue": ISSUE_SAFETY_BLOCKED,
                    "message": (
                        f"'{name or 'this activity'}' cannot be {verb}: the safety "
                        f"check found an absolute contraindication – {finding}. The "
                        "caregiver cannot consent past this. Report it to them and "
                        "either propose a different option or ask them how to "
                        "proceed. Nothing was written."
                    ),
                },
                ensure_ascii=False,
            )

        # A `caution` does NOT refuse the write. It used to, refused once per
        # activity and released on the caregiver's next turn, so that the decision
        # would demonstrably reach them — and measured on gpt-oss-20b at low
        # reasoning effort (scenarios 3, 13, 14) the model read the refusal,
        # reported it, and never called the tool again: the caregiver said "go
        # ahead" and nothing was written. The mechanism improved the *observability*
        # of the branch and made the actual outcome worse, which is the wrong trade
        # for a product whose job is to apply the change.
        #
        # It was also the wrong rule on its own terms. A caution is by definition a
        # relative risk the caregiver may accept; refusing a legal write is the
        # assistant arrogating the decision, which is exactly what the manager's
        # prompt forbids it to do. Reporting it and asking is the whole duty, and
        # that duty belongs to the prompt, not to a latch here.
        #
        # The delivery gate still fires: _record_safety_verdict raises
        # ISSUE_SAFETY_CAUTION when the checker returns the verdict, which happens
        # before the write, so test.py sees it in Chat.turn_issues either way.
        return None

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
            if issue == ISSUE_SAFETY_CHECK_REQUIRED:
                # Not a signal: it belongs with the validation errors excluded
                # above. Nothing is for the caregiver to decide — the model simply
                # called the write before the checker and can fix that on the next
                # iteration of its own loop. Recording it would open the delivery
                # gate on a mechanical slip: the assistant re-checks, writes, and
                # closes with "anything else?", whose question mark is enough for
                # assistant_handed_back, and the caregiver gets its reaction
                # instructions for a problem that never reached it.
                self.safety_checks_skipped += 1
                logger.info(
                    "[CHAT][SAFETY] Write refused for want of a check – not a "
                    "caregiver-facing cause, no signal recorded"
                )
            elif issue:
                signals.append(issue)
        elif tool_name == "get_medicine_data" and MEDICINE_NOT_FOUND_MARKER in result:
            signals.append(SIGNAL_MEDICINE_NOT_FOUND)

        for signal in signals:
            self._note_signal(signal, source=tool_name)

    def _note_signal(self, signal: str, source: str = "checker") -> None:
        """Record a blocking cause for this turn and for the run, once each."""
        if signal not in self._turn_issues:
            self._turn_issues.append(signal)
        if signal not in self.issue_signals_seen:
            self.issue_signals_seen.append(signal)
        logger.info(f"[CHAT][ISSUE] {source} raised '{signal}'")

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
                                "name": clean_tool_name(tc.function.name),
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            for tc in msg.tool_calls:
                # The name is cleaned before anything looks at it: some providers
                # leak the model's own format markers into it (see
                # utils.clean_tool_name), and a name that fails to resolve costs a
                # whole iteration of this loop.
                tool_name = clean_tool_name(tc.function.name)
                if tool_name != tc.function.name:
                    logger.warning(
                        f"[{agent.name.upper()}][TOOL] Backend returned the tool name as "
                        f"{tc.function.name!r} – dispatching {tool_name!r}"
                    )
                # Here the chat supervisor decide which agent to call or to close the session
                result = self.execute_tool(agent, tool_name, tc.function.arguments)
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
        self._turn_index += 1
        self._turn_issues = []
        self._turn_writes = []
        # self._verdicts is deliberately NOT cleared here: see _enforce_safety_gate.
        res = self._run_agent_loop(self.chat_agent, user_message)
        self._record_unsupported_claim(res)
        logger.debug(f"[TIMING] {time() - start:.2f}s")
        logger.info(f"[CHAT] ASSISTANT: {res}")
        return res

    def _record_unsupported_claim(self, reply: str) -> None:
        """
        Note a reply that announces a change no tool actually performed.

        Deliberately *not* a gate and not a grade: it changes no behaviour, it
        only puts the occurrence in the report. That is what makes a phrase match
        acceptable here where it was measured out of the delivery gate — a false
        positive costs a column entry a reviewer dismisses, not a caregiver being
        handed its instructions for a problem that never happened.

        The textual half is also only half the test: it fires solely when no
        write succeeded during the whole turn, which is read from tool results.
        A turn that wrote something and misdescribed *what* it wrote is not this;
        the diff the judge grades on catches that one.
        """
        if self._turn_writes or not claims_applied_change(reply):
            return
        self.unsupported_claims.append(
            {
                "turn": self._turn_index,
                "reply": (reply or "").strip()[:400],
            }
        )
        logger.warning(
            f"[CHAT][CLAIM] Turn {self._turn_index}: the reply announces a change to "
            "the therapy but no write tool succeeded during this turn"
        )

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
