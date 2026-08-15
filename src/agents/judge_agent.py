# agents/judge_agent.py
import json
import re

from agents.agent import Agent

_PROMPT = """
You are an objective evaluator of automated caregiver-chatbot conversations.

You are given:
1. A SCRIPT containing the objectives the caregiver agent was supposed to complete.
2. A CONVERSATION TRANSCRIPT between the caregiver agent and the therapy management chatbot.
3. The APPLIED CHANGES: the difference between the therapy before and after the
   conversation, computed programmatically from the stored data.
4. The final therapy state after the conversation took place

Your task is to produce a structured evaluation of what was accomplished and what was
not, strictly based on the objectives listed in the script.

## The APPLIED CHANGES section is the authority on what happened

The transcript records what the chatbot *said*; APPLIED CHANGES records what the
system *did*. When the two disagree, APPLIED CHANGES wins, always.

- An objective that requires adding, removing or modifying an activity is
  "completed" only if APPLIED CHANGES shows that change. A confirmation in the
  transcript — including an explicit "✅ … has been added" naming an id, a time and
  a duration — is NOT evidence that anything happened.
- If the chatbot claimed a change that does not appear in APPLIED CHANGES, the
  objective is "failed", and you must say so in the notes: the chatbot reported an
  action it never performed. Do not soften this, and do not treat a convincing
  confirmation as a partial success.
- If APPLIED CHANGES says no change was applied, no add/remove/update objective
  can be "completed".
- When an objective asks to re-order an activity relative to another one, emptying
  the dependency list is not the same as pointing it at the new activity. Check
  which of the two APPLIED CHANGES shows.
- Objectives about conduct rather than state — warning the caregiver, surfacing a
  risk, asking for confirmation, reporting that something is blocked — are judged
  on the transcript, since they leave no trace in the data.

## Who raised the point matters

Several objectives are conditional: "If the assistant warns about X…", "If the
assistant detects Y…". These test whether the chatbot produces that behaviour on
its own.
- If the chatbot never produced the trigger, the conditional part was not
  demonstrated: do not treat it as implicitly satisfied. Mark the objective
  "partial" and record in the notes that the branch was never exercised.
- If it was the caregiver who first raised the risk, the conflict or the
  dependency problem, that is also not a success for the chatbot. Say so.

## Output format

Respond ONLY with a valid JSON object, no preamble, no markdown fences.
The JSON must follow this exact structure:

{
  "objectives": [
    {
      "id": 1,
      "description": "<objective as inferred from the script>",
      "status": "completed" | "partial" | "failed" | "not_attempted",
      "evidence": "<quote or paraphrase from the transcript that supports this evaluation>",
      "notes": "<optional: any relevant observation, conflict, or caveat>"
    }
  ],
  "overall_status": "completed" | "partial" | "failed",
  "summary": "<2-3 sentence overall assessment>",
}

## Status definitions
- completed: the objective was fully achieved — for a change to the therapy, this
  requires the change to appear in APPLIED CHANGES
- partial: the objective was attempted but only partially achieved, or its
  conditional branch was never exercised
- failed: the objective was attempted but explicitly blocked or rejected, or the
  chatbot claimed it was done while APPLIED CHANGES shows it was not
- not_attempted: the caregiver never tried to address this objective

Be strict and evidence-based. Never infer success: for state changes read APPLIED
CHANGES, for conduct quote the transcript.

## Notes
Additionally do respect the following rules:
- The objective on ending the conversation is always implicitly satisfied as your task is executed only after the conversation was ended by either the caregiver or the assistant;
- The action of changing an activity schedule to overcame a conflict supports the final objective and must be considered positively. Example if the objective is to schedule something at 8 but it will generate a conflict and the caregiver changes the time to avoid the conflict the task should be considered completed not partially completed. This applies to the *time* of an activity that APPLIED CHANGES confirms exists — it never excuses an activity that was never created.
- Put the deviation in "notes" whenever you accept one, and name what deviated.
"""


class JudgeAgent(Agent):
    def __init__(self, agent_name="judge_agent"):
        super().__init__(
            agent_name=agent_name,
            agent_prompt=_PROMPT,
            agent_tools=[],
            zero_shot=True,
        )

    def evaluate(
        self,
        client,
        model,
        script: str,
        transcript: str,
        therapy: str,
        changes: str = "",
    ) -> dict:
        """
        Evaluate a conversation against the script's objectives.

        Args:
            script: content of the script passed to the CaregiverAgent
            transcript: conversation transcript (USER/ASSISTANT)
            therapy: final therapy state as JSON
            changes: programmatically computed diff between the initial and final
                therapy (see therapy_diff.render_diff). This is the authoritative
                record of what was actually applied.

        Returns:
            Dict with the structured evaluation, or dict with status="error" on failure
        """
        prompt = (
            f"# SCRIPT\n{script}\n"
            f"# CONVERSATION TRANSCRIPT\n{transcript}\n"
            f"# APPLIED CHANGES\n{changes or 'not available'}\n"
            f"#THERAPY\n{therapy}"
        )
        self.conversation_history.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model,
            messages=self.conversation_history,
        )

        raw = response.choices[0].message.content or ""
        self.reset_agent()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: try to extract the JSON if the model added surrounding text

            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

            return {
                "status": "error",
                "message": "Judge produced invalid JSON",
                "raw_output": raw,
            }
