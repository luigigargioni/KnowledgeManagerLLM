# agents/judge_agent.py
import json
import re

from agents.agent import Agent

_PROMPT = """
You are an objective evaluator of automated caregiver-chatbot conversations.

You are given:
1. A SCRIPT containing the objectives the caregiver agent was supposed to complete.
2. A CONVERSATION TRANSCRIPT between the caregiver agent and the therapy management chatbot.
3. The final therapy state after the conversation took place

Your task is to analyze the transcript and produce a structured evaluation of what was
accomplished and what was not, strictly based on the objectives listed in the script.

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
- completed: the objective was fully achieved and confirmed in the conversation
- partial: the objective was attempted but only partially achieved
- failed: the objective was attempted but explicitly blocked or rejected
- not_attempted: the caregiver never tried to address this objective

Be strict and evidence-based. Do not infer success unless the transcript explicitly confirms it.
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
        self, client, model, script: str, transcript: str, therapy: str
    ) -> dict:
        """
        Evaluate a conversation against the script's objectives.

        Args:
            script: content of the script passed to the CaregiverAgent
            transcript: conversation transcript (USER/ASSISTANT)

        Returns:
            Dict with the structured evaluation, or dict with status="error" on failure
        """
        prompt = f"# SCRIPT\n{script}\n# CONVERSATION TRANSCRIPT\n{transcript}\n#THERAPY\n{therapy}"
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
