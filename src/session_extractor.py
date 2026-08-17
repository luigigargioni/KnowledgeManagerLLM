"""
End-of-session LLM extraction module.

When the caregiver saves a session this module:
1. Calls the LLM with a specialised system prompt to extract any CONFLICT RESOLUTIONS
   that occurred during the conversation and persists them in the vector DB.
2. Calls the LLM with a specialised system prompt to extract any PATIENT PREFERENCES
   that emerged during the conversation and upserts them in the vector DB
   (new preference overwrites the old one if they describe the same concept).
"""

import json
import re

from config_loader import MODEL
from llm_client import make_main_client
from prompts import _CONFLICT_EXTRACTION_PROMPT, _PREFERENCE_EXTRACTION_PROMPT
from utils import get_current_logger, visible_turns

logger = get_current_logger()

# One client for the module. make_main_client() builds a fresh OpenAI instance —
# and with it a fresh HTTP connection pool — on every call, and both extractors
# run at every end of session.
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = make_main_client()
    return _client


def _format_conversation(conversation_history: list[dict]) -> str:
    """
    Turn the raw conversation history into readable text for the extractor.
    Only user and assistant turns are included; tool results are excluded because
    they contain raw JSON with patient data that inflates the context and is not
    useful for preference/conflict extraction.
    System messages and init-context tool injections are also skipped.
    """
    lines = []
    for msg in visible_turns(conversation_history):
        speaker = "Caregiver" if msg["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {msg['content']}")
    return "\n".join(lines)


def _call_llm(system_prompt: str, user_text: str) -> list[dict]:
    """
    Call the LLM with a specialised system prompt.
    Works with every provider supported by llm_client.make_client().
    Returns the parsed JSON list, or an empty list on failure.
    """
    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Conversation to analyse:\n\n{user_text}",
                },
            ],
        )
        content: str = (response.choices[0].message.content or "[]").strip()

        # Strip markdown code fences if the model wraps the JSON.
        # Handles: ```json, ```JSON, ```, trailing whitespace/blank lines.
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\s*\n?", "", content)
            content = re.sub(r"\n?```\s*$", "", content.rstrip())
            content = content.strip()

        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        logger.warning("[EXTRACTOR] LLM returned non-list JSON – ignoring")
        return []

    except json.JSONDecodeError as e:
        logger.error(f"[EXTRACTOR] Failed to parse LLM JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"[EXTRACTOR] LLM call failed: {e}")
        return []


# ─── Public API ───────────────────────────────────────────────────────────────


def extract_and_save_conflict_resolutions(
    conversation_history: list[dict],
    vector_db,
    patient_id: str,
) -> int:
    """
    Extract conflict resolutions from the conversation history and save them to ChromaDB.
    Returns the number of records saved.
    """
    conversation_text = _format_conversation(conversation_history)
    if not conversation_text.strip():
        return 0

    logger.info("[EXTRACTOR] Extracting conflict resolutions…")
    items = _call_llm(_CONFLICT_EXTRACTION_PROMPT, conversation_text)

    saved = 0
    for item in items:
        description = item.get("description", "").strip()
        activity_name = item.get("activity_name", "")
        if description:
            ok = vector_db.add_conflict_resolution(
                description=description,
                patient_id=patient_id,
                activity_name=activity_name,
            )
            if ok:
                saved += 1
                logger.debug(f"[EXTRACTOR] Conflict resolution saved: {description[:80]!r}")

    logger.info(f"[EXTRACTOR] {saved} conflict resolution(s) saved")
    return saved


def extract_and_save_patient_preferences(
    conversation_history: list[dict],
    vector_db,
    patient_id: str,
) -> int:
    """
    Extract patient preferences from the conversation history and upsert them into ChromaDB.
    If a new preference conflicts with an existing one the old record is replaced.
    Returns the number of records saved/updated.
    """
    conversation_text = _format_conversation(conversation_history)
    if not conversation_text.strip():
        return 0

    logger.info("[EXTRACTOR] Extracting patient preferences…")
    items = _call_llm(_PREFERENCE_EXTRACTION_PROMPT, conversation_text)

    saved = 0
    for item in items:
        description = item.get("description", "").strip()
        category = item.get("category", "other")
        if description:
            ok, action = vector_db.upsert_patient_preference(
                patient_id=patient_id,
                preference_text=description,
                category=category,
            )
            if ok:
                saved += 1
                logger.debug(f"[EXTRACTOR] Preference {action}: {description[:80]!r}")

    logger.info(f"[EXTRACTOR] {saved} patient preference(s) saved/updated")
    return saved
