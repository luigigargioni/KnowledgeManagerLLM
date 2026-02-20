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
import logging

import requests

from config_loader import LLM_TIMEOUT, MODEL, OLLAMA_URL
from prompts import _CONFLICT_EXTRACTION_PROMPT, _PREFERENCE_EXTRACTION_PROMPT

logger = logging.getLogger("knowledge_manager")


def _format_conversation(conversation_history: list[dict]) -> str:
    """
    Turn the raw conversation history into readable text for the extractor.
    Skips system and tool messages – only user and assistant turns are kept.
    """
    lines = []
    for msg in conversation_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"Caregiver: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    return "\n".join(lines)


def _call_llm(system_prompt: str, user_text: str) -> list[dict]:
    """
    Call the local Ollama LLM with a specialised system prompt.
    Returns the parsed JSON list, or an empty list on failure.
    """
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Conversation to analyse:\n\n{user_text}",
            },
        ],
        "stream": False,
    }
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat", json=payload, timeout=LLM_TIMEOUT
        )
        response.raise_for_status()
        content: str = response.json().get("message", {}).get("content", "[]").strip()

        # Strip markdown code fences if the model wraps the JSON
        if content.startswith("```"):
            lines = content.splitlines()
            # remove first line (``` or ```json) and last line (```)
            content = "\n".join(
                lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
            )

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
                logger.debug(
                    f"[EXTRACTOR] Conflict resolution saved: {description[:80]!r}"
                )

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
