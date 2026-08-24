_CONFLICT_EXTRACTION_PROMPT = """You are a specialist in analysing therapy management conversations.
Your task: extract every ACTIVITY CONFLICT that occurred in the conversation below AND how it was resolved.

A conflict includes:
- An activity incompatible with a medicine or medical condition that was rejected or changed.
- Any safety issue raised about an activity and the resolution adopted.

Don't include minor scheduling conflicts that were resolved by simply choosing another time slot without any safety concern or medical incompatibility.

For each conflict output a JSON array with objects containing:
  "description":      Clear, self-contained text describing BOTH the conflict AND its resolution
                      (must be useful as a standalone retrieval document in the future).
  "activity_name":    Name of the primary activity involved.
  "resolution_type":  One of: "rescheduled" | "rejected" | "modified" | "alternative_suggested".

If NO meaningful conflicts or resolutions are found output an empty array: []

IMPORTANT: respond ONLY with a valid JSON array. No markdown fences, no explanation, no preamble.
"""

_PREFERENCE_EXTRACTION_PROMPT = """You are a specialist in extracting patient preferences from conversations between a caregiver and an assistant to manage a patient's therapy.
Your task: analyse the conversation below and extract every PATIENT PREFERENCE mentioned.

A preference is any information about what the patient:
- Likes or dislikes (foods, activities, times of day, environments, etc.).
- Tolerates well or poorly.
- Prefers for comfort, habit or personal reasons.
- Follows as a regular routine.

Dont' consider a preference any information that is purely medical (e.g. "the patient has diabetes, so they can't eat sugar" is NOT a preference; but "the patient usually eats fruit in the morning and prefers that to sugary snacks" is a preference).

For each preference output a JSON array with objects containing:
  "description":  Clear, self-contained text describing the preference
                  (must be useful as a standalone retrieval document in the future).
  "category":     One of: "food" | "activity" | "schedule" | "medication" | "comfort" |
                  "social" | "other".

If NO meaningful preferences are found output an empty array: []

IMPORTANT: respond ONLY with a valid JSON array. No markdown fences, no explanation, no preamble.
"""
