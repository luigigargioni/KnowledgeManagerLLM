# TODO


1. Evaluate the use of the TOON format [library](https://github.com/toon-format/toon-python) to speed up JSON analysis by the LLM --> For now, no: upon further reading, since it is a new standard the models are not yet trained on it and therefore need to be instructed. In our case the JSON is very short, so the savings would be minimal. At the moment, it’s not worth the effort.



# Notes / To Discuss
- If a medication needs to be taken on a full stomach and we assume there is a conflict with the time chosen by the user, does it make sense for the deterministic algorithm to suggest taking it before lunch? Should the LLM therefore combine both the semantic check and the deterministic one to resolve conflicts? (possible limitation or future work)

- When a medication is added, a check is performed in the vector database for contraindications, but the reverse is not done. That is, when an activity (e.g., a walk) is added, we should check all medications the patient is taking to verify that the activity is not contraindicated (possible limitation or future work)