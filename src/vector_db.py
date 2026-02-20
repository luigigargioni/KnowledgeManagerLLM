"""
Vector Database Manager using ChromaDB.
Embeddings model: all-MiniLM-L6-v2 (ChromaDB's default embedding function).
Manages 4 collections:

  READ-ONLY at runtime (seeded from files, never written during a session):
  - medicines:             Pharmacological data indexed from .md files (RAG for medicine info)
  - patient_history:       Historical safety events per patient (checked before managing activities)

  READ-WRITE at runtime (updated at end of each session via session_extractor):
  - conflict_resolutions:  Past conflict resolutions (proactive hints when conflicts arise)
  - patient_preferences:   Patient preferences per patient (personalise therapy suggestions)
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path

from chromadb.utils import embedding_functions

import chromadb
from config_loader import CHROMA_DB_PATH, MEDICINES_FOLDER, PATIENTS_DATA_FOLDER

logger = logging.getLogger("knowledge_manager")

# ─── Collection names ────────────────────────────────────────────────────────
COLLECTION_MEDICINES = "medicines"
COLLECTION_PATIENT_HISTORY = "patient_history"
COLLECTION_CONFLICT_RESOLUTIONS = "conflict_resolutions"
COLLECTION_PATIENT_PREFERENCES = "patient_preferences"

# Cosine distance threshold below which two preferences are considered the same concept
# (distance range: 0 = identical, 2 = opposite; reasonable duplicate threshold ≈ 0.20–0.30)
PREFERENCE_DEDUP_THRESHOLD = 0.25


class VectorDBManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(CHROMA_DB_PATH)
        self.client: chromadb.PersistentClient | None = None
        self._ef = embedding_functions.DefaultEmbeddingFunction()

        # collection references (set after initialize())
        self._medicines = None
        self._patient_history = None
        self._conflict_resolutions = None
        self._patient_preferences = None

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """
        Open (or create) the persistent ChromaDB store and ensure all 4 collections exist.
        Returns True on success, False on error.
        """
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)

            cos = {"hnsw:space": "cosine"}
            self._medicines = self.client.get_or_create_collection(
                name=COLLECTION_MEDICINES, embedding_function=self._ef, metadata=cos
            )
            self._patient_history = self.client.get_or_create_collection(
                name=COLLECTION_PATIENT_HISTORY,
                embedding_function=self._ef,
                metadata=cos,
            )
            self._conflict_resolutions = self.client.get_or_create_collection(
                name=COLLECTION_CONFLICT_RESOLUTIONS,
                embedding_function=self._ef,
                metadata=cos,
            )
            self._patient_preferences = self.client.get_or_create_collection(
                name=COLLECTION_PATIENT_PREFERENCES,
                embedding_function=self._ef,
                metadata=cos,
            )

            logger.info(
                "[VECTOR_DB] Initialized – "
                f"medicines:{self._medicines.count()} "
                f"history:{self._patient_history.count()} "
                f"conflicts:{self._conflict_resolutions.count()} "
                f"preferences:{self._patient_preferences.count()}"
            )
            return True
        except Exception as e:
            logger.error(f"[VECTOR_DB] Initialization failed: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════════════════
    # MEDICINES
    # ═══════════════════════════════════════════════════════════════════════════

    def seed_medicines(self, medicines_folder: Path = None) -> int:
        """
        Index every .md file in the medicines folder as a single document
        (chunk size = entire file; medicine files are small enough to keep whole
        so that all contraindicaitons and dosage info are always retrieved together).
        Idempotent: files already indexed are skipped.
        Returns the number of newly indexed files.
        """
        folder = medicines_folder or MEDICINES_FOLDER
        md_files = list(folder.glob("*.md"))
        if not md_files:
            logger.warning(f"[VECTOR_DB] No .md files found in {folder}")
            return 0

        existing_ids: set[str] = set(self._medicines.get()["ids"])
        count = 0

        for md_file in md_files:
            doc_id = md_file.stem.lower()  # e.g. "aspirina"
            if doc_id in existing_ids:
                logger.debug(
                    f"[VECTOR_DB] Medicine '{doc_id}' already indexed – skipping"
                )
                continue
            try:
                content = md_file.read_text(encoding="utf-8")
                self._medicines.add(
                    documents=[content],
                    ids=[doc_id],
                    metadatas=[
                        {
                            "name": md_file.stem,
                            "file": md_file.name,
                            "indexed_at": datetime.now().isoformat(),
                        }
                    ],
                )
                logger.info(f"[VECTOR_DB] Indexed medicine: {doc_id}")
                count += 1
            except Exception as e:
                logger.error(f"[VECTOR_DB] Error indexing {md_file}: {e}")

        return count

    def query_medicines(self, query: str, n_results: int = 1) -> str:
        """
        RAG retrieval on the medicines collection.
        Returns the relevant document concatenated as a string ready for the LLM.
        Only 1 document is returned by default since each file contains comprehensive info about a single medicine.
        """
        try:
            total = self._medicines.count()
            if total == 0:
                return f"No medicine data available for: {query}"

            results = self._medicines.query(
                query_texts=[query],
                n_results=min(n_results, total),
            )
            docs: list[str] = results.get("documents", [[]])[0]
            if not docs:
                return f"No medicine information found for: {query}"

            return "\n\n---\n\n".join(docs)
        except Exception as e:
            logger.error(f"[VECTOR_DB] query_medicines error: {e}")
            return f"Error querying medicine data: {e}"

    # ═══════════════════════════════════════════════════════════════════════════
    # PATIENT HISTORY
    # ═══════════════════════════════════════════════════════════════════════════

    def seed_patient_history(self, patient_id: str, events: list[dict]) -> int:
        """
        Seed historical safety events for a patient.
        Each event dict must contain:
          - description (str)   – human-readable account of what happened
          - activity_name (str) – activity involved
          - event_type (str)    – "danger" | "warning"
          - date (str)          – YYYY-MM-DD
        Idempotent: uses deterministic IDs.
        Returns the number of new events inserted.
        """
        existing_ids: set[str] = set(self._patient_history.get()["ids"])
        count = 0

        for event in events:
            safe_name = event.get("activity_name", "unknown").replace(" ", "_").lower()
            event_id = f"ph_{patient_id}_{event.get('date', 'unknown')}_{safe_name}"

            if event_id in existing_ids:
                continue

            self._patient_history.add(
                documents=[event["description"]],
                ids=[event_id],
                metadatas=[
                    {
                        "patient_id": str(patient_id),
                        "activity_name": event.get("activity_name", ""),
                        "event_type": event.get("event_type", "warning"),
                        "date": event.get("date", ""),
                    }
                ],
            )
            count += 1

        logger.info(
            f"[VECTOR_DB] Seeded {count} patient history events for patient {patient_id}"
        )
        return count

    def query_patient_history(
        self, patient_id: str, activity_description: str, n_results: int = 3
    ) -> list[dict]:
        """
        Retrieve past dangerous/notable events for a patient that are semantically
        similar to the given activity description.
        Returns a list of event dicts (empty list if none are relevant).
        """
        try:
            total = self._patient_history.count()
            if total == 0:
                return []

            results = self._patient_history.query(
                query_texts=[activity_description],
                n_results=min(n_results, total),
                where={"patient_id": str(patient_id)},
            )
            events = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(docs, metas, distances):
                if dist < 0.70:  # cosine distance threshold for relevance
                    events.append(
                        {
                            "description": doc,
                            "activity_name": meta.get("activity_name", ""),
                            "event_type": meta.get("event_type", ""),
                            "date": meta.get("date", ""),
                            "relevance_score": round(1 - dist, 3),
                        }
                    )

            return events
        except Exception as e:
            logger.error(f"[VECTOR_DB] query_patient_history error: {e}")
            return []

    # ═══════════════════════════════════════════════════════════════════════════
    # CONFLICT RESOLUTIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def seed_conflict_resolutions(
        self, patient_id: str, resolutions: list[dict]
    ) -> int:
        """
        Seed historical conflict resolution patterns for a patient.
        Each resolution dict must contain:
          - description (str)   – human-readable account of how the conflict was resolved
          - activity_name (str) – activity involved
          - date (str)          – YYYY-MM-DD
        Idempotent: uses deterministic IDs.
        Returns the number of new resolutions inserted.
        """
        existing_ids: set[str] = set(self._conflict_resolutions.get()["ids"])
        count = 0

        for res in resolutions:
            safe_name = res.get("activity_name", "unknown").replace(" ", "_").lower()
            res_id = f"cr_{patient_id}_{res.get('date', 'unknown')}_{safe_name}"

            if res_id in existing_ids:
                continue

            self._conflict_resolutions.add(
                documents=[res["description"]],
                ids=[res_id],
                metadatas=[
                    {
                        "patient_id": str(patient_id),
                        "activity_name": res.get("activity_name", ""),
                        "date": res.get("date", ""),
                    }
                ],
            )
            count += 1

        logger.info(
            f"[VECTOR_DB] Seeded {count} conflict resolution(s) for patient {patient_id}"
        )
        return count

    def query_conflict_resolutions(
        self, conflict_description: str, patient_id: str = None, n_results: int = 3
    ) -> list[dict]:
        """
        Retrieve past conflict resolution patterns similar to the current conflict.
        When patient_id is provided, results are filtered to that patient only.
        Returns a list of resolution dicts.
        """
        try:
            total = self._conflict_resolutions.count()
            if total == 0:
                return []

            query_kwargs: dict = {
                "query_texts": [conflict_description],
                "n_results": min(n_results, total),
            }
            if patient_id is not None:
                query_kwargs["where"] = {"patient_id": str(patient_id)}

            results = self._conflict_resolutions.query(**query_kwargs)
            items = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(docs, metas, distances):
                if dist < 0.65:
                    items.append(
                        {
                            "description": doc,
                            "activity_name": meta.get("activity_name", ""),
                            "patient_id": meta.get("patient_id", ""),
                            "date": meta.get("date", ""),
                            "relevance_score": round(1 - dist, 3),
                        }
                    )

            return items
        except Exception as e:
            logger.error(f"[VECTOR_DB] query_conflict_resolutions error: {e}")
            return []

    def add_conflict_resolution(
        self, description: str, patient_id: str, activity_name: str = ""
    ) -> bool:
        """Persist a conflict resolution pattern."""
        try:
            resolution_id = (
                f"cr_{patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_"
                f"{str(uuid.uuid4())[:8]}"
            )
            self._conflict_resolutions.add(
                documents=[description],
                ids=[resolution_id],
                metadatas=[
                    {
                        "patient_id": str(patient_id),
                        "activity_name": activity_name,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                    }
                ],
            )
            logger.info(f"[VECTOR_DB] Added conflict resolution: {resolution_id}")
            return True
        except Exception as e:
            logger.error(f"[VECTOR_DB] add_conflict_resolution error: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════════════════
    # PATIENT PREFERENCES
    # ═══════════════════════════════════════════════════════════════════════════

    def query_patient_preferences(
        self, patient_id: str, query: str = "", n_results: int = 10
    ) -> list[dict]:
        """
        Retrieve patient preferences.
        - If `query` is provided, performs a semantic search and returns the closest matches.
        - If `query` is empty, returns all preferences for the patient.
        """
        try:
            total = self._patient_preferences.count()
            if total == 0:
                return []

            if query:
                results = self._patient_preferences.query(
                    query_texts=[query],
                    n_results=min(n_results, total),
                    where={"patient_id": str(patient_id)},
                )
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]

                return [
                    {
                        "description": doc,
                        "category": meta.get("category", ""),
                        "date": meta.get("date", ""),
                    }
                    for doc, meta, dist in zip(docs, metas, distances)
                    if dist < 0.80
                ]
            else:
                # Return all preferences for this patient
                results = self._patient_preferences.get(
                    where={"patient_id": str(patient_id)},
                )
                docs = results.get("documents", [])
                metas = results.get("metadatas", [])
                return [
                    {
                        "description": doc,
                        "category": meta.get("category", ""),
                        "date": meta.get("date", ""),
                    }
                    for doc, meta in zip(docs, metas)
                ]
        except Exception as e:
            logger.error(f"[VECTOR_DB] query_patient_preferences error: {e}")
            return []

    def upsert_patient_preference(
        self, patient_id: str, preference_text: str, category: str = "general"
    ) -> tuple[bool, str]:
        """
        Add or overwrite a patient preference.
        If a very similar preference already exists (cosine distance < PREFERENCE_DEDUP_THRESHOLD),
        the old record is deleted and the new one is added in its place.
        Returns (success: bool, action: str) where action ∈ {"added", "replaced", "error"}.
        """
        try:
            total = self._patient_preferences.count()
            action = "added"

            if total > 0:
                existing = self._patient_preferences.query(
                    query_texts=[preference_text],
                    n_results=min(3, total),
                    where={"patient_id": str(patient_id)},
                )
                existing_ids: list[str] = existing.get("ids", [[]])[0]
                distances: list[float] = existing.get("distances", [[]])[0]

                for ex_id, dist in zip(existing_ids, distances):
                    if dist < PREFERENCE_DEDUP_THRESHOLD:
                        self._patient_preferences.delete(ids=[ex_id])
                        logger.info(
                            f"[VECTOR_DB] Replaced conflicting preference {ex_id} "
                            f"for patient {patient_id}"
                        )
                        action = "replaced"
                        break

            pref_id = (
                f"pref_{patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_"
                f"{str(uuid.uuid4())[:8]}"
            )
            self._patient_preferences.add(
                documents=[preference_text],
                ids=[pref_id],
                metadatas=[
                    {
                        "patient_id": str(patient_id),
                        "category": category,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                    }
                ],
            )
            logger.info(
                f"[VECTOR_DB] Preference {action} for patient {patient_id}: {pref_id}"
            )
            return True, action
        except Exception as e:
            logger.error(f"[VECTOR_DB] upsert_patient_preference error: {e}")
            return False, "error"

    # ─── Seed helpers (file-based) ────────────────────────────────────────────

    def seed_patient_preferences(self, patient_id: str, preferences: list[dict]) -> int:
        """
        Seed static preferences from a file using deterministic IDs derived from
        a content hash. Unlike upsert_patient_preference, this method NEVER
        overwrites dynamically learned preferences already stored in ChromaDB.
        Already-seeded entries (same hash) are silently skipped.
        Returns the number of newly inserted preferences.
        """
        import hashlib

        existing_ids: set[str] = set(self._patient_preferences.get()["ids"])
        count = 0

        for pref in preferences:
            desc = pref.get("description", "").strip()
            if not desc:
                continue
            content_hash = hashlib.md5(desc.encode("utf-8")).hexdigest()[:12]
            pref_id = f"pref_seed_{patient_id}_{content_hash}"

            if pref_id in existing_ids:
                logger.debug(
                    f"[VECTOR_DB] Seed preference already present, skipping: {pref_id}"
                )
                continue

            try:
                self._patient_preferences.add(
                    documents=[desc],
                    ids=[pref_id],
                    metadatas=[
                        {
                            "patient_id": str(patient_id),
                            "category": pref.get("category", "other"),
                            "date": "seed",
                        }
                    ],
                )
                count += 1
            except Exception as e:
                logger.error(
                    f"[VECTOR_DB] seed_patient_preferences error for {pref_id}: {e}"
                )

        logger.info(
            f"[VECTOR_DB] Seeded {count} preference(s) for patient {patient_id}"
        )
        return count

    def seed_patient_data(self, patient_id: str, patients_folder: Path = None) -> None:
        """
        Load and index history events, preferences and conflict resolutions for a patient
        from JSON files:
          <patients_folder>/<patient_id>/history.json
          <patients_folder>/<patient_id>/preferences.json
          <patients_folder>/<patient_id>/conflict_resolutions.json

        All files are optional – if absent, the respective collection is simply not seeded.
        Seeding is idempotent: records already present in ChromaDB are skipped.

        history.json format – JSON array of objects:
          { "description": str, "activity_name": str,
            "event_type": "danger"|"warning", "date": "YYYY-MM-DD" }

        preferences.json format – JSON array of objects:
          { "description": str, "category": str }

        conflict_resolutions.json format – JSON array of objects:
          { "description": str, "activity_name": str, "date": "YYYY-MM-DD" }
        """
        import json as _json

        folder = (patients_folder or PATIENTS_DATA_FOLDER) / str(patient_id)

        # ── Patient history ────────────────────────────────────────────────
        history_file = folder / "history.json"
        if history_file.exists():
            try:
                events = _json.loads(history_file.read_text(encoding="utf-8"))
                n = self.seed_patient_history(patient_id=patient_id, events=events)
                logger.info(
                    f"[VECTOR_DB] Seeded {n} history event(s) for patient {patient_id} "
                    f"from {history_file}"
                )
            except Exception as e:
                logger.error(f"[VECTOR_DB] Failed to load {history_file}: {e}")
        else:
            logger.debug(
                f"[VECTOR_DB] No history file found for patient {patient_id} at {history_file}"
            )

        # ── Patient preferences ────────────────────────────────────────────
        preferences_file = folder / "preferences.json"
        if preferences_file.exists():
            try:
                preferences = _json.loads(preferences_file.read_text(encoding="utf-8"))
                n = self.seed_patient_preferences(
                    patient_id=patient_id, preferences=preferences
                )
                logger.info(
                    f"[VECTOR_DB] Seeded {n} new preference(s) for patient "
                    f"{patient_id} from {preferences_file}"
                )
            except Exception as e:
                logger.error(f"[VECTOR_DB] Failed to load {preferences_file}: {e}")
        else:
            logger.debug(
                f"[VECTOR_DB] No preferences file found for patient {patient_id} at "
                f"{preferences_file}"
            )

        # ── Conflict resolutions ───────────────────────────────────────────
        conflict_file = folder / "conflict_resolutions.json"
        if conflict_file.exists():
            try:
                resolutions = _json.loads(conflict_file.read_text(encoding="utf-8"))
                n = self.seed_conflict_resolutions(
                    patient_id=patient_id, resolutions=resolutions
                )
                logger.info(
                    f"[VECTOR_DB] Seeded {n} conflict resolution(s) for patient {patient_id} "
                    f"from {conflict_file}"
                )
            except Exception as e:
                logger.error(f"[VECTOR_DB] Failed to load {conflict_file}: {e}")
        else:
            logger.debug(
                f"[VECTOR_DB] No conflict resolutions file found for patient {patient_id} "
                f"at {conflict_file}"
            )

        logger.info(
            f"[VECTOR_DB] Patient data seeding complete for patient {patient_id}"
        )
