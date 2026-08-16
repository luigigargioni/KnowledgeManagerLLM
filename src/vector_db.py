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

import uuid
from datetime import datetime
from pathlib import Path

from chromadb.utils import embedding_functions

import chromadb
from config_loader import CHROMA_DB_PATH, MEDICINES_FOLDER, PATIENTS_DATA_FOLDER
from utils import get_current_logger

logger = get_current_logger()

# ─── Collection names ────────────────────────────────────────────────────────
COLLECTION_MEDICINES = "medicines"
COLLECTION_PATIENT_HISTORY = "patient_history"
COLLECTION_CONFLICT_RESOLUTIONS = "conflict_resolutions"
COLLECTION_PATIENT_PREFERENCES = "patient_preferences"

# Cosine distance threshold for de-duplicating patient preferences.
# When a new preference is added, any existing entry closer than this value is replaced
# rather than kept alongside the new one, preventing semantically equivalent preferences
# from accumulating across sessions.
# 0.25 is intentionally strict: only near-identical phrasings are merged; distinct but
# related preferences (e.g. "prefers morning activities" vs "dislikes afternoon sessions")
# are preserved as separate entries.
# (distance range: 0 = identical, 2 = opposite; reasonable duplicate threshold ≈ 0.20–0.30)
PREFERENCE_DEDUP_THRESHOLD = 0.25

# Cosine distance threshold for de-duplicating conflict resolutions.
# Same rationale as PREFERENCE_DEDUP_THRESHOLD: replaces near-identical resolutions on
# re-insert so repeated sessions about the same conflict do not bloat the collection.
# 0.25 keeps de-dup conservative – two resolutions that address the same conflict from
# slightly different angles are NOT merged.
CONFLICT_DEDUP_THRESHOLD = 0.25

# Maximum cosine distance accepted when querying past conflict resolution hints.
# 0.65 is a moderate threshold: only resolutions with a meaningful semantic overlap with
# the current conflict are surfaced. A looser value would return loosely related past
# resolutions that could mislead the LLM; a tighter value would miss useful hints when
# the conflict is described with different wording across sessions.
CONFLICT_QUERY_THRESHOLD = 0.65

# Maximum cosine distance accepted for a medicine lookup to be considered a valid match
# when NO name-based match is found.
# A name-based match (query normalised ⊇ or == medicine name) always wins regardless of
# distance, so this threshold only acts as a guard against returning completely unrelated
# medicine data when the requested drug is genuinely absent from the knowledge base.
# 0.80 is permissive on purpose: the embedding model compares a short drug name against
# a full .md document, which structurally inflates the cosine distance even for correct
# matches. Empirically, valid matches for all-MiniLM-L6-v2 on this corpus sit in the
# 0.30–0.65 range, so 0.80 leaves a safe margin while still blocking truly unrelated docs.
MEDICINE_DISTANCE_THRESHOLD = 0.80

# Cosine distance thresholds for patient history RAG queries.
# The two values used to differ (0.85 for "danger", 0.70 for "warning") on the
# assumption that warning-level events are less costly to miss. Measured against
# the current dataset that assumption did not hold: every event the scenarios are
# designed to surface is typed "warning", and the enriched queries land between
# 0.45 and 0.85 — so the stricter value silently dropped exactly the events under
# test (e.g. Frank's dehydration event against an outdoor gardening request sat at
# 0.75, and Rose's overnight-stay event at 0.845).
# Both are therefore aligned at 0.85. The cost is roughly one extra low-relevance
# event per query, which the caller can weigh using the relevance_score returned
# with each event; the benefit is that safety history stops being invisible.
# The constants stay separate so the distinction can be re-tuned if the dataset
# starts using "danger" meaningfully.
PATIENT_HISTORY_DANGER_THRESHOLD = 0.85
PATIENT_HISTORY_WARNING_THRESHOLD = 0.85

# Maximum cosine distance accepted when querying patient preferences.
# 0.80 is intentionally permissive: preferences are heterogeneous (dietary, physical,
# cognitive) and a broader retrieval net ensures the LLM receives relevant context even
# when the query wording does not closely mirror how the preference was originally stored.
# Unlike medicines or conflict hints, returning a slightly off-topic preference carries
# little risk – the LLM can simply ignore it.
PREFERENCE_QUERY_THRESHOLD = 0.80

# Phrase carried by every "the medicine is not in the knowledge base" answer of
# query_medicines. The lookup returns raw document text, so there is no status
# field to test: this marker is what makes the miss recognisable in code rather
# than by guessing from how the assistant phrased it afterwards (see
# chat.Chat._record_issue_signals). Keep it in all the no-match branches.
MEDICINE_NOT_FOUND_MARKER = "not found in the local knowledge base"


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

    def counts(self) -> dict[str, int]:
        """Number of documents currently held in each collection."""
        return {
            COLLECTION_MEDICINES: self._medicines.count(),
            COLLECTION_PATIENT_HISTORY: self._patient_history.count(),
            COLLECTION_CONFLICT_RESOLUTIONS: self._conflict_resolutions.count(),
            COLLECTION_PATIENT_PREFERENCES: self._patient_preferences.count(),
        }

    def reset(self) -> bool:
        """
        Drop and recreate all four collections, then reopen them.

        Seeding is idempotent by document id, which means a store left over from
        an earlier dataset is never corrected: stale documents keep answering
        queries under ids that no longer describe the current patients, and
        renamed source files are indexed alongside their obsolete versions
        instead of replacing them. A batch run must therefore start from an empty
        store rather than trusting whatever is on disk.
        """
        if self.client is None:
            logger.error("[VECTOR_DB] reset() called before initialize()")
            return False
        try:
            for name in (
                COLLECTION_MEDICINES,
                COLLECTION_PATIENT_HISTORY,
                COLLECTION_CONFLICT_RESOLUTIONS,
                COLLECTION_PATIENT_PREFERENCES,
            ):
                try:
                    self.client.delete_collection(name)
                except Exception:
                    # Not an error: the collection may simply not exist yet.
                    logger.debug(f"[VECTOR_DB] Nothing to delete for '{name}'")
            logger.info("[VECTOR_DB] All collections dropped")
            return self.initialize()
        except Exception as e:
            logger.error(f"[VECTOR_DB] reset failed: {e}")
            return False

    def seed_all_patients(self, patients_folder: Path = None) -> list[str]:
        """
        Seed every patient that has a data folder. Returns the ids seeded.
        """
        folder = patients_folder or PATIENTS_DATA_FOLDER
        if not folder.exists():
            logger.error(f"[VECTOR_DB] Patients data folder not found: {folder}")
            return []

        patient_ids = sorted(
            (p.name for p in folder.iterdir() if p.is_dir()),
            key=lambda name: (
                not name.isdigit(),
                int(name) if name.isdigit() else name,
            ),
        )
        for patient_id in patient_ids:
            self.seed_patient_data(patient_id, patients_folder=folder)

        logger.info(
            f"[VECTOR_DB] Seeded {len(patient_ids)} patient folder(s): {', '.join(patient_ids)}"
        )
        return patient_ids

    def verify_seed(self, medicines_folder: Path = None) -> list[str]:
        """
        Check that the store actually holds what the source files describe.

        Returns a list of human-readable problems; empty means the store is
        consistent with the data folders. Seeding failures used to be swallowed
        by the try/except in seed_patient_data and only surfaced as an agent that
        mysteriously never warned about anything, so the caller is expected to
        treat a non-empty result as fatal.
        """
        problems: list[str] = []

        folder = medicines_folder or MEDICINES_FOLDER
        expected_medicines = {f.stem.lower() for f in folder.glob("*.md")}
        indexed_medicines = set(self._medicines.get()["ids"])
        missing = expected_medicines - indexed_medicines
        if missing:
            problems.append(
                f"{len(missing)} medicine file(s) not indexed: {', '.join(sorted(missing))}"
            )
        unexpected = indexed_medicines - expected_medicines
        if unexpected:
            problems.append(
                f"{len(unexpected)} indexed medicine(s) with no source file "
                f"(stale store?): {', '.join(sorted(unexpected))}"
            )

        history_meta = self._patient_history.get(include=["metadatas"])["metadatas"]
        history_patients = {m.get("patient_id") for m in history_meta}
        for patient_dir in sorted(PATIENTS_DATA_FOLDER.glob("*/history.json")):
            patient_id = patient_dir.parent.name
            if patient_id not in history_patients:
                problems.append(f"patient {patient_id} has history.json but no indexed events")

        for problem in problems:
            logger.error(f"[VECTOR_DB] Seed verification: {problem}")
        if not problems:
            logger.info(f"[VECTOR_DB] Seed verified – {self.counts()}")
        return problems

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
                logger.debug(f"[VECTOR_DB] Medicine '{doc_id}' already indexed – skipping")
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
                return f"Medicine '{query}' was {MEDICINE_NOT_FOUND_MARKER} (collection empty)."

            results = self._medicines.query(
                query_texts=[query],
                n_results=min(n_results, total),
                include=["documents", "distances", "metadatas"],
            )
            docs: list[str] = results.get("documents", [[]])[0]
            distances: list[float] = results.get("distances", [[]])[0]

            if not docs:
                return f"Medicine '{query}' was {MEDICINE_NOT_FOUND_MARKER} (no result)."

            metadatas: list[dict] = results.get("metadatas", [[]])[0]

            # ── Step 1: name-based lookup (always wins) ──────────────────────
            # Normalise the query and check whether any indexed medicine name is
            # a substring of (or equal to) the query, or vice-versa.
            # This handles cases like query="Aulin" matching metadata name="aulin"
            # even when the embedding distance of the short name vs the full .md
            # document would otherwise exceed the threshold.
            query_norm = query.strip().lower()
            name_matched = [
                doc
                for doc, meta in zip(docs, metadatas)
                if query_norm in meta.get("name", "").lower()
                or meta.get("name", "").lower() in query_norm
            ]
            if name_matched:
                logger.info(f"[VECTOR_DB] query_medicines: name-based match for '{query}'")
                return "\n\n---\n\n".join(name_matched)

            # ── Step 2: distance-based fallback ─────────────────────────────
            # Only used when the medicine name is not directly in the query
            # (e.g. a description like "anti-inflammatory for headache").
            matched = [
                doc for doc, dist in zip(docs, distances) if dist <= MEDICINE_DISTANCE_THRESHOLD
            ]

            if not matched:
                logger.info(
                    f"[VECTOR_DB] query_medicines: no match within threshold for '{query}' "
                    f"(best distance={distances[0]:.3f} > {MEDICINE_DISTANCE_THRESHOLD})"
                )
                return (
                    f"Medicine '{query}' was {MEDICINE_NOT_FOUND_MARKER} "
                    f"(no sufficiently similar entry; best distance={distances[0]:.3f}). "
                    "Do NOT proceed – ask the caregiver to verify contraindications manually."
                )

            return "\n\n---\n\n".join(matched)
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

        logger.info(f"[VECTOR_DB] Seeded {count} patient history events for patient {patient_id}")
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
                # Apply a tighter distance threshold for safety-critical "danger" events
                # so they are surfaced even when the semantic match is imperfect.
                event_type = meta.get("event_type", "warning")
                threshold = (
                    PATIENT_HISTORY_DANGER_THRESHOLD
                    if event_type == "danger"
                    else PATIENT_HISTORY_WARNING_THRESHOLD
                )
                if dist < threshold:
                    events.append(
                        {
                            "description": doc,
                            "activity_name": meta.get("activity_name", ""),
                            "event_type": event_type,
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

    def seed_conflict_resolutions(self, patient_id: str, resolutions: list[dict]) -> int:
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

        logger.info(f"[VECTOR_DB] Seeded {count} conflict resolution(s) for patient {patient_id}")
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
                if dist < CONFLICT_QUERY_THRESHOLD:
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
        """Persist a conflict resolution pattern.
        Replaces semantically duplicate entries (cosine distance < CONFLICT_DEDUP_THRESHOLD)
        so repeated sessions with the same conflict do not accumulate redundant records.
        """
        try:
            total = self._conflict_resolutions.count()
            action = "added"

            if total > 0:
                existing = self._conflict_resolutions.query(
                    query_texts=[description],
                    n_results=min(3, total),
                    where={"patient_id": str(patient_id)},
                )
                existing_ids: list[str] = existing.get("ids", [[]])[0]
                distances: list[float] = existing.get("distances", [[]])[0]

                for ex_id, dist in zip(existing_ids, distances):
                    if dist < CONFLICT_DEDUP_THRESHOLD:
                        self._conflict_resolutions.delete(ids=[ex_id])
                        logger.info(
                            f"[VECTOR_DB] Replaced duplicate conflict resolution {ex_id} "
                            f"for patient {patient_id}"
                        )
                        action = "replaced"
                        break

            resolution_id = (
                f"cr_{patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:8]}"
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
            logger.info(f"[VECTOR_DB] Conflict resolution {action}: {resolution_id}")
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
                        "relevance_score": round(1 - dist, 3),
                    }
                    for doc, meta, dist in zip(docs, metas, distances)
                    if dist < PREFERENCE_QUERY_THRESHOLD
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
            logger.info(f"[VECTOR_DB] Preference {action} for patient {patient_id}: {pref_id}")
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
                logger.debug(f"[VECTOR_DB] Seed preference already present, skipping: {pref_id}")
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
                logger.error(f"[VECTOR_DB] seed_patient_preferences error for {pref_id}: {e}")

        logger.info(f"[VECTOR_DB] Seeded {count} preference(s) for patient {patient_id}")
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
                n = self.seed_patient_preferences(patient_id=patient_id, preferences=preferences)
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
                n = self.seed_conflict_resolutions(patient_id=patient_id, resolutions=resolutions)
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

        logger.info(f"[VECTOR_DB] Patient data seeding complete for patient {patient_id}")
