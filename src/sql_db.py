"""
Module for managing the PostgreSQL database
Uses SQLAlchemy as ORM with psycopg2 as driver
"""

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

from config_loader import DB_CONNECTION_STRING, PATIENTS_DATA_FOLDER, THERAPY_FILE
from utils import get_current_logger

logger = get_current_logger()


# region BASE & MODELS


class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    name = Column(String(255), nullable=False)
    gender = Column(String(255), nullable=False)
    birth_date = Column(DateTime, default=datetime.now, nullable=False)
    medical_conditions = Column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    # Relationship with therapy versions
    therapy_versions = relationship(
        "TherapyVersion", back_populates="patient", order_by="TherapyVersion.created_at"
    )

    def calculate_age(self):
        """Calculate the patient's age based on the birth date"""
        today = datetime.now()
        age = today.year - self.birth_date.year
        # Correct if the birthday has not yet occurred this year
        if (today.month, today.day) < (self.birth_date.month, self.birth_date.day):
            age -= 1
        return age

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "gender": self.gender,
            "age": self.calculate_age(),
            "birth_date": self.birth_date.isoformat(),
            "medical_conditions": self.medical_conditions,
            "created_at": self.created_at.isoformat(),
        }


class TherapyVersion(Base):
    __tablename__ = "therapy_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    activities = Column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    notes = Column(Text)

    # Relationship with the patient
    patient = relationship("Patient", back_populates="therapy_versions")

    def to_dict(self):
        return {
            "id": self.id,
            "patient_id": self.patient.id if self.patient else None,
            "patient_name": self.patient.name if self.patient else None,
            "created_at": self.created_at.isoformat(),
            "activities": self.activities,
            "notes": self.notes,
        }


# endregion


# region  DATABASE MANAGER


class DatabaseManager:
    def __init__(self, connection_string: str = DB_CONNECTION_STRING):
        """
        Initialize the database

        Args:
            connection_string: string to connect to the PostgreSQL
                               eg: "postgresql://user:password@localhost:5432/dbname"
        """
        self.connection_string = connection_string
        self.engine = None
        self.SessionLocal = None

    def connect(self):
        """Connects to the database and creates the tables if they do no exist"""
        try:
            self.engine = create_engine(
                self.connection_string,
                echo=False,
                pool_pre_ping=True,  # Test connection begore use
                pool_size=5,
                max_overflow=10,
            )
            self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
            # Migration to create tables if not exist
            Base.metadata.create_all(self.engine)
            logger.info("[DB] Connected and tables ensured")
            return True

        except SQLAlchemyError as e:
            logger.error(f"[DB] Connection failed: {e}")
            return False

    def get_session(self):
        """Get database session"""
        if not self.SessionLocal:
            raise RuntimeError("Database is not initialized. Call connect() first.")
        return self.SessionLocal()

    def disconnect(self):
        """Close a connection to a database"""
        if self.engine:
            self.engine.dispose()
            logger.info("[DB] Disconnected")

    # ─── PATIENTS ───────────────────────────────

    def create_patient(
        self,
        name: str,
        gender: str,
        birth_date: datetime,
        medical_conditions: list = None,
    ) -> dict:
        """
        Create a new patient

        Args:
            name: Full name of the patient
            gender: Gender (e.g. "Male", "Female")
            birth_date: Date of birth (datetime object)
            medical_conditions: List of medical conditions (default: [])
        """
        with self.get_session() as session:
            try:
                patient = Patient(
                    name=name,
                    gender=gender,
                    birth_date=birth_date,
                    medical_conditions=medical_conditions or [],
                )
                session.add(patient)
                session.commit()
                session.refresh(patient)
                logger.info(f"[DB] Created patient: {name} (ID: {patient.id})")
                return {"status": "success", "patient": patient.to_dict()}
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"[DB] Error creating patient: {e}")
                return {"status": "error", "message": str(e)}

    def get_patient(self, patient_id: int) -> dict:
        """Retrieve a patient by numeric ID"""
        with self.get_session() as session:
            try:
                patient = session.query(Patient).filter_by(id=patient_id).first()
                if not patient:
                    return {
                        "status": "error",
                        "message": f"Patient with ID {patient_id} not found",
                    }
                return {"status": "success", "patient": patient.to_dict()}
            except SQLAlchemyError as e:
                logger.error(f"[DB] Error getting patient: {e}")
                return {"status": "error", "message": str(e)}

    def get_patient_by_name(self, name: str) -> dict:
        """Retrieve a patient by name (case-insensitive)"""
        with self.get_session() as session:
            try:
                patient = session.query(Patient).filter(Patient.name.ilike(f"%{name}%")).first()
                if not patient:
                    return {
                        "status": "error",
                        "message": f"Patient '{name}' not found",
                    }
                return {"status": "success", "patient": patient.to_dict()}
            except SQLAlchemyError as e:
                logger.error(f"[DB] Error getting patient by name: {e}")
                return {"status": "error", "message": str(e)}

    def get_all_patients(self) -> dict:
        """Retrieve all patients ordered by name"""
        with self.get_session() as session:
            try:
                patients = session.query(Patient).order_by(Patient.name).all()
                logger.info(f"[DB] Retrieved {len(patients)} patients")
                return {
                    "status": "success",
                    "patients": [p.to_dict() for p in patients],
                }
            except SQLAlchemyError as e:
                logger.error(f"[DB] Error getting patients: {e}")
                return {"status": "error", "message": str(e)}

    def update_patient(self, patient_id: int, **updates) -> dict:
        """
        Update patient fields

        Args:
            patient_id: Patient ID
            **updates: Fields to update (name, gender, birth_date, medical_conditions)
        """
        with self.get_session() as session:
            try:
                patient = session.query(Patient).filter_by(id=patient_id).first()
                if not patient:
                    return {
                        "status": "error",
                        "message": f"Patient with ID {patient_id} not found",
                    }

                for key, value in updates.items():
                    if hasattr(patient, key):
                        setattr(patient, key, value)

                session.commit()
                session.refresh(patient)
                logger.info(f"[DB] Updated patient ID {patient_id}")
                return {"status": "success", "patient": patient.to_dict()}
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"[DB] Error updating patient: {e}")
                return {"status": "error", "message": str(e)}

    # Therapies

    def save_therapy_version(self, patient_id: int, activities: list, notes: str = None) -> dict:
        """Save a new therapy version (append-only)"""
        with self.get_session() as session:
            try:
                patient = session.query(Patient).filter_by(id=patient_id).first()
                if not patient:
                    return {
                        "status": "error",
                        "message": f"Patient with ID {patient_id} not found",
                    }

                version = TherapyVersion(
                    patient_id=patient.id,
                    activities=activities,
                    notes=notes,
                    created_at=datetime.now(),
                )
                session.add(version)
                session.commit()
                session.refresh(version)

                logger.info(
                    f"[DB] Saved therapy version {version.id} for patient {patient.name} "
                    f"({len(activities)} activities)"
                )
                return {"status": "success", "version": version.to_dict()}

            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"[DB] Error saving therapy version: {e}")
                return {"status": "error", "message": str(e)}

    def get_latest_therapy(self, patient_id: int) -> dict:
        """Get the latest therapy version for a patient"""
        with self.get_session() as session:
            try:
                patient = session.query(Patient).filter_by(id=patient_id).first()
                if not patient:
                    return {
                        "status": "error",
                        "message": f"Patient with ID {patient_id} not found",
                    }

                therapy = (
                    session.query(TherapyVersion)
                    .filter_by(patient_id=patient.id)
                    .order_by(TherapyVersion.created_at.desc())
                    .first()
                )

                if not therapy:
                    return {
                        "status": "success",
                        "therapy": None,
                        "message": "No therapy saved for this patient",
                    }

                return {"status": "success", "therapy": therapy.to_dict()}

            except SQLAlchemyError as e:
                logger.error(f"[DB] Error getting latest therapy: {e}")
                return {"status": "error", "message": str(e)}

    def get_therapy_history(self, patient_id: int) -> dict:
        """Get the full therapy history for a patient"""
        with self.get_session() as session:
            try:
                patient = session.query(Patient).filter_by(id=patient_id).first()
                if not patient:
                    return {
                        "status": "error",
                        "message": f"Patient with ID {patient_id} not found",
                    }

                versions = (
                    session.query(TherapyVersion)
                    .filter_by(patient_id=patient.id)
                    .order_by(TherapyVersion.created_at.desc())
                    .all()
                )

                logger.info(f"[DB] Retrieved {len(versions)} therapy versions for {patient.name}")
                return {
                    "status": "success",
                    "patient": patient.to_dict(),
                    "total_versions": len(versions),
                    "versions": [v.to_dict() for v in versions],
                }

            except SQLAlchemyError as e:
                logger.error(f"[DB] Error getting therapy history: {e}")
                return {"status": "error", "message": str(e)}

    def load_session(self, patient_id: int) -> dict:
        """
        Load the latest therapy version from the DB and write therapy.json.
        If no version exists, create an empty JSON with only the patient's demographic data.
        """
        therapy_path = THERAPY_FILE
        therapy_path.parent.mkdir(exist_ok=True)

        patient_result = self.get_patient(patient_id)
        if patient_result["status"] == "error":
            logger.error(f"[DB] load_session failed: {patient_result['message']}")
            return patient_result

        patient = patient_result["patient"]
        latest = self.get_latest_therapy(patient_id)

        if latest.get("status") == "error":
            logger.error(f"[DB] load_session failed: {latest.get('message')}")
            return latest

        if latest.get("therapy") is None:
            logger.info(f"[DB] No therapy found for patient ID {patient_id}, creating empty JSON")
            data = {
                "patient_id": patient["id"],
                "patient_full_name": patient["name"],
                "gender": patient["gender"],
                "birth_date": patient["birth_date"],
                "age": patient["age"],
                "medical_conditions": patient["medical_conditions"],
                "activities": [],
            }
        else:
            therapy = latest["therapy"]
            logger.info(f"[DB] Loading therapy version {therapy['id']} for patient ID {patient_id}")

            def _parse_date(d: str) -> datetime:
                """Accept YYYY-MM-DD or ISO 8601 datetime strings."""
                for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
                    try:
                        return datetime.strptime(d, fmt)
                    except ValueError:
                        continue
                raise ValueError(f"Unrecognised date format: {d!r}")

            # .get, not [...]: an activity stored without the key (older rows, or
            # anything written outside add_therapy_activity) made both list
            # comprehensions raise KeyError and took the whole session load down.
            def _is_expired(activity: dict) -> bool:
                valid_until = activity.get("valid_until")
                if not valid_until:
                    return False
                try:
                    return _parse_date(valid_until) <= datetime.now()
                except ValueError:
                    logger.warning(
                        f"[DB] Unparseable valid_until {valid_until!r} on "
                        f"{activity.get('activity_id')} – treating as still valid"
                    )
                    return False

            stored_activities = therapy.get("activities") or []
            valid_activities = [x for x in stored_activities if not _is_expired(x)]
            expired_activities = [x for x in stored_activities if _is_expired(x)]
            data = {
                "patient_id": patient["id"],
                "patient_full_name": patient["name"],
                "gender": patient["gender"],
                "birth_date": patient["birth_date"],
                "age": patient["age"],
                "medical_conditions": patient["medical_conditions"],
                "activities": valid_activities,
                "expired_activities": expired_activities,
            }

        with open(therapy_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        logger.info(f"[DB] therapy.json written: {len(data['activities'])} activities")
        return {"status": "success", "data": data}

    def save_session(self, notes: str = None) -> dict:
        """
        Read the current therapy.json and save a new version to the DB.
        Called at the end of each session (CLI or Streamlit).
        """
        therapy_path = THERAPY_FILE

        if not therapy_path.exists():
            msg = f"therapy.json not found in '{THERAPY_FILE}'"
            logger.warning(f"[DB] save_session skipped: {msg}")
            return {"status": "error", "message": msg}

        try:
            with open(therapy_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            patient_id = data.get("patient_id")
            activities = data.get("activities", [])

            if not patient_id:
                msg = "No patient_id in therapy.json"
                logger.warning(f"[DB] save_session skipped: {msg}")
                return {"status": "error", "message": msg}

            return self.save_therapy_version(
                patient_id=patient_id,
                activities=activities,
                notes=notes or "Auto-saved at session end",
            )

        except Exception as e:
            logger.error(f"[DB] save_session error: {e}")
            return {"status": "error", "message": str(e)}

    def seed_test_data(self, patient_id: str, patients_folder: Path = None) -> dict:
        """
        Insert test data into the database.
        Idempotent: does not create duplicates if run multiple times.
        """

        folder = (patients_folder or PATIENTS_DATA_FOLDER) / str(patient_id)
        therapy_file = folder / "therapy.json"

        # `therapy` used to be assigned only inside the try below, so a patient
        # folder without a therapy.json — which is every one of them in this repo,
        # they only carry history.json — reached `if therapy:` with the name
        # unbound and raised UnboundLocalError. Both callers (main.py and the
        # Streamlit app) invoke this as soon as PostgreSQL answers, so the crash
        # was on the startup path whenever the database was reachable.
        therapy = None

        if therapy_file.exists():
            try:
                therapy = json.loads(therapy_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"[DB] Failed to load therapy file {therapy_file}: {e}")
        else:
            logger.debug(f"[DB] No therapy file found for patient {patient_id} at {therapy_file}")

        if not therapy:
            msg = f"No seed therapy for patient {patient_id} at {therapy_file}"
            logger.info(f"[DB] seed skipped: {msg}")
            return {"status": "skipped", "message": msg}

        name = therapy.get("patient_full_name") or "John Doe"
        result = self.get_patient_by_name(name)

        if result["status"] == "success":
            patient_id = result["patient"]["id"]
            logger.info(f"[DB] seed: patient {name} already exists (ID: {patient_id}), skipping")
        else:
            # Create the patient. strptime raises on an unexpected format rather
            # than returning something falsy, so the default has to be a real
            # fallback around the parse, not an `or` after it.
            raw_birth = therapy.get("birth_date")
            birth_date = datetime(1957, 5, 15)
            if raw_birth:
                for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                    try:
                        birth_date = datetime.strptime(raw_birth, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    logger.warning(
                        f"[DB] seed: unrecognised birth_date {raw_birth!r} for {name} – "
                        f"using {birth_date:%Y-%m-%d}"
                    )
            create_result = self.create_patient(
                name=name,
                gender=therapy.get("gender") or "Unknown",
                birth_date=birth_date,
                medical_conditions=therapy.get("medical_conditions") or [],
            )
            if create_result["status"] == "error":
                return create_result

            patient_id = create_result["patient"]["id"]
            logger.info(f"[DB] seed: patient {name} created (ID: {patient_id})")

        latest = self.get_latest_therapy(patient_id)
        if latest.get("therapy") is not None:
            logger.info("[DB] seed: therapy version already exists, skipping")
            return {"status": "success", "message": "Seed already applied"}

        return self.save_therapy_version(
            patient_id=patient_id,
            activities=therapy.get("activities") or [],
            notes="Initial seed data",
        )


# endregion
