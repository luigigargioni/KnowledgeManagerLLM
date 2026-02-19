"""
Modulo per la gestione del database PostgreSQL
Usa SQLAlchemy come ORM con psycopg2 come driver
"""

import json
import logging
from datetime import datetime

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

from config_loader import DB_CONNECTION_STRING, THERAPY_FILE

logger = logging.getLogger("knowledge_manager")


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

    # Relazione con le versioni della terapia
    therapy_versions = relationship(
        "TherapyVersion", back_populates="patient", order_by="TherapyVersion.created_at"
    )

    def calculate_age(self):
        """Calcola l'età del paziente in base alla data di nascita"""
        today = datetime.now()
        age = today.year - self.birth_date.year
        # Correggi se il compleanno non è ancora passato quest'anno
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


class Caregiver(Base):
    __tablename__ = "caregivers"

    id = Column(Integer, primary_key=True, autoincrement=True, unique=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255))
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    # preferences?

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
        }


class TherapyVersion(Base):
    __tablename__ = "therapy_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    activities = Column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    notes = Column(Text)

    # Relazione con il paziente
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
            self.SessionLocal = sessionmaker(
                bind=self.engine, autoflush=False, autocommit=False
            )
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
        Crea un nuovo paziente

        Args:
            name: Nome completo del paziente
            gender: Genere (es: "Male", "Female")
            birth_date: Data di nascita (datetime object)
            medical_conditions: Lista di condizioni mediche (default: [])
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
        """Recupera un paziente dal suo ID numerico"""
        with self.get_session() as session:
            try:
                patient = session.query(Patient).filter_by(id=patient_id).first()
                if not patient:
                    return {
                        "status": "error",
                        "message": f"Paziente con ID {patient_id} non trovato",
                    }
                return {"status": "success", "patient": patient.to_dict()}
            except SQLAlchemyError as e:
                logger.error(f"[DB] Error getting patient: {e}")
                return {"status": "error", "message": str(e)}

    def get_patient_by_name(self, name: str) -> dict:
        """Recupera un paziente dal nome (case-insensitive)"""
        with self.get_session() as session:
            try:
                patient = (
                    session.query(Patient)
                    .filter(Patient.name.ilike(f"%{name}%"))
                    .first()
                )
                if not patient:
                    return {
                        "status": "error",
                        "message": f"Paziente '{name}' non trovato",
                    }
                return {"status": "success", "patient": patient.to_dict()}
            except SQLAlchemyError as e:
                logger.error(f"[DB] Error getting patient by name: {e}")
                return {"status": "error", "message": str(e)}

    def get_all_patients(self) -> dict:
        """Recupera tutti i pazienti ordinati per nome"""
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
        Aggiorna i campi di un paziente

        Args:
            patient_id: ID del paziente
            **updates: Campi da aggiornare (name, gender, birth_date, medical_conditions)
        """
        with self.get_session() as session:
            try:
                patient = session.query(Patient).filter_by(id=patient_id).first()
                if not patient:
                    return {
                        "status": "error",
                        "message": f"Paziente con ID {patient_id} non trovato",
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

    # ─── CAREGIVERS ─────────────────────────────

    def create_caregiver(self, name: str, email: str = None) -> dict:
        with self.get_session() as session:
            try:
                caregiver = Caregiver(name=name, email=email)
                session.add(caregiver)
                session.commit()
                session.refresh(caregiver)
                logger.info(f"[DB] Created caregiver: {name} (ID: {caregiver.id})")
                return {"status": "success", "caregiver": caregiver.to_dict()}
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"[DB] Error creating caregiver: {e}")
                return {"status": "error", "message": str(e)}

    def get_all_caregivers(self) -> dict:
        with self.get_session() as session:
            try:
                caregivers = session.query(Caregiver).order_by(Caregiver.name).all()
                return {
                    "status": "success",
                    "caregivers": [c.to_dict() for c in caregivers],
                }
            except SQLAlchemyError as e:
                logger.error(f"[DB] Error getting caregivers: {e}")
                return {"status": "error", "message": str(e)}

    # Therapies

    def save_therapy_version(
        self, patient_id: int, activities: list, notes: str = None
    ) -> dict:
        """Salva una nuova versione della terapia (append-only)"""
        with self.get_session() as session:
            try:
                patient = session.query(Patient).filter_by(id=patient_id).first()
                if not patient:
                    return {
                        "status": "error",
                        "message": f"Paziente con ID {patient_id} non trovato",
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
        """Ottiene l'ultima versione della terapia per un paziente"""
        with self.get_session() as session:
            try:
                patient = session.query(Patient).filter_by(id=patient_id).first()
                if not patient:
                    return {
                        "status": "error",
                        "message": f"Paziente con ID {patient_id} non trovato",
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
                        "message": "Nessuna terapia salvata per questo paziente",
                    }

                return {"status": "success", "therapy": therapy.to_dict()}

            except SQLAlchemyError as e:
                logger.error(f"[DB] Error getting latest therapy: {e}")
                return {"status": "error", "message": str(e)}

    def get_therapy_history(self, patient_id: int) -> dict:
        """Ottiene tutto lo storico delle terapie per un paziente"""
        with self.get_session() as session:
            try:
                patient = session.query(Patient).filter_by(id=patient_id).first()
                if not patient:
                    return {
                        "status": "error",
                        "message": f"Paziente con ID {patient_id} non trovato",
                    }

                versions = (
                    session.query(TherapyVersion)
                    .filter_by(patient_id=patient.id)
                    .order_by(TherapyVersion.created_at.desc())
                    .all()
                )

                logger.info(
                    f"[DB] Retrieved {len(versions)} therapy versions for {patient.name}"
                )
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
        Carica l'ultima versione della terapia dal DB e scrive il therapy.json.
        Se non esiste nessuna versione, crea un JSON vuoto con i soli dati anagrafici.
        """
        therapy_path = THERAPY_FILE
        therapy_path.parent.mkdir(exist_ok=True)

        patient_result = self.get_patient(patient_id)
        if patient_result["status"] == "error":
            logger.error(f"[DB] load_session failed: {patient_result['message']}")
            return patient_result

        patient = patient_result["patient"]
        latest = self.get_latest_therapy(patient_id)

        if latest["therapy"] is None:
            logger.info(
                f"[DB] No therapy found for patient ID {patient_id}, creating empty JSON"
            )
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
            logger.info(
                f"[DB] Loading therapy version {therapy['id']} for patient ID {patient_id}"
            )
            valid_activities = [
                x
                for x in therapy["activities"]
                if not x["valid_until"]
                or datetime.strptime(x["valid_until"], "%Y-%m-%d") > datetime.now()
            ]
            expired_activities = [
                x
                for x in therapy["activities"]
                if x["valid_until"]
                and datetime.strptime(x["valid_until"], "%Y-%m-%d") <= datetime.now()
            ]
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
        Legge il therapy.json corrente e salva una nuova versione nel DB.
        Chiamato alla fine di ogni sessione (CLI o Streamlit).
        """
        therapy_path = THERAPY_FILE

        if not therapy_path.exists():
            msg = f"therapy.json non trovato in '{THERAPY_FILE}'"
            logger.warning(f"[DB] save_session skipped: {msg}")
            return {"status": "error", "message": msg}

        try:
            with open(therapy_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            patient_id = data.get("patient_id")
            activities = data.get("activities", [])

            if not patient_id:
                msg = "Nessun patient_id nel therapy.json"
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

    def seed_test_data(self) -> dict:
        """
        Inserisce dati di test nel database.
        Idempotente: non crea duplicati se eseguita più volte.
        """
        result = self.get_patient_by_name("Mario Rossi")

        if result["status"] == "success":
            patient_id = result["patient"]["id"]
            logger.info(
                f"[DB] seed: patient Mario Rossi already exists (ID: {patient_id}), skipping"
            )
        else:
            # Crea il paziente
            birth_date = datetime(1957, 5, 15)
            create_result = self.create_patient(
                name="Mario Rossi",
                gender="Male",
                birth_date=birth_date,
                medical_conditions=[
                    "Diabete di tipo 1",
                    "Celiachia",
                    "Forte insufficienza renale",
                ],
            )
            if create_result["status"] == "error":
                return create_result

            patient_id = create_result["patient"]["id"]
            logger.info(f"[DB] seed: patient Mario Rossi created (ID: {patient_id})")

        latest = self.get_latest_therapy(patient_id)
        if latest["therapy"] is not None:
            logger.info("[DB] seed: therapy version already exists, skipping")
            return {"status": "success", "message": "Seed already applied"}

        activities = [
            {
                "activity_id": "lb_001",
                "name": "Misurazione glicemia",
                "description": "Controllo glicemia a digiuno",
                "day_of_week": [1, 3, 5],
                "time": "07:30",
                "duration_minutes": 10,
                "dependencies": [],
                "valid_from": None,
                "valid_until": None,
            },
            {
                "activity_id": "lb_002",
                "name": "Colazione",
                "description": "Colazione salata con proteine",
                "day_of_week": [1, 3, 5],
                "time": "08:00",
                "duration_minutes": 20,
                "dependencies": ["Misurazione glicemia"],
                "valid_from": None,
                "valid_until": None,
            },
            {
                "activity_id": "lw_001",
                "name": "Light walk",
                "description": "A light 1-hour walk in the morning.",
                "day_of_week": [1],
                "time": "08:35",
                "duration_minutes": 60,
                "dependencies": [],
                "valid_from": None,
                "valid_until": None,
            },
            {
                "activity_id": "lu_001",
                "name": "Pranzo",
                "description": "",
                "day_of_week": [1, 2, 3, 4, 5, 6, 7],
                "time": "12:00",
                "duration_minutes": 60,
                "dependencies": [],
                "valid_from": None,
                "valid_until": None,
            },
            {
                "activity_id": "ta_001",
                "name": "Take Tachipirina",
                "description": "Take 1000 mg Tachipirina tablet once daily.",
                "day_of_week": [1, 2, 3, 4, 5, 6, 7],
                "time": "13:00",
                "duration_minutes": 5,
                "dependencies": [],
                "valid_from": None,
                "valid_until": None,
            },
        ]

        result = self.save_therapy_version(
            patient_id=patient_id, activities=activities, notes="Initial seed data"
        )
        return result


# endregion
