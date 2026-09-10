"""
Stage 7 of the pipeline: Publish record
Database schema for storing digitized, officer-verified land records.

UPGRADED: Added fields for submission tracking, duplicate detection,
multi-language support, geo-coordinates, audit trail, and village
demo dataset support.
"""
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./land_records.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class LandRecord(Base):
    __tablename__ = "land_records"

    id = Column(Integer, primary_key=True, index=True)
    record_id_str = Column(String, unique=True, index=True)  # e.g. "REC001"
    filename = Column(String, default="")

    # --- Core identity fields (original + expanded) ---
    owner_name = Column(String, default="")
    father_name = Column(String, default="")
    survey_number = Column(String, default="")
    sub_division = Column(String, default="")
    khasra_no = Column(String, default="")
    khata_no = Column(String, default="")

    # --- Location fields ---
    village = Column(String, default="")
    mandal = Column(String, default="")  # Mandal/Taluk
    tehsil = Column(String, default="")
    district = Column(String, default="")
    state = Column(String, default="")

    # --- Land details ---
    area_hectare = Column(String, default="")
    land_type = Column(String, default="")        # Agricultural, Residential, etc.
    classification = Column(String, default="")
    mutation_no = Column(String, default="")

    # --- Document metadata ---
    document_type = Column(String, default="")     # Khatauni, Patta, RoR, etc.
    document_source = Column(String, default="")   # Scanned, Digital, Legacy PDF
    language = Column(String, default="english")   # english, telugu, hindi, mixed

    # --- Status tracking ---
    submission_status = Column(String, default="not_submitted")  # submitted, pending, not_submitted
    verification_status = Column(String, default="pending")      # verified, under_review, pending, rejected, published
    duplicate_status = Column(String, default="none")            # none, possible_duplicate, confirmed_duplicate, cleared

    # --- Confidence & validation ---
    field_confidence = Column(Text, default="")
    overall_confidence = Column(Float, default=0.0)
    ocr_confidence = Column(Float, default=0.0)
    status = Column(String, default="pending_review")  # pending_review | verified (legacy compat)
    validation_warnings = Column(Text, default="")
    validation_results = Column(Text, default="")  # JSON: structured validation results

    # --- Geo-coordinates (illustrative) ---
    latitude = Column(Float, default=0.0)
    longitude = Column(Float, default=0.0)

    # --- Timestamps ---
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_by = Column(String, default="")

    # --- Processing metadata ---
    ocr_text = Column(Text, default="")
    processing_stage = Column(String, default="")  # Current pipeline stage
    source = Column(String, default="user_upload", index=True)  # "demo" | "user_upload"
    file_hash = Column(String, default="", index=True)          # SHA-256 of raw uploaded bytes


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    record_id = Column(Integer, index=True)
    record_id_str = Column(String, default="")
    action = Column(String, nullable=False)      # uploaded, ocr_completed, fields_extracted, etc.
    details = Column(Text, default="")
    performed_by = Column(String, default="system")
    timestamp = Column(DateTime, default=datetime.utcnow)


def init_db():
    from sqlalchemy import text
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        for col_name, col_def in [("source", "VARCHAR DEFAULT 'user_upload'"), ("file_hash", "VARCHAR DEFAULT ''")]:
            try:
                conn.execute(text(f"ALTER TABLE land_records ADD COLUMN {col_name} {col_def}"))
                conn.commit()
            except Exception:
                pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def add_audit_entry(db, record_id: int, record_id_str: str, action: str,
                    details: str = "", performed_by: str = "system"):
    """Helper to add an audit log entry."""
    entry = AuditLog(
        record_id=record_id,
        record_id_str=record_id_str,
        action=action,
        details=details,
        performed_by=performed_by,
    )
    db.add(entry)
    db.commit()
    return entry
