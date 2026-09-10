"""
Demo Village Dataset Generator — 300 Landholders
--------------------------------------------------
Generates a deterministic (seeded) dataset of 300 demo landholders
for "Rampur Village" with realistic Indian names, survey numbers,
and configurable submission status distribution.

All data is clearly DEMO — no real personal information is used.

Usage:
    from seed_data import generate_village_data, load_demo_data
    records = generate_village_data()
    load_demo_data(db_session)
"""
import random
from datetime import datetime, timedelta
from typing import List, Dict

# ---- CONFIGURABLE PARAMETERS ----

VILLAGE_NAME = "Rampur"
MANDAL_NAME = "Tenali"
DISTRICT_NAME = "Guntur"
STATE_NAME = "Andhra Pradesh"

TOTAL_RECORDS = 300

# Submission status distribution (approximate percentages)
SUBMITTED_PCT = 0.60    # 55-65%
PENDING_PCT = 0.20      # 15-25%
NOT_SUBMITTED_PCT = 0.20  # 15-25%

# Village center coordinates (illustrative — Guntur district, AP)
VILLAGE_CENTER_LAT = 16.3050
VILLAGE_CENTER_LNG = 80.4370
VILLAGE_SPREAD = 0.025  # coordinate jitter radius

# ---- NAME DATA (demo only, not real people) ----

FIRST_NAMES_MALE = [
    "Ramesh", "Suresh", "Venkatesh", "Krishna", "Ravi",
    "Srinivas", "Narayana", "Lakshman", "Gopal", "Vijay",
    "Anil", "Sunil", "Manoj", "Rajesh", "Mahesh",
    "Ashok", "Vinod", "Pramod", "Satish", "Ganesh",
    "Harish", "Dinesh", "Umesh", "Mohan", "Sohan",
    "Prabhakar", "Shankar", "Bhaskar", "Sekhar", "Chandra",
    "Sai", "Srinu", "Raju", "Babu", "Rao",
    "Naidu", "Prasad", "Murali", "Kiran", "Praveen",
    "Anand", "Deepak", "Ramana", "Siva", "Venu",
]

FIRST_NAMES_FEMALE = [
    "Lakshmi", "Saraswati", "Padma", "Radha", "Sita",
    "Gita", "Anita", "Sunita", "Kamala", "Vijaya",
    "Durga", "Parvati", "Savitri", "Rukmini", "Tulasi",
    "Manga", "Ratna", "Vani", "Jaya", "Rani",
    "Shanti", "Bharati", "Latha", "Aruna", "Pushpa",
]

LAST_NAMES = [
    "Reddy", "Naidu", "Rao", "Kumar", "Singh",
    "Yadav", "Sharma", "Verma", "Gupta", "Prasad",
    "Devi", "Chowdary", "Patel", "Varma", "Setty",
    "Goud", "Mudiraj", "Kamma", "Raju", "Pillai",
]

FATHER_NAMES_MALE = [
    "Venkaiah", "Ramaiah", "Subbaiah", "Narasimha", "Lakshmaiah",
    "Gangaiah", "Mallaiah", "Bala", "Chinna", "Pedda",
    "Kotaiah", "Apparao", "Satyam", "Dharma", "Bheem",
]

# Telugu names for language variety
TELUGU_NAMES = [
    "వెంకటేష్ రెడ్డి", "లక్ష్మీ నాయుడు", "రామారావు",
    "సీతమ్మ", "సుబ్బారావు", "నరసింహ రావు",
    "కృష్ణ ప్రసాద్", "పద్మావతి", "గోపాలరావు",
    "వేణు గోపాల్", "శ్రీనివాస రావు", "రుక్మిణి దేవి",
]

HINDI_NAMES = [
    "राम प्रसाद शर्मा", "सीता देवी", "मोहन लाल",
    "गीता वर्मा", "राजेश कुमार", "सुनीता गुप्ता",
    "विनोद यादव", "कमला सिंह", "अशोक तिवारी",
    "प्रभा देवी", "महेश चौधरी", "शशि शर्मा",
]

LAND_TYPES = [
    "Agricultural", "Agricultural", "Agricultural", "Agricultural",
    "Residential", "Residential",
    "Commercial",
    "Irrigated", "Irrigated",
    "Barren",
    "Wetland",
    "Garden/Orchard",
]

DOCUMENT_TYPES = [
    "Khatauni", "Khatauni", "Khatauni",
    "Patta", "Patta",
    "RoR Extract", "RoR Extract",
    "Mutation Record",
    "Sale Deed",
    "Adangal",
    "Pahani",
]

DOCUMENT_SOURCES = [
    "Scanned Document", "Scanned Document", "Scanned Document",
    "Digital Upload",
    "Legacy PDF", "Legacy PDF",
    "Photographed Document",
    "Revenue Office Copy",
]

LANGUAGES = [
    "Telugu", "Telugu", "Telugu", "Telugu", "Telugu",   # ~45%
    "English", "English", "English", "English",          # ~35%
    "Hindi", "Hindi",                                     # ~20%
]

VERIFICATION_STATUSES_FOR_SUBMITTED = [
    "verified", "verified", "verified",
    "under_review",
    "published", "published",
]

VERIFICATION_STATUSES_FOR_PENDING = [
    "pending", "pending",
    "under_review",
]


def _seeded_random():
    """Return a seeded random instance for deterministic generation."""
    return random.Random(42)


def generate_village_data() -> List[Dict]:
    """Generate 300 deterministic demo records for the village."""
    rng = _seeded_random()
    records = []

    # Calculate exact counts
    n_submitted = int(TOTAL_RECORDS * SUBMITTED_PCT)
    n_pending = int(TOTAL_RECORDS * PENDING_PCT)
    n_not_submitted = TOTAL_RECORDS - n_submitted - n_pending

    # Create status list and shuffle deterministically
    statuses = (
        ["submitted"] * n_submitted +
        ["pending"] * n_pending +
        ["not_submitted"] * n_not_submitted
    )
    rng.shuffle(statuses)

    # Track survey numbers to create realistic distribution
    used_surveys = set()

    for i in range(TOTAL_RECORDS):
        record_num = i + 1
        record_id_str = f"REC{record_num:03d}"
        submission_status = statuses[i]

        # Choose language for this record
        lang = rng.choice(LANGUAGES)

        # Generate name based on language
        if lang == "Telugu" and rng.random() < 0.3:
            owner_name = rng.choice(TELUGU_NAMES)
            father_name = rng.choice(FATHER_NAMES_MALE) + " " + rng.choice(LAST_NAMES)
        elif lang == "Hindi" and rng.random() < 0.3:
            owner_name = rng.choice(HINDI_NAMES)
            father_name = rng.choice(FATHER_NAMES_MALE) + " " + rng.choice(LAST_NAMES)
        else:
            if rng.random() < 0.7:
                first = rng.choice(FIRST_NAMES_MALE)
            else:
                first = rng.choice(FIRST_NAMES_FEMALE)
            last = rng.choice(LAST_NAMES)
            owner_name = f"{first} {last}"
            father_name = f"{rng.choice(FATHER_NAMES_MALE)} {last}"

        # Generate survey number (some will share survey numbers with different sub-divisions)
        base_survey = rng.randint(1, 250)
        sub_div_num = rng.choice(["", "A", "B", "C", "1", "2", "3"])
        survey_number = f"{base_survey}/{rng.randint(1, 20)}" if rng.random() < 0.4 else str(base_survey)

        khata_no = str(rng.randint(100, 9999))
        khasra_no = str(rng.randint(100, 999))

        # Land area (in acres for AP context)
        area = round(rng.uniform(0.25, 12.0), 2)
        area_str = f"{area}"

        land_type = rng.choice(LAND_TYPES)
        doc_type = rng.choice(DOCUMENT_TYPES)
        doc_source = rng.choice(DOCUMENT_SOURCES)

        # OCR confidence (realistic distribution)
        if submission_status == "submitted":
            ocr_conf = round(rng.uniform(70, 99), 1)
        elif submission_status == "pending":
            ocr_conf = round(rng.uniform(55, 92), 1)
        else:
            ocr_conf = round(rng.uniform(0, 85), 1)

        # Verification status based on submission status
        if submission_status == "submitted":
            verification = rng.choice(VERIFICATION_STATUSES_FOR_SUBMITTED)
        elif submission_status == "pending":
            verification = rng.choice(VERIFICATION_STATUSES_FOR_PENDING)
        else:
            verification = "not_applicable"

        # Duplicate status & realistic duplicate records for demo
        val_warnings = ""
        if len(records) > 5 and rng.random() < 0.06:
            prior = rng.choice(records)
            duplicate_status = "possible_duplicate"
            survey_number = prior["survey_number"]
            sub_div_num = prior.get("sub_division", "")
            khasra_no = prior.get("khasra_no", "")
            village = prior["village"]
            val_warnings = f"Possible duplicate of {prior['record_id_str']}: Matching survey number {survey_number} in village {village}"
        else:
            duplicate_status = "none"

        # Coordinates (jittered around village center)
        lat = VILLAGE_CENTER_LAT + rng.uniform(-VILLAGE_SPREAD, VILLAGE_SPREAD)
        lng = VILLAGE_CENTER_LNG + rng.uniform(-VILLAGE_SPREAD, VILLAGE_SPREAD)

        # Timestamps
        days_ago = rng.randint(1, 180)
        created = datetime.utcnow() - timedelta(days=days_ago)
        updated = created + timedelta(hours=rng.randint(1, 48))

        # Legacy compat status
        if submission_status == "submitted" and verification in ("verified", "published"):
            legacy_status = "verified"
        else:
            legacy_status = "pending_review"

        # Processing stage
        if submission_status == "not_submitted":
            stage = "not_submitted"
        elif submission_status == "pending":
            stages = ["uploaded", "ocr_processing", "field_extraction", "validation"]
            stage = rng.choice(stages)
        else:
            if verification == "published":
                stage = "published"
            elif verification == "verified":
                stage = "verified"
            else:
                stage = "officer_review"

        record = {
            "record_id_str": record_id_str,
            "filename": "",
            "owner_name": owner_name,
            "father_name": father_name,
            "survey_number": survey_number,
            "sub_division": sub_div_num,
            "khasra_no": khasra_no,
            "khata_no": khata_no,
            "village": VILLAGE_NAME,
            "mandal": MANDAL_NAME,
            "tehsil": MANDAL_NAME,
            "district": DISTRICT_NAME,
            "state": STATE_NAME,
            "area_hectare": area_str,
            "land_type": land_type,
            "classification": land_type,
            "mutation_no": f"M-{rng.randint(1000, 9999)}",
            "document_type": doc_type,
            "document_source": doc_source,
            "language": lang.lower(),
            "submission_status": submission_status,
            "verification_status": verification,
            "duplicate_status": duplicate_status,
            "overall_confidence": ocr_conf,
            "ocr_confidence": ocr_conf,
            "field_confidence": "",
            "status": legacy_status,
            "validation_warnings": val_warnings,
            "latitude": round(lat, 6),
            "longitude": round(lng, 6),
            "created_at": created,
            "last_updated": updated,
            "processing_stage": stage,
            "source": "demo",
            "file_hash": "",
        }
        records.append(record)

    return records


def load_demo_data(db) -> int:
    """Load 300 demo records into the database. Returns count loaded."""
    from models import LandRecord, AuditLog

    # Clear existing demo data
    db.query(AuditLog).delete()
    db.query(LandRecord).delete()
    db.commit()

    records = generate_village_data()
    count = 0

    for rec_data in records:
        record = LandRecord(**rec_data)
        db.add(record)
        db.flush()

        # Add audit entries for records that have progressed
        if rec_data["submission_status"] != "not_submitted":
            audit_time = rec_data["created_at"]
            db.add(AuditLog(
                record_id=record.id,
                record_id_str=rec_data["record_id_str"],
                action="document_uploaded",
                details=f"Document uploaded ({rec_data['document_source']})",
                performed_by="system",
                timestamp=audit_time,
            ))
            if rec_data["processing_stage"] not in ("uploaded", "not_submitted"):
                db.add(AuditLog(
                    record_id=record.id,
                    record_id_str=rec_data["record_id_str"],
                    action="ocr_completed",
                    details=f"OCR completed. Language: {rec_data['language']}. Confidence: {rec_data['ocr_confidence']}%",
                    performed_by="system",
                    timestamp=audit_time + timedelta(minutes=1),
                ))
                db.add(AuditLog(
                    record_id=record.id,
                    record_id_str=rec_data["record_id_str"],
                    action="fields_extracted",
                    details="Structured fields extracted from OCR text",
                    performed_by="system",
                    timestamp=audit_time + timedelta(minutes=2),
                ))
            if rec_data["processing_stage"] in ("validation", "officer_review", "verified", "published"):
                db.add(AuditLog(
                    record_id=record.id,
                    record_id_str=rec_data["record_id_str"],
                    action="validation_completed",
                    details="Cross-validation checks completed",
                    performed_by="system",
                    timestamp=audit_time + timedelta(minutes=3),
                ))
            if rec_data["verification_status"] in ("verified", "published"):
                db.add(AuditLog(
                    record_id=record.id,
                    record_id_str=rec_data["record_id_str"],
                    action="record_approved",
                    details="Record verified and approved by officer",
                    performed_by="officer_demo",
                    timestamp=audit_time + timedelta(minutes=5),
                ))
            if rec_data["verification_status"] == "published":
                db.add(AuditLog(
                    record_id=record.id,
                    record_id_str=rec_data["record_id_str"],
                    action="record_published",
                    details="Record published to digital ledger",
                    performed_by="officer_demo",
                    timestamp=audit_time + timedelta(minutes=7),
                ))

        count += 1

    db.commit()
    return count


def generate_single_sample(db) -> dict:
    """Generate and insert a single new sample record."""
    from models import LandRecord, AuditLog
    rng = random.Random()

    # Find next record number
    max_rec = db.query(LandRecord).order_by(LandRecord.id.desc()).first()
    next_num = (max_rec.id + 1) if max_rec else 1
    record_id_str = f"REC{next_num:03d}"

    first = rng.choice(FIRST_NAMES_MALE + FIRST_NAMES_FEMALE)
    last = rng.choice(LAST_NAMES)

    record = LandRecord(
        record_id_str=record_id_str,
        owner_name=f"{first} {last}",
        father_name=f"{rng.choice(FATHER_NAMES_MALE)} {last}",
        survey_number=str(rng.randint(1, 250)),
        sub_division=rng.choice(["", "A", "B"]),
        khasra_no=str(rng.randint(100, 999)),
        khata_no=str(rng.randint(100, 9999)),
        village=VILLAGE_NAME,
        mandal=MANDAL_NAME,
        tehsil=MANDAL_NAME,
        district=DISTRICT_NAME,
        state=STATE_NAME,
        area_hectare=str(round(rng.uniform(0.25, 8.0), 2)),
        land_type=rng.choice(LAND_TYPES),
        classification=rng.choice(LAND_TYPES),
        document_type=rng.choice(DOCUMENT_TYPES),
        document_source="Digital Upload",
        language=rng.choice(["english", "telugu", "hindi"]),
        submission_status="pending",
        verification_status="pending",
        duplicate_status="none",
        overall_confidence=round(rng.uniform(60, 95), 1),
        ocr_confidence=round(rng.uniform(60, 95), 1),
        status="pending_review",
        latitude=round(VILLAGE_CENTER_LAT + rng.uniform(-VILLAGE_SPREAD, VILLAGE_SPREAD), 6),
        longitude=round(VILLAGE_CENTER_LNG + rng.uniform(-VILLAGE_SPREAD, VILLAGE_SPREAD), 6),
        processing_stage="uploaded",
        source="demo",
        file_hash="",
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    db.add(AuditLog(
        record_id=record.id,
        record_id_str=record_id_str,
        action="record_created",
        details="Sample record generated via demo controls",
        performed_by="demo_system",
    ))
    db.commit()

    return {"id": record.id, "record_id_str": record_id_str, "owner_name": record.owner_name}
