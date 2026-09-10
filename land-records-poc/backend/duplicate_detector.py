"""
Duplicate Detection Module
---------------------------
Backend-enforced duplicate detection for land records.
Checks multiple field combinations with exact + normalized matching.

Used BEFORE creating new records and during cross-validation.
"""
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session
from models import LandRecord


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip whitespace,
    remove extra spaces, remove common punctuation."""
    if not text:
        return ""
    # Unicode normalize
    text = unicodedata.normalize("NFKC", text)
    # Lowercase
    text = text.lower().strip()
    # Remove common punctuation but keep Devanagari/Telugu chars
    text = re.sub(r'[.,;:!?\'\"()\-/\\]', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def check_duplicate(db: Session, fields: Dict[str, str],
                    exclude_id: Optional[int] = None) -> List[Dict]:
    """
    Check for potential duplicate records based on multiple criteria.
    Returns a list of potential matches with match details.

    Checks (in priority order):
    1. Survey Number + Village + Sub-Division (exact)
    2. Khata Number + Village (exact)
    3. Owner Name (normalized) + Survey Number
    4. Survey Number + Village (without sub-division)
    """
    matches = []
    seen_ids = set()

    query = db.query(LandRecord)
    if exclude_id:
        query = query.filter(LandRecord.id != exclude_id)

    survey = fields.get("survey_number", "") or fields.get("khasra_no", "")
    sub_div = fields.get("sub_division", "")
    khata = fields.get("khata_no", "")
    owner = fields.get("owner_name", "")
    village = fields.get("village", "")

    # Check 1: Survey Number + Village + Sub-Division
    if survey and village:
        q = query.filter(
            LandRecord.village == village
        ).filter(
            (LandRecord.survey_number == survey) | (LandRecord.khasra_no == survey)
        )
        if sub_div:
            q = q.filter(LandRecord.sub_division == sub_div)

        for record in q.all():
            if record.id not in seen_ids:
                seen_ids.add(record.id)
                matches.append({
                    "record_id": record.id,
                    "record_id_str": record.record_id_str or f"REC{record.id:03d}",
                    "owner_name": record.owner_name,
                    "survey_number": record.survey_number or record.khasra_no,
                    "village": record.village,
                    "khata_no": record.khata_no,
                    "match_type": "survey_village_exact",
                    "match_score": 95,
                    "match_reason": f"Same survey number ({survey}) in village {village}",
                })

    # Check 2: Khata Number + Village
    if khata and village and not matches:
        q = query.filter(
            LandRecord.khata_no == khata,
            LandRecord.village == village
        )
        for record in q.all():
            if record.id not in seen_ids:
                seen_ids.add(record.id)
                matches.append({
                    "record_id": record.id,
                    "record_id_str": record.record_id_str or f"REC{record.id:03d}",
                    "owner_name": record.owner_name,
                    "survey_number": record.survey_number or record.khasra_no,
                    "village": record.village,
                    "khata_no": record.khata_no,
                    "match_type": "khata_village_exact",
                    "match_score": 85,
                    "match_reason": f"Same khata number ({khata}) in village {village}",
                })

    # Check 3: Normalized owner name + survey number
    if owner and survey and not matches:
        norm_owner = normalize_text(owner)
        all_records = query.filter(
            (LandRecord.survey_number == survey) | (LandRecord.khasra_no == survey)
        ).all()

        for record in all_records:
            if record.id not in seen_ids:
                record_owner = normalize_text(record.owner_name)
                if record_owner and norm_owner and record_owner == norm_owner:
                    seen_ids.add(record.id)
                    matches.append({
                        "record_id": record.id,
                        "record_id_str": record.record_id_str or f"REC{record.id:03d}",
                        "owner_name": record.owner_name,
                        "survey_number": record.survey_number or record.khasra_no,
                        "village": record.village,
                        "khata_no": record.khata_no,
                        "match_type": "owner_survey_normalized",
                        "match_score": 80,
                        "match_reason": f"Same owner ({owner}) with survey number ({survey})",
                    })

    # Check 4: Mutation / Document Number
    mutation = fields.get("mutation_no", "")
    if mutation and not matches:
        q = query.filter(LandRecord.mutation_no == mutation)
        for record in q.all():
            if record.id not in seen_ids:
                seen_ids.add(record.id)
                matches.append({
                    "record_id": record.id,
                    "record_id_str": record.record_id_str or f"REC{record.id:03d}",
                    "owner_name": record.owner_name,
                    "survey_number": record.survey_number or record.khasra_no,
                    "village": record.village,
                    "khata_no": record.khata_no,
                    "match_type": "mutation_exact",
                    "match_score": 90,
                    "match_reason": f"Same mutation/document number ({mutation})",
                })

    return matches


def mark_duplicate(db: Session, record_id: int, status: str = "possible_duplicate"):
    """Mark a record's duplicate status."""
    record = db.query(LandRecord).get(record_id)
    if record:
        record.duplicate_status = status
        db.commit()


def clear_duplicate(db: Session, record_id: int):
    """Clear duplicate flag from a record."""
    record = db.query(LandRecord).get(record_id)
    if record:
        record.duplicate_status = "cleared"
        db.commit()
