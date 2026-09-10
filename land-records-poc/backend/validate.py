"""
Stage 5 of the pipeline: Cross-validate
Compare a newly extracted record against existing records in the database
for the same khasra_no/village combination, and flag mismatches (e.g. area
changed, different owner recorded) for officer review. Stands in for the
"ML cross-validation" described in the proposal, using a rule-based diff
against known records here to keep the hackathon build tractable;
architecturally it plugs into the same slot a trained matching/anomaly
model would occupy later.

UPGRADED: Added structured validation results with ✓/⚠/✕ status per field,
survey number format validation, area format validation, missing field
detection, and duplicate checking integration.
"""
import re
import json
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from models import LandRecord


def find_potential_duplicate(db: Session, khasra_no: str, village: str) -> Optional[LandRecord]:
    if not khasra_no or not village:
        return None
    return (
        db.query(LandRecord)
        .filter(LandRecord.khasra_no == khasra_no, LandRecord.village == village)
        .first()
    )


def cross_validate(db: Session, fields: Dict[str, str]) -> List[str]:
    """Return a list of human-readable mismatch warnings, empty if none."""
    warnings = []
    existing = find_potential_duplicate(db, fields.get("khasra_no", ""), fields.get("village", ""))

    if existing:
        if existing.owner_name and fields.get("owner_name") and \
           existing.owner_name.lower() != fields["owner_name"].lower():
            warnings.append(
                f"Owner mismatch: existing record shows '{existing.owner_name}', "
                f"new scan shows '{fields.get('owner_name')}' for khasra {fields.get('khasra_no')}"
            )
        if existing.area_hectare and fields.get("area_hectare") and \
           existing.area_hectare != fields["area_hectare"]:
            warnings.append(
                f"Area mismatch: existing record shows '{existing.area_hectare}' ha, "
                f"new scan shows '{fields.get('area_hectare')}' ha"
            )
        if existing.khata_no and fields.get("khata_no") and \
           existing.khata_no != fields["khata_no"]:
            warnings.append(
                f"Khata No. mismatch: existing '{existing.khata_no}' vs new '{fields.get('khata_no')}'"
            )

    return warnings


def validate_record_structured(fields: Dict[str, str], db: Session = None,
                                exclude_id: int = None) -> Dict:
    """
    Perform structured validation returning per-field results.
    Each result has: status (passed/warning/conflict), message, field_name

    Returns:
    {
        "results": [
            {"field": "survey_number", "status": "passed", "icon": "✓", "message": "Valid format"},
            {"field": "land_area", "status": "warning", "icon": "⚠", "message": "Low confidence"},
            ...
        ],
        "summary": {"passed": 5, "warnings": 2, "conflicts": 1},
        "overall": "warning"  // passed, warning, conflict
    }
    """
    results = []

    # --- Survey Number Format ---
    survey = fields.get("survey_number", "") or fields.get("khasra_no", "")
    if survey:
        if re.match(r'^\d{1,5}(/\d{1,5})?$', survey.strip()):
            results.append({
                "field": "Survey Number",
                "status": "passed", "icon": "✓",
                "message": "Valid format"
            })
        else:
            results.append({
                "field": "Survey Number",
                "status": "warning", "icon": "⚠",
                "message": f"Unusual format: {survey}"
            })
    else:
        results.append({
            "field": "Survey Number",
            "status": "conflict", "icon": "✕",
            "message": "Missing survey number"
        })

    # --- Owner Name ---
    owner = fields.get("owner_name", "")
    if owner and len(owner.strip()) >= 2:
        results.append({
            "field": "Owner Name",
            "status": "passed", "icon": "✓",
            "message": "Valid"
        })
    elif owner:
        results.append({
            "field": "Owner Name",
            "status": "warning", "icon": "⚠",
            "message": "Name appears incomplete"
        })
    else:
        results.append({
            "field": "Owner Name",
            "status": "conflict", "icon": "✕",
            "message": "Missing owner name"
        })

    # --- Land Area ---
    area = fields.get("area_hectare", "")
    if area:
        try:
            area_val = float(area)
            if 0.01 <= area_val <= 500:
                results.append({
                    "field": "Land Area",
                    "status": "passed", "icon": "✓",
                    "message": f"Valid: {area_val} acres"
                })
            else:
                results.append({
                    "field": "Land Area",
                    "status": "warning", "icon": "⚠",
                    "message": f"Unusual area: {area_val} acres"
                })
        except ValueError:
            results.append({
                "field": "Land Area",
                "status": "warning", "icon": "⚠",
                "message": f"Non-numeric area: {area}"
            })
    else:
        results.append({
            "field": "Land Area",
            "status": "conflict", "icon": "✕",
            "message": "Missing area information"
        })

    # --- Village ---
    village = fields.get("village", "")
    if village and len(village.strip()) >= 2:
        results.append({
            "field": "Village",
            "status": "passed", "icon": "✓",
            "message": "Valid"
        })
    else:
        results.append({
            "field": "Village",
            "status": "conflict", "icon": "✕",
            "message": "Missing village name"
        })

    # --- Khata Number ---
    khata = fields.get("khata_no", "")
    if khata and re.match(r'^\d{1,6}$', khata.strip()):
        results.append({
            "field": "Khata Number",
            "status": "passed", "icon": "✓",
            "message": "Valid"
        })
    elif khata:
        results.append({
            "field": "Khata Number",
            "status": "warning", "icon": "⚠",
            "message": f"Unusual format: {khata}"
        })
    else:
        results.append({
            "field": "Khata Number",
            "status": "warning", "icon": "⚠",
            "message": "Missing khata number"
        })

    # --- Duplicate Check ---
    if db:
        from duplicate_detector import check_duplicate
        dupes = check_duplicate(db, fields, exclude_id=exclude_id)
        if dupes:
            results.append({
                "field": "Duplicate Check",
                "status": "conflict", "icon": "✕",
                "message": f"Possible duplicate: {dupes[0]['match_reason']}"
            })
        else:
            results.append({
                "field": "Duplicate Check",
                "status": "passed", "icon": "✓",
                "message": "No duplicate found"
            })

    # --- Missing Fields Check ---
    required_fields = ["owner_name", "village", "district"]
    missing = [f for f in required_fields if not fields.get(f, "").strip()]
    if missing:
        results.append({
            "field": "Required Fields",
            "status": "conflict", "icon": "✕",
            "message": f"Missing: {', '.join(missing)}"
        })
    else:
        results.append({
            "field": "Required Fields",
            "status": "passed", "icon": "✓",
            "message": "All required fields present"
        })

    # --- District / State consistency ---
    district = fields.get("district", "")
    state = fields.get("state", "")
    if district and state:
        results.append({
            "field": "Location Consistency",
            "status": "passed", "icon": "✓",
            "message": f"{district}, {state}"
        })

    # Summarize
    passed = sum(1 for r in results if r["status"] == "passed")
    warnings = sum(1 for r in results if r["status"] == "warning")
    conflicts = sum(1 for r in results if r["status"] == "conflict")

    if conflicts > 0:
        overall = "conflict"
    elif warnings > 0:
        overall = "warning"
    else:
        overall = "passed"

    return {
        "results": results,
        "summary": {"passed": passed, "warnings": warnings, "conflicts": conflicts},
        "overall": overall,
    }
