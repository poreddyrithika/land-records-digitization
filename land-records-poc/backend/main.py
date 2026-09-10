"""
Intelligent Land Record Digitization and Validation System - Backend API
Implements the 7-stage pipeline from the proposal:
  1. Scan & pre-process   -> preprocess.py
  2. Detect type & layout -> (single template in this PoC; see NOTE below)
  3. OCR + handwriting    -> ocr_engine.py
  4. NLP field mapping    -> extract.py
  5. Cross-validate       -> validate.py
  6. Officer review       -> PATCH /records/{id}/review
  7. Publish record       -> models.py (LandRecord table)

NOTE on stage 2 (layout detection): this PoC assumes a single khatauni
layout (the one gen_data.py produces). In production this is where
LayoutLMv3 classifies the document type before OCR; swapping it in means
adding a classifier call before preprocess_image() and branching the label
schema per detected type. Left out here to keep the hackathon build
tractable — call this out explicitly if a judge asks.

UPGRADED: Added demo dataset loading, duplicate detection, enhanced
validation, multi-language OCR support, audit trail, statistics API,
map API, record lifecycle management, and PDF support.
"""
import os
import json
import shutil
import uuid
import hashlib
from datetime import datetime
from typing import List, Optional, Union

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func, or_

from extract import extract_fields, overall_confidence, CONFIDENCE_LOW_THRESHOLD
from models import LandRecord, AuditLog, get_db, init_db, add_audit_entry
from hybrid_ocr import run_hybrid_ocr
from preprocess import preprocess_image, preprocess_with_fallback
from validate import cross_validate, validate_record_structured
from duplicate_detector import check_duplicate, normalize_text
from ocr_engine import run_ocr, words_to_lines, detect_language_from_text, get_full_ocr_text
from map_extractor import process_land_map
from handwriting_detector import analyze_document_handwriting, get_all_dataset_benchmarks
from gemini_chat import answer_record_question

# Set to False to skip TrOCR entirely and use Tesseract-only (fast, no
# torch/transformers dependency needed). Set True once handwriting support
# is installed and tested — see handwriting_ocr.py for setup notes.
ENABLE_HANDWRITING_OCR = False

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Intelligent Land Record Digitization API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

init_db()



SCHEMA_FIELDS = [
    "owner_name", "father_name", "survey_number", "sub_division",
    "khasra_no", "khata_no", "area_hectare", "village", "mandal",
    "tehsil", "district", "state", "land_type", "classification", "mutation_no",
]

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".pdf", ".gif", ".webp"}


# ---- Pydantic Models ----

class ReviewPayload(BaseModel):
    corrections: dict          # field_name -> corrected value
    reviewed_by: str = "officer_demo"


class RecordOut(BaseModel):
    id: int
    record_id_str: Optional[str] = ""
    filename: str
    owner_name: str
    father_name: Optional[str] = ""
    survey_number: Optional[str] = ""
    sub_division: Optional[str] = ""
    khasra_no: str
    khata_no: str
    area_hectare: str
    village: str
    mandal: Optional[str] = ""
    tehsil: str
    district: str
    state: Optional[str] = ""
    classification: str
    mutation_no: str
    land_type: Optional[str] = ""
    document_type: Optional[str] = ""
    document_source: Optional[str] = ""
    language: Optional[str] = "english"
    submission_status: Optional[str] = "not_submitted"
    verification_status: Optional[str] = "pending"
    duplicate_status: Optional[str] = "none"
    overall_confidence: float
    ocr_confidence: Optional[float] = 0.0
    field_confidence: str
    status: str
    validation_warnings: str
    validation_results: Optional[str] = ""
    latitude: Optional[float] = 0.0
    longitude: Optional[float] = 0.0
    processing_stage: Optional[str] = ""
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = ""
    source: Optional[str] = "user_upload"
    file_hash: Optional[str] = ""
    is_file_duplicate: Optional[bool] = False
    duplicate_of_id_str: Optional[str] = None
    duplicate_message: Optional[str] = None

    class Config:
        from_attributes = True


class RecordCreate(BaseModel):
    owner_name: str = ""
    father_name: str = ""
    survey_number: str = ""
    sub_division: str = ""
    khasra_no: str = ""
    khata_no: str = ""
    village: str = ""
    mandal: str = ""
    tehsil: str = ""
    district: str = ""
    state: str = ""
    area_hectare: str = ""
    land_type: str = ""
    classification: str = ""
    mutation_no: str = ""
    document_type: str = ""
    document_source: str = ""
    language: str = "english"
    submission_status: str = "pending"
    latitude: float = 0.0
    longitude: float = 0.0


class RecordUpdate(BaseModel):
    owner_name: Optional[str] = None
    father_name: Optional[str] = None
    survey_number: Optional[str] = None
    sub_division: Optional[str] = None
    khasra_no: Optional[str] = None
    khata_no: Optional[str] = None
    village: Optional[str] = None
    mandal: Optional[str] = None
    tehsil: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    area_hectare: Optional[str] = None
    land_type: Optional[str] = None
    classification: Optional[str] = None
    mutation_no: Optional[str] = None
    document_type: Optional[str] = None
    language: Optional[str] = None
    submission_status: Optional[str] = None
    verification_status: Optional[str] = None


class DuplicateCheckRequest(BaseModel):
    survey_number: str = ""
    sub_division: str = ""
    khasra_no: str = ""
    khata_no: str = ""
    owner_name: str = ""
    village: str = ""


class AuditOut(BaseModel):
    id: int
    record_id: int
    record_id_str: str
    action: str
    details: str
    performed_by: str
    timestamp: Optional[datetime] = None

    class Config:
        from_attributes = True


class ChatPayload(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    record_id: int
    record_id_str: Optional[str] = ""
    grounded: bool = True
    model: str = "gemini-1.5-flash"


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/auth/login")
def login_endpoint(payload: LoginRequest):
    u = payload.username.strip().lower()
    if "officer" in u or "tehsildar" in u or "admin" in u:
        return {
            "status": "ok",
            "role": "officer",
            "username": payload.username,
            "display_name": "Tehsildar (Officer) · S. Sharma",
            "role_title": "Tehsildar",
            "department": "Land Revenue Department",
            "permissions": ["upload", "review", "dashboard", "map", "verify", "edit", "export"]
        }
    else:
        return {
            "status": "ok",
            "role": "citizen",
            "username": payload.username,
            "display_name": "Citizen Portal · Public Access",
            "role_title": "Citizen",
            "department": "Public Citizen Services",
            "permissions": ["map", "search", "view_record", "print_certificate"]
        }



# ---- GEOCODING & MAP COORDINATES FOR NEWLY ADDED PLACES ----

LOCATION_COORDINATES = {
    # Rajasthan locations
    "jaipur": (26.9124, 75.7873),
    "जयपुर": (26.9124, 75.7873),
    "दुर्गापुरा": (26.8521, 75.7942),
    "durgapura": (26.8521, 75.7942),
    "बस्सी": (26.8322, 76.0425),
    "bassi": (26.8322, 76.0425),
    "rajasthan": (26.5, 74.8),
    "राजस्थान": (26.5, 74.8),
    # UP locations
    "rampur": (28.8154, 79.0256),
    "रामपुर": (28.8154, 79.0256),
    "meerut": (28.9845, 77.7064),
    "मेरठ": (28.9845, 77.7064),
    "sadar": (28.9800, 77.7000),
    "सदर": (28.9800, 77.7000),
    "bareilly": (28.3670, 79.4304),
    "meerganj": (28.5414, 79.2081),
    "aligarh": (27.8974, 78.0880),
    "etawah": (26.7769, 79.0305),
    "kanpur": (26.4499, 80.3319),
    "uttar pradesh": (26.8467, 80.9462),
    # Telangana / AP locations
    "choutuppal": (17.2514, 78.9038),
    "చౌటుప్పల్": (17.2514, 78.9038),
    "yadadri": (17.5898, 78.9460),
    "యాదాద్రి": (17.5898, 78.9460),
    "telangana": (17.8496, 79.1152),
    "andhra pradesh": (15.9129, 79.7400),
}


def assign_coordinates_if_missing(record: LandRecord):
    """Auto-assigns geographic coordinates for newly added places/districts/states."""
    if record.latitude and record.longitude and (record.latitude != 0 or record.longitude != 0):
        return

    place_keys = [
        (record.village or "").lower(),
        (record.tehsil or record.mandal or "").lower(),
        (record.district or "").lower(),
        (record.state or "").lower()
    ]

    base_lat, base_lng = None, None
    for pk in place_keys:
        if not pk:
            continue
        for loc_k, coords in LOCATION_COORDINATES.items():
            if loc_k in pk:
                base_lat, base_lng = coords
                break
        if base_lat is not None:
            break

    if base_lat is None:
        v_hash = abs(hash(record.village or record.district or record.state or "India")) % 1000
        base_lat = 26.5 + (v_hash % 50) * 0.05
        base_lng = 77.0 + ((v_hash // 50) % 50) * 0.05

    import random
    record.latitude = round(base_lat + random.uniform(-0.008, 0.008), 6)
    record.longitude = round(base_lng + random.uniform(-0.008, 0.008), 6)


# ---- UPLOAD (original, preserved + enhanced) ----

def _process_pdf(pdf_path: str, saved_name: str):
    """
    Process legacy / multi-page land-record PDFs:
    1. Detect text layer: extract digital text if available.
    2. Render every page at 300 DPI as high-resolution PNG image.
    3. Return primary preview image, list of all page images, and extracted text.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            raise ValueError("PDF has no pages")

        page_images = []
        digital_text = ""

        mat = fitz.Matrix(300 / 72, 300 / 72)
        for page_num in range(len(doc)):
            page = doc[page_num]
            txt = page.get_text()
            if txt.strip():
                digital_text += f"\n--- Page {page_num + 1} ---\n" + txt

            pix = page.get_pixmap(matrix=mat)
            img_name = saved_name.replace(".pdf", f"_p{page_num}.png")
            img_path = os.path.join(UPLOAD_DIR, img_name)
            pix.save(img_path)
            page_images.append((img_path, img_name))

        doc.close()
        primary_path, primary_name = page_images[0]
        return primary_path, primary_name, page_images, digital_text
    except ImportError:
        try:
            from PIL import Image
            img = Image.open(pdf_path)
            img_name = saved_name.replace(".pdf", ".png")
            img_path = os.path.join(UPLOAD_DIR, img_name)
            img.save(img_path)
            return img_path, img_name, [(img_path, img_name)], ""
        except Exception:
            raise ValueError(
                "PDF processing requires PyMuPDF (pip install PyMuPDF). "
                "Could not process PDF with available libraries."
            )


def compute_file_hash(path: str) -> str:
    """Computes SHA-256 hash of raw file bytes for file-level duplicate detection."""
    try:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        return sha.hexdigest()
    except Exception:
        return ""


def process_file_pipeline(
    file_path: str,
    orig_filename: str,
    saved_name: str,
    is_pdf: bool,
    language: str,
    document_type_hint: str,
    db: Session,
) -> RecordOut:
    # 1. FILE-LEVEL DUPLICATE DETECTION (SHA-256 of raw uploaded bytes)
    file_hash = compute_file_hash(file_path)
    if file_hash:
        existing = db.query(LandRecord).filter(LandRecord.file_hash == file_hash).first()
        if existing:
            # Skip expensive OCR & extraction pipeline entirely, do not create duplicate DB row
            rec_out = RecordOut.model_validate(existing)
            rec_out.is_file_duplicate = True
            rec_out.duplicate_of_id_str = existing.record_id_str or f"REC{existing.id:03d}"
            rec_out.duplicate_message = f"Exact duplicate of {rec_out.duplicate_of_id_str}: File previously uploaded"
            return rec_out

    page_images = []
    digital_pdf_text = ""
    primary_path = file_path

    if is_pdf:
        try:
            primary_path, saved_name, page_images, digital_pdf_text = _process_pdf(file_path, saved_name)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"PDF processing failed: {e}")

    try:
        clean_img, used_recovery_pass = preprocess_with_fallback(primary_path)

        all_words = []
        images_to_ocr = [clean_img]
        if is_pdf and len(page_images) > 1:
            for p_path, _ in page_images[1:3]:
                try:
                    p_clean, _ = preprocess_with_fallback(p_path)
                    images_to_ocr.append(p_clean)
                except Exception:
                    pass

        for img in images_to_ocr:
            page_words = run_hybrid_ocr(img, use_handwriting=ENABLE_HANDWRITING_OCR)
            all_words.extend(page_words)

        full_ocr_text = get_full_ocr_text(all_words)
        if digital_pdf_text:
            full_ocr_text = digital_pdf_text + "\n" + full_ocr_text

        detected_lang = language
        if language == "auto" or not language:
            detected_lang, _ = detect_language_from_text(full_ocr_text)

        fields = extract_fields(all_words)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Pipeline failed: {e}")

    field_values = {k: fields[k].value for k in SCHEMA_FIELDS if k in fields}
    field_conf_str = ",".join(f"{k}:{fields[k].confidence}" for k in SCHEMA_FIELDS if k in fields)
    overall = overall_confidence(fields)

    hw_analysis = analyze_document_handwriting(clean_img)
    is_handwritten = (document_type_hint.lower() == "handwritten") or hw_analysis.is_handwritten

    if is_handwritten and not ENABLE_HANDWRITING_OCR:
        overall = min(overall, 74.0) if overall else 74.0
        for k in field_values:
            if fields.get(k) and fields[k].value:
                fields[k].confidence = min(fields[k].confidence, 74.0)

    warnings = cross_validate(db, field_values)

    dup_matches = check_duplicate(db, field_values)
    dup_status = "possible_duplicate" if dup_matches else "none"
    if dup_matches:
        warnings.append(f"Possible duplicate of {dup_matches[0]['record_id_str']}: {dup_matches[0]['match_reason']}")

    max_rec = db.query(LandRecord).order_by(LandRecord.id.desc()).first()
    next_num = (max_rec.id + 1) if max_rec else 1
    record_id_str = f"REC{next_num:03d}"

    if is_pdf and len(page_images) > 1:
        doc_source = "Legacy Multi-Page PDF"
        final_doc_type = "Legacy Multi-Page PDF"
    elif is_pdf:
        doc_source = "Legacy PDF"
        final_doc_type = "Digital PDF Upload"
    elif document_type_hint != "auto":
        final_doc_type = document_type_hint
        doc_source = "Manual Document Upload"
    elif hw_analysis.classification == "mixed_print_handwritten":
        final_doc_type = "Mixed Print+Handwriting (FIR Style)"
        doc_source = "Mixed Government Register"
    elif hw_analysis.classification == "handwritten":
        final_doc_type = "Handwritten Record (IIIT-HW / CRNN)"
        doc_source = "Handwritten Field Survey"
    else:
        final_doc_type = "Scanned Printed Document"
        doc_source = "Official Revenue Scan"

    record = LandRecord(
        record_id_str=record_id_str,
        filename=saved_name,
        field_confidence=field_conf_str,
        overall_confidence=overall,
        ocr_confidence=overall,
        status="pending_review" if overall < CONFIDENCE_LOW_THRESHOLD or warnings or is_handwritten else "verified",
        validation_warnings="; ".join(warnings),
        submission_status="submitted",
        verification_status="under_review" if overall < CONFIDENCE_LOW_THRESHOLD or warnings or is_handwritten else "verified",
        duplicate_status=dup_status,
        language=detected_lang,
        document_type=final_doc_type,
        document_source=doc_source,
        processing_stage="officer_review" if overall < CONFIDENCE_LOW_THRESHOLD or warnings or is_handwritten else "verified",
        ocr_text=full_ocr_text[:3000] if full_ocr_text else "",
        source="user_upload",
        file_hash=file_hash,
        **field_values,
    )

    # Assign coordinates for new state, district, or place
    assign_coordinates_if_missing(record)

    db.add(record)
    db.commit()
    db.refresh(record)

    # Audit trail
    add_audit_entry(db, record.id, record_id_str, "document_uploaded",
                    f"File: {orig_filename}, Type: {final_doc_type}, Language: {detected_lang}")
    add_audit_entry(db, record.id, record_id_str, "ocr_completed",
                    f"OCR completed ({'Handwriting Mode' if is_handwritten else 'Printed'}). Confidence: {overall}%. Language: {detected_lang}")
    add_audit_entry(db, record.id, record_id_str, "fields_extracted",
                    f"Extracted {len([f for f in fields.values() if f.value])} fields from {orig_filename}")
    add_audit_entry(db, record.id, record_id_str, "validation_completed",
                    f"Warnings: {len(warnings)}, Duplicate: {dup_status}")

    rec_out = RecordOut.model_validate(record)
    rec_out.is_file_duplicate = False
    return rec_out


@app.post("/upload", response_model=RecordOut)
async def upload_record(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    document_type_hint: str = Form("auto"),
    db: Session = Depends(get_db),
):
    """
    Stages 1, 3, 4, 5, 7: run the full pipeline on an uploaded scan / legacy PDF.
    Maintains exact contract with frontend single-upload.
    """
    ext = os.path.splitext(file.filename or "")[1].lower() or ".jpg"
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    saved_name = f"{uuid.uuid4().hex}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return process_file_pipeline(
        file_path=saved_path,
        orig_filename=file.filename or saved_name,
        saved_name=saved_name,
        is_pdf=(ext == ".pdf"),
        language=language,
        document_type_hint=document_type_hint,
        db=db,
    )


@app.post("/upload/batch", response_model=List[RecordOut])
async def upload_batch(
    files: List[UploadFile] = File(...),
    language: str = Form("auto"),
    document_type_hint: str = Form("auto"),
    db: Session = Depends(get_db),
):
    """
    Stage 1-7 Batch Processing: Accepts multiple files (images and PDFs) at once.
    For PDFs, each page is extracted as a separate 300 DPI image and processed
    through the pipeline as an individual record.
    Returns the list of all created records.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for batch upload")

    created_records = []
    for file in files:
        ext = os.path.splitext(file.filename or "")[1].lower() or ".jpg"
        if ext not in ALLOWED_EXTENSIONS:
            continue

        raw_bytes = await file.read()
        saved_name = f"{uuid.uuid4().hex}{ext}"
        saved_path = os.path.join(UPLOAD_DIR, saved_name)
        with open(saved_path, "wb") as f:
            f.write(raw_bytes)

        if ext == ".pdf":
            try:
                import fitz
                doc = fitz.open(saved_path)
                mat = fitz.Matrix(300 / 72, 300 / 72)
                for page_idx in range(len(doc)):
                    page = doc[page_idx]
                    pix = page.get_pixmap(matrix=mat)
                    page_img_name = f"{saved_name}_p{page_idx + 1}.png"
                    page_img_path = os.path.join(UPLOAD_DIR, page_img_name)
                    pix.save(page_img_path)

                    rec = process_file_pipeline(
                        file_path=page_img_path,
                        orig_filename=f"{file.filename or 'doc'} (Page {page_idx + 1})",
                        saved_name=page_img_name,
                        is_pdf=False,
                        language=language,
                        document_type_hint=document_type_hint,
                        db=db,
                    )
                    created_records.append(rec)
                doc.close()
            except Exception:
                rec = process_file_pipeline(
                    file_path=saved_path,
                    orig_filename=file.filename or saved_name,
                    saved_name=saved_name,
                    is_pdf=True,
                    language=language,
                    document_type_hint=document_type_hint,
                    db=db,
                )
                created_records.append(rec)
        else:
            rec = process_file_pipeline(
                file_path=saved_path,
                orig_filename=file.filename or saved_name,
                saved_name=saved_name,
                is_pdf=False,
                language=language,
                document_type_hint=document_type_hint,
                db=db,
            )
            created_records.append(rec)

    return created_records


# ---- CADASTRAL / VILLAGE MAP EXTRACTION ----

@app.post("/upload/map")
async def upload_map_endpoint(
    file: UploadFile = File(...),
    village: str = Form("Rampur"),
    db: Session = Depends(get_db),
):
    """
    Process scanned village maps, cadastral parcel maps, and survey maps.
    Extracts parcel contours, survey numbers, village labels, and estimated area.
    """
    ext = os.path.splitext(file.filename or "")[1].lower() or ".jpg"
    saved_name = f"map_{uuid.uuid4().hex}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if ext == ".pdf":
        try:
            saved_path, saved_name, _, _ = _process_pdf(saved_path, saved_name)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"PDF conversion failed: {e}")

    result = process_land_map(saved_path, village_hint=village)
    result["image_filename"] = saved_name
    result["image_url"] = f"/uploads/{saved_name}"

    # Log audit entry
    add_audit_entry(db, 0, "MAP-UPLOAD", "cadastral_map_processed",
                    f"Processed map {file.filename}: {result['parcel_count']} parcels detected for {result['village']}")

    return result


# ---- RECORDS CRUD ----

@app.get("/records", response_model=List[RecordOut])
def list_records(
    status: Optional[str] = None,
    submission_status: Optional[str] = None,
    language: Optional[str] = None,
    village: Optional[str] = None,
    document_type: Optional[str] = None,
    source: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(default=500, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    q = db.query(LandRecord)
    if source:
        q = q.filter(LandRecord.source == source)
    if status:
        q = q.filter(LandRecord.status == status)
    if submission_status:
        q = q.filter(LandRecord.submission_status == submission_status)
    if language:
        q = q.filter(LandRecord.language == language.lower())
    if village:
        q = q.filter(LandRecord.village == village)
    if document_type:
        q = q.filter(LandRecord.document_type == document_type)
    if search:
        search_term = f"%{search}%"
        q = q.filter(
            or_(
                LandRecord.owner_name.ilike(search_term),
                LandRecord.survey_number.ilike(search_term),
                LandRecord.khasra_no.ilike(search_term),
                LandRecord.khata_no.ilike(search_term),
                LandRecord.record_id_str.ilike(search_term),
                LandRecord.village.ilike(search_term),
                LandRecord.father_name.ilike(search_term),
            )
        )
    return q.order_by(LandRecord.id.desc()).offset(offset).limit(limit).all()
 
 
# ---- FILTER VALUES ----
 
@app.get("/records/filters")
def get_filter_values(db: Session = Depends(get_db)):
    """Return distinct values for filter dropdowns."""
    villages = [r[0] for r in db.query(LandRecord.village).distinct().all() if r[0]]
    districts = [r[0] for r in db.query(LandRecord.district).distinct().all() if r[0]]
    states = [r[0] for r in db.query(LandRecord.state).distinct().all() if r[0]]
    languages = [r[0] for r in db.query(LandRecord.language).distinct().all() if r[0]]
    doc_types = [r[0] for r in db.query(LandRecord.document_type).distinct().all() if r[0]]
    land_types = [r[0] for r in db.query(LandRecord.land_type).distinct().all() if r[0]]
    return {
        "villages": sorted(villages),
        "districts": sorted(districts),
        "states": sorted(states),
        "languages": sorted(languages),
        "document_types": sorted(doc_types),
        "land_types": sorted(land_types),
    }
 
 
# ---- SEARCH ----
 
@app.get("/records/search/global")
def search_records(q: str = Query(..., min_length=1), db: Session = Depends(get_db)):
    """
    Global search across owner name, father name, survey number, khasra no,
    khata no, record ID, village, tehsil, and district.
    Supports English, Hindi (Devanagari), and Telugu unicode matching.
    """
    search_term = f"%{q}%"
    results = db.query(LandRecord).filter(
        or_(
            LandRecord.owner_name.ilike(search_term),
            LandRecord.father_name.ilike(search_term),
            LandRecord.survey_number.ilike(search_term),
            LandRecord.khasra_no.ilike(search_term),
            LandRecord.khata_no.ilike(search_term),
            LandRecord.record_id_str.ilike(search_term),
            LandRecord.village.ilike(search_term),
            LandRecord.tehsil.ilike(search_term),
            LandRecord.mandal.ilike(search_term),
            LandRecord.district.ilike(search_term),
        )
    ).limit(50).all()
    return [
        {
            "id": r.id,
            "record_id_str": r.record_id_str,
            "owner_name": r.owner_name,
            "father_name": r.father_name or "",
            "survey_number": r.survey_number or r.khasra_no,
            "khasra_no": r.khasra_no or "",
            "khata_no": r.khata_no or "",
            "area_hectare": r.area_hectare or "",
            "village": r.village,
            "language": r.language or "english",
            "document_type": r.document_type or "Scanned Document",
            "submission_status": r.submission_status,
            "verification_status": r.verification_status,
            "ocr_confidence": r.ocr_confidence or r.overall_confidence,
        }
        for r in results
    ]


# ---- HANDWRITING & DATASET BENCHMARKS ----

@app.get("/models/handwriting/datasets")
def get_dataset_benchmarks():
    """
    Returns benchmark dataset profiles:
    - IIIT-HW-Dev & IIIT-HW-Words (Devanagari CRNN)
    - PHDIndic_11 (Multi-script Indic handwriting)
    - FIR Dataset (Legal mixed print+handwriting, arXiv:2306.02142)
    - FUNSD / XFUND (Multilingual form layout)
    - AI4Bharat Naamapadam (Indic NER)
    """
    return get_all_dataset_benchmarks()
 
 
@app.get("/records/{record_id}", response_model=RecordOut)
def get_record(record_id: int, db: Session = Depends(get_db)):
    record = db.query(LandRecord).get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return record


@app.post("/records/{record_id}/chat", response_model=ChatResponse)
async def chat_with_record(record_id: int, payload: ChatPayload, db: Session = Depends(get_db)):
    """
    Stage 2: Gemini Chatbot for Record Document Q&A.
    Answers natural language queries strictly grounded in the record's extracted fields,
    validation warnings, confidence scores, and raw OCR text.
    """
    record = db.query(LandRecord).get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    chat_result = await answer_record_question(record, payload.question)
    return ChatResponse(
        answer=chat_result.get("answer", "No response generated."),
        record_id=record.id,
        record_id_str=record.record_id_str or f"REC{record.id:03d}",
        grounded=chat_result.get("grounded", True),
        model=chat_result.get("model", "gemini-1.5-flash")
    )


@app.post("/records", response_model=RecordOut)
def create_record(payload: RecordCreate, db: Session = Depends(get_db)):
    """Create a new record with duplicate check enforcement."""
    # Duplicate check
    fields_dict = payload.model_dump()
    dup_matches = check_duplicate(db, fields_dict)
    if dup_matches:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Possible duplicate record detected",
                "matches": dup_matches,
            }
        )

    # Generate record ID
    max_rec = db.query(LandRecord).order_by(LandRecord.id.desc()).first()
    next_num = (max_rec.id + 1) if max_rec else 1
    record_id_str = f"REC{next_num:03d}"

    record = LandRecord(
        record_id_str=record_id_str,
        status="pending_review",
        verification_status="pending",
        **fields_dict,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    add_audit_entry(db, record.id, record_id_str, "record_created",
                    f"Record created manually for {payload.owner_name}")

    return record


@app.put("/records/{record_id}", response_model=RecordOut)
def update_record(record_id: int, payload: RecordUpdate, db: Session = Depends(get_db)):
    record = db.query(LandRecord).get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    changes = []
    for field_name, value in payload.model_dump(exclude_none=True).items():
        old_value = getattr(record, field_name, None)
        if old_value != value:
            setattr(record, field_name, value)
            changes.append(f"{field_name}: '{old_value}' → '{value}'")

    record.last_updated = datetime.utcnow()
    db.commit()
    db.refresh(record)

    if changes:
        add_audit_entry(db, record.id, record.record_id_str or "", "record_updated",
                        "; ".join(changes), "officer_demo")

    return record


# ---- DUPLICATE CHECK ----

@app.post("/records/check-duplicate")
def check_duplicate_endpoint(payload: DuplicateCheckRequest, db: Session = Depends(get_db)):
    fields_dict = payload.model_dump()
    matches = check_duplicate(db, fields_dict)
    return {
        "has_duplicate": len(matches) > 0,
        "matches": matches,
    }


# ---- OFFICER ACTIONS ----

@app.patch("/records/{record_id}/review", response_model=RecordOut)
def review_record(record_id: int, payload: ReviewPayload, db: Session = Depends(get_db)):
    """Stage 6: officer review. Applies corrections to low-confidence
    fields and marks the record verified."""
    record = db.query(LandRecord).get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    changes = []
    for field_name, value in payload.corrections.items():
        if hasattr(record, field_name):
            old_val = getattr(record, field_name)
            if old_val != value:
                setattr(record, field_name, value)
                changes.append(f"{field_name}: '{old_val}' → '{value}'")

    record.status = "verified"
    record.verification_status = "verified"
    record.reviewed_by = payload.reviewed_by
    record.reviewed_at = datetime.utcnow()
    record.last_updated = datetime.utcnow()
    record.processing_stage = "verified"
    db.commit()
    db.refresh(record)

    if changes:
        add_audit_entry(db, record.id, record.record_id_str or "", "officer_edited",
                        "; ".join(changes), payload.reviewed_by)
    add_audit_entry(db, record.id, record.record_id_str or "", "record_approved",
                    "Record verified and approved", payload.reviewed_by)

    return record


@app.post("/records/{record_id}/approve", response_model=RecordOut)
def approve_record(record_id: int, db: Session = Depends(get_db)):
    record = db.query(LandRecord).get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    record.status = "verified"
    record.verification_status = "verified"
    record.reviewed_at = datetime.utcnow()
    record.reviewed_by = "officer_demo"
    record.last_updated = datetime.utcnow()
    record.processing_stage = "verified"
    db.commit()
    db.refresh(record)

    add_audit_entry(db, record.id, record.record_id_str or "", "record_approved",
                    "Record approved by officer", "officer_demo")
    return record


@app.post("/records/{record_id}/reject", response_model=RecordOut)
def reject_record(record_id: int, db: Session = Depends(get_db)):
    record = db.query(LandRecord).get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    record.verification_status = "rejected"
    record.status = "pending_review"
    record.last_updated = datetime.utcnow()
    record.processing_stage = "rejected"
    db.commit()
    db.refresh(record)

    add_audit_entry(db, record.id, record.record_id_str or "", "record_rejected",
                    "Record rejected by officer", "officer_demo")
    return record


@app.post("/records/{record_id}/publish", response_model=RecordOut)
def publish_record(record_id: int, db: Session = Depends(get_db)):
    record = db.query(LandRecord).get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    record.status = "verified"
    record.verification_status = "published"
    record.submission_status = "submitted"
    record.last_updated = datetime.utcnow()
    record.processing_stage = "published"
    db.commit()
    db.refresh(record)

    add_audit_entry(db, record.id, record.record_id_str or "", "record_published",
                    "Record published to digital ledger", "officer_demo")
    return record


@app.post("/records/{record_id}/confirm-duplicate")
def confirm_duplicate(record_id: int, db: Session = Depends(get_db)):
    record = db.query(LandRecord).get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    record.duplicate_status = "confirmed_duplicate"
    record.last_updated = datetime.utcnow()
    db.commit()
    add_audit_entry(db, record.id, record.record_id_str or "", "duplicate_confirmed",
                    "Record confirmed as duplicate", "officer_demo")
    return {"status": "ok", "duplicate_status": "confirmed_duplicate"}


@app.post("/records/{record_id}/clear-duplicate")
def clear_duplicate_endpoint(record_id: int, db: Session = Depends(get_db)):
    record = db.query(LandRecord).get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    record.duplicate_status = "cleared"
    record.last_updated = datetime.utcnow()
    db.commit()
    add_audit_entry(db, record.id, record.record_id_str or "", "duplicate_cleared",
                    "Duplicate flag cleared by officer", "officer_demo")
    return {"status": "ok", "duplicate_status": "cleared"}


# ---- DASHBOARD STATISTICS ----

@app.get("/dashboard/statistics")
def dashboard_statistics(source: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(LandRecord)
    if source:
        q = q.filter(LandRecord.source == source)

    total = q.count()

    submitted = q.filter(LandRecord.submission_status == "submitted").count()
    pending = q.filter(LandRecord.submission_status == "pending").count()
    not_submitted = q.filter(LandRecord.submission_status == "not_submitted").count()

    verified = q.filter(LandRecord.verification_status.in_(["verified", "published"])).count()
    under_review = q.filter(LandRecord.verification_status == "under_review").count()
    published = q.filter(LandRecord.verification_status == "published").count()

    duplicates = q.filter(
        or_(
            LandRecord.duplicate_status.in_(["possible_duplicate", "confirmed_duplicate", "duplicate"]),
            LandRecord.duplicate_status.like("%duplicate%"),
            LandRecord.validation_warnings.like("%duplicate%")
        )
    ).count()

    avg_confidence = q.filter(LandRecord.ocr_confidence > 0).with_entities(func.avg(LandRecord.ocr_confidence)).scalar() or 0.0

    high_confidence = q.filter(LandRecord.ocr_confidence >= 85).count()
    med_confidence = q.filter(LandRecord.ocr_confidence >= 65, LandRecord.ocr_confidence < 85).count()
    low_confidence = q.filter(LandRecord.ocr_confidence > 0, LandRecord.ocr_confidence < 65).count()

    # Language distribution
    lang_dist = {}
    for lang, cnt in q.with_entities(LandRecord.language, func.count(LandRecord.id)).group_by(LandRecord.language).all():
        lang_dist[lang or "unknown"] = cnt

    # Document type distribution
    doc_dist = {}
    for dtype, cnt in q.with_entities(LandRecord.document_type, func.count(LandRecord.id)).group_by(LandRecord.document_type).all():
        doc_dist[dtype or "unknown"] = cnt

    # Land type distribution
    land_dist = {}
    for ltype, cnt in q.with_entities(LandRecord.land_type, func.count(LandRecord.id)).group_by(LandRecord.land_type).all():
        land_dist[ltype or "unknown"] = cnt

    submission_rate = round(submitted / total * 100, 1) if total else 0
    verification_rate = round(verified / total * 100, 1) if total else 0
    duplicate_rate = round(duplicates / total * 100, 1) if total else 0

    # Verification status distribution
    verif_dist = {}
    for vs, cnt in q.with_entities(LandRecord.verification_status, func.count(LandRecord.id)).group_by(LandRecord.verification_status).all():
        verif_dist[vs or "unknown"] = cnt

    # Processing stage distribution
    stage_dist = {}
    for ps, cnt in q.with_entities(LandRecord.processing_stage, func.count(LandRecord.id)).group_by(LandRecord.processing_stage).all():
        stage_dist[ps or "unknown"] = cnt

    return {
        "total": total,
        "submitted": submitted,
        "pending": pending,
        "not_submitted": not_submitted,
        "verified": verified,
        "under_review": under_review,
        "published": published,
        "duplicates": duplicates,
        "avg_confidence": round(avg_confidence, 1),
        "high_confidence": high_confidence,
        "med_confidence": med_confidence,
        "low_confidence": low_confidence,
        "confidence_distribution": {
            "high": high_confidence,
            "medium": med_confidence,
            "low": low_confidence,
        },
        "submission_rate": submission_rate,
        "verification_rate": verification_rate,
        "duplicate_rate": duplicate_rate,
        "language_distribution": lang_dist,
        "document_type_distribution": doc_dist,
        "land_type_distribution": land_dist,
        "verification_status_distribution": verif_dist,
        "processing_stage_distribution": stage_dist,
    }


# ---- MAP RECORDS ----

@app.get("/map/records")
def map_records(
    submission_status: Optional[str] = None,
    language: Optional[str] = None,
    village: Optional[str] = None,
    district: Optional[str] = None,
    state: Optional[str] = None,
    document_type: Optional[str] = None,
    db: Session = Depends(get_db),
):
    q = db.query(LandRecord).filter(LandRecord.latitude != 0, LandRecord.longitude != 0)
    if submission_status:
        q = q.filter(LandRecord.submission_status == submission_status)
    if language:
        q = q.filter(LandRecord.language == language.lower())
    if village:
        q = q.filter(LandRecord.village == village)
    if district:
        q = q.filter(LandRecord.district == district)
    if state:
        q = q.filter(LandRecord.state == state)
    if document_type:
        q = q.filter(LandRecord.document_type == document_type)

    records = q.all()
    return [
        {
            "record_id": r.id,
            "record_id_str": r.record_id_str or f"REC{r.id:03d}",
            "owner_name": r.owner_name or "Unknown",
            "father_name": r.father_name or "",
            "survey_number": r.survey_number or r.khasra_no or "",
            "khata_no": r.khata_no or "",
            "village": r.village or "",
            "mandal": r.mandal or r.tehsil or "",
            "district": r.district or "",
            "state": r.state or "",
            "area_hectare": r.area_hectare or "",
            "land_type": r.land_type or "",
            "submission_status": r.submission_status or "not_submitted",
            "verification_status": r.verification_status or "pending",
            "duplicate_status": r.duplicate_status or "none",
            "ocr_confidence": r.ocr_confidence or r.overall_confidence or 0,
            "language": r.language or "english",
            "document_type": r.document_type or "",
            "processing_stage": r.processing_stage or "",
            "latitude": r.latitude,
            "longitude": r.longitude,
            "last_updated": r.last_updated.isoformat() if r.last_updated else None,
        }
        for r in records
    ]


# ---- AUDIT TRAIL ----

@app.get("/audit/{record_id}", response_model=List[AuditOut])
def get_audit_trail(record_id: int, db: Session = Depends(get_db)):
    entries = db.query(AuditLog).filter(
        AuditLog.record_id == record_id
    ).order_by(AuditLog.timestamp.asc()).all()
    return entries


# ---- DEMO DATA MANAGEMENT ----

@app.post("/demo/load")
def load_demo_data_endpoint(db: Session = Depends(get_db)):
    from seed_data import load_demo_data
    count = load_demo_data(db)
    return {"status": "ok", "records_loaded": count, "message": f"Loaded {count} demo records for Rampur Village"}


@app.post("/demo/reset")
def reset_demo_data(db: Session = Depends(get_db)):
    db.query(AuditLog).delete()
    db.query(LandRecord).delete()
    db.commit()
    return {"status": "ok", "message": "All demo data cleared"}


@app.post("/demo/generate-sample")
def generate_sample_endpoint(db: Session = Depends(get_db)):
    from seed_data import generate_single_sample
    result = generate_single_sample(db)
    return {"status": "ok", "record": result}


# ---- VALIDATION ----

@app.post("/records/{record_id}/validate")
def validate_record_endpoint(record_id: int, db: Session = Depends(get_db)):
    record = db.query(LandRecord).get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    fields = {
        "owner_name": record.owner_name,
        "survey_number": record.survey_number or record.khasra_no,
        "khasra_no": record.khasra_no,
        "khata_no": record.khata_no,
        "area_hectare": record.area_hectare,
        "village": record.village,
        "district": record.district,
        "state": record.state,
    }
    result = validate_record_structured(fields, db=db, exclude_id=record_id)
    record.validation_results = json.dumps(result)
    record.last_updated = datetime.utcnow()
    db.commit()
    return result


# ---- STATIC FILES ----

@app.get("/uploads/{filename}")
def get_uploaded_image(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(FRONTEND_DIR):
    from fastapi.staticfiles import StaticFiles
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Intelligent Land Record Digitization API",
        "ui_url": "http://localhost:8000/ui/",
        "docs_url": "http://localhost:8000/docs"
    }
