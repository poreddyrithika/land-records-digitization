# Intelligent Land Record Digitization & Validation System
**SIH26018 · Team Spirit — Working Prototype**

An AI-based platform that digitizes and validates legacy land records —
scanned registers, handwritten documents, and legacy PDFs — automatically
extracting structured fields (owner, khasra/survey no., khata no., area,
village, tehsil, district, classification, mutation no.), flagging
low-confidence and duplicate entries for officer review, and exposing the
verified data through both an officer dashboard and a citizen-facing
public portal.

This is a **working proof of concept**, not the production DILRMP system.
Every place a real government integration was substituted for a
hackathon-tractable equivalent is called out explicitly below — answer
honestly with this table if a judge asks "is this the real thing."

## Problem statement → what we built

| Expected solution (per SIH26018) | Status in this PoC |
|---|---|
| Multilingual document recognition | ✅ English, Hindi, Telugu — printed (Tesseract) + handwritten (TrOCR, Devanagari fine-tune) |
| Automatic extraction from scans, images, legacy PDFs | ✅ Multi-page PDF splitting (PyMuPDF) + image upload, both routed through the same OCR/extraction pipeline |
| Classification into predefined land-record fields | ✅ Regex + fuzzy label matching in `extract.py`, restricted to fields that genuinely appear in the source documents |
| Automated validation, cross-database checks, duplicate detection | ✅ Rule-based cross-validation + exact/near duplicate detection (`validate.py`, `duplicate_detector.py`) with officer confirm/clear actions |
| Confidence scoring, uncertain-field identification | ✅ Per-field confidence + overall confidence threshold routes low-confidence records to officer review automatically |
| Human-assisted verification workflow | ✅ Officer Review Queue — prioritized by duplicate > low confidence > validation conflict > missing fields > pending |
| AI-driven learning that improves over time | Officer corrections are saved to the DB but nothing retrains on them yet |
| Integration with LRMS/DILRMP/GIS/cadastral maps | Cadastral map upload + parcel/survey-number extraction is implemented (`map_extractor.py`); no live Bhoomi/DILRMP/GIS integration exists (no public sandbox to connect to) |
| Secure repository with metadata + audit trails | ✅ Full audit log per record (`AuditLog` table), record lifecycle stages tracked end-to-end |
| Interactive dashboards (processed count, accuracy, validation status, pending cases, state/district progress) | ✅ Records Dashboard (stats + searchable table) and a Leaflet Map View (village-level pins, verified/pending status, filters by state/district/village/language/doc type) |
| APIs for government-system integration | ✅ REST API (FastAPI) — every UI action is a documented endpoint, ready to be called by another system |
| Role-based access control | ✅ Demo-level: Officer (Tehsildar) vs Citizen login, with the citizen view restricted to a read-only map + official Record-of-Rights extract |

## What actually works right now, end to end

1. Upload a scanned image or multi-page PDF → pre-processing → OCR
   (Tesseract for printed text, TrOCR-Devanagari for handwriting,
   confidence-gated hybrid routing) → field extraction → cross-validation
   → duplicate check.
2. Records that are clean and unambiguous go straight to **Verified**.
   Anything with low OCR confidence, a validation warning, or a possible
   duplicate is routed to the **Officer Review Queue**, where a Tehsildar
   can correct fields, confirm/dismiss duplicates, and approve or reject.
3. Verified records appear in the **Records Dashboard** (officer) and on
   the **Map View**, which is also what a **Citizen** login sees — a
   read-only, searchable public portal showing an official-style Record
   of Rights extract per parcel, without exposing OCR confidence internals.
4. A Gemini-backed chat assistant answers questions about any individual
   record, strictly grounded in that record's extracted fields and OCR
   text (no open-ended hallucination).

Tested end-to-end on synthetic khatauni-style samples with realistic scan
degradation and mixed English/Hindi/Telugu handwritten-style text — see
`EVALUATION.md` for the real accuracy numbers to quote in the demo
(headline: 98.4% field accuracy on clean scans, dropping on degraded
scans as expected, with a confidence-triggered recovery pass that more
than tripled heavy-degradation accuracy).

## Project structure

land-records-poc/
├── backend/
│ ├── main.py # FastAPI app — all endpoints, 7-stage pipeline orchestration
│ ├── preprocess.py # Stage 1: deskew, denoise, binarize (+ recovery pass)
│ ├── document_input.py # Legacy PDF → page images (PyMuPDF)
│ ├── ocr_engine.py # Stage 3: Tesseract (eng+hin+tel)
│ ├── handwriting_ocr.py # Stage 3: TrOCR, Devanagari fine-tune (opt-in)
│ ├── hybrid_ocr.py # Confidence-gated routing between the two OCR engines
│ ├── doc_classifier.py # Printed vs handwritten / doc-type detection
│ ├── extract.py # Stage 4: field mapping + confidence scoring
│ ├── validate.py # Stage 5: cross-validation against existing records
│ ├── duplicate_detector.py # Exact + fuzzy duplicate detection
│ ├── map_extractor.py # Cadastral/village map parcel extraction
│ ├── gemini_chat.py # Grounded per-record chat assistant
│ ├── models.py # Stage 7: SQLite schema + audit log
│ └── requirements.txt
├── frontend/
│ └── index.html # Officer + Citizen UI: login, upload, review queue, dashboard, map
├── scripts/
│ ├── gen_data.py # Synthetic multilingual/handwritten-style sample generator
│ └── evaluate.py # Batch accuracy evaluation
└── data/sample_records/ # Generated sample scans + ground_truth.json


## How to run it

**1. Install system dependencies (Tesseract OCR + languages, poppler for PDFs):**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-hin tesseract-ocr-tel poppler-utils
```

**2. Install Python dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

**3. Start the backend:**
```bash
python3 -m uvicorn main:app --reload --port 8000
```

**4. Open the frontend:**
The backend also serves the frontend itself at `http://localhost:8000/ui`
— or open `frontend/index.html` directly in a browser. It talks to
whatever origin it's loaded from (`window.location.origin`), so the same
build works locally, over an ngrok/Cloudflare tunnel, or deployed.

**5. Try it:**
Sign in as **Tehsildar** to upload, review, and manage records, or as
**Citizen** to search and view the public map + Record-of-Rights extracts.
Upload any sample from `data/sample_records/` through "Upload & Process" —
clean scans auto-verify, degraded ones land in the Review Queue with
red/orange fields to correct.

## Enabling handwriting OCR (TrOCR, Devanagari fine-tune)

Off by default (`ENABLE_HANDWRITING_OCR = False` in `main.py`) — needs
extra dependencies and a ~1.3GB model download.
```bash
pip install torch "transformers==4.46.3" sentencepiece
```
Set `ENABLE_HANDWRITING_OCR = True` in `backend/main.py` and restart.
`transformers` must stay pinned at `4.46.3` — newer 5.x releases fail to
load TrOCR's tokenizer (a known compatibility break, not a config issue).
`hybrid_ocr.py` runs Tesseract first and only re-checks low-confidence
words through TrOCR, keeping whichever engine scored higher.

## What's real vs. simulated — be upfront about this in Q&A

| Component | This PoC | Production plan |
|---|---|---|
| Printed-text OCR | Tesseract (`eng+hin+tel`), two-pass recovery on low confidence | Tesseract/PaddleOCR, same idea |
| Handwriting OCR | TrOCR fine-tuned for Devanagari, confidence-triggered, opt-in | CRNN/TrOCR fine-tuned further on land-record-specific handwriting |
| Field mapping | Regex + fuzzy label matching | IndicBERT/IndicNER |
| Layout/doc-type detection | Single hardcoded layout + handwritten-vs-printed classifier | LayoutLMv3/Donut |
| Duplicate detection | Exact hash match + khasra/khata/village rule match | ML-based fuzzy entity matching |
| Cadastral map processing | Parcel/survey-number extraction from map images | Full GIS vectorization (PostGIS + Bhu-Naksha overlay) |
| Bhoomi/DILRMP integration | Not connected — no public sandbox available | Real API integration once government access is granted |
| Active-learning retraining | Not implemented | Retrain OCR/NER on accumulated officer corrections |
| Auth / role access | Demo login (officer/citizen profiles) | Government SSO + granular RBAC |
| Deployment | Local dev server (tunnel-able) | Docker/Kubernetes hybrid-cloud |

## Limitations
- Telugu OCR has a known line-grouping issue: font-metric differences push
  label/value pairs outside `extract.py`'s fixed grouping tolerance more
  often than with Hindi — documented as a specific fixable bug, not just
  "Telugu is harder."
- Only one document layout is supported; a second layout (e.g. a sale deed
  format) would need its own field schema in `extract.py`.
- Cross-validation only checks khasra_no + khata_no + village against
  existing records — no fuzzy/partial matching yet.
- Map coordinates are illustrative village-center points, not surveyed
  parcel boundaries.
