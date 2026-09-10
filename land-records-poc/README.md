# Intelligent Land Record Digitization & Validation System
**SIH26018 · Team Spirit — Working Prototype**

This is a working, testable implementation of the pipeline described in your
SIH proposal. It's a **proof of concept, not the production system** — every
place a real component was substituted for a hackathon-tractable one is
called out below so you can answer honestly in Q&A.

## What actually works right now

Upload a scanned land record image → it flows through all 7 pipeline stages
→ low-confidence fields get flagged → an officer corrects them in a review
UI → the verified record lands in a searchable dashboard.

Tested end-to-end on 20 synthetic khatauni-style images with realistic scan
degradation (skew, noise, blur, fading) — see `EVALUATION.md` for real
accuracy numbers you can quote in your demo.

## Project structure
```
land-records-poc/
├── backend/
│   ├── main.py          # FastAPI app — all endpoints
│   ├── preprocess.py    # Stage 1: deskew, denoise, binarize
│   ├── ocr_engine.py    # Stage 3: Tesseract OCR wrapper
│   ├── extract.py       # Stage 4: field mapping + confidence scoring
│   ├── validate.py      # Stage 5: cross-validation against existing records
│   ├── models.py        # Stage 7: SQLite schema
│   └── requirements.txt
├── frontend/
│   └── index.html       # Single-page UI: upload / review queue / dashboard
├── scripts/
│   ├── gen_data.py       # Synthetic data generator (see below)
│   └── evaluate.py       # Batch accuracy evaluation
└── data/sample_records/  # 20 generated sample scans + ground_truth.json
```

## How to run it

**1. Install system dependency (Tesseract OCR):**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-hin
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
Just open `frontend/index.html` directly in a browser (double-click it, or
`open frontend/index.html`). It talks to `http://localhost:8000` — update
the `API` constant at the top of the `<script>` tag if you deploy the
backend elsewhere.

**5. Try it:**
Upload any image from `data/sample_records/` through the "Upload & Process"
tab. Clean ones auto-verify; degraded ones land in "Officer Review Queue"
with red/orange fields you can correct before publishing.

## Adding handwriting support (TrOCR, fine-tuned for Devanagari)

The problem statement explicitly requires recognizing handwritten text.
Tesseract (used by default) handles printed text well but is genuinely
poor at handwriting — that's not a config issue, it's what Tesseract's
underlying model was and wasn't trained on. Handwriting support is built
but **off by default** (`ENABLE_HANDWRITING_OCR = False` in `main.py`)
because it needs extra dependencies and a large model download.

**Which model, and why:** `handwriting_ocr.py` uses
`paudelanil/trocr-devanagari-2` — a community fine-tune of Microsoft's
TrOCR on Devanagari handwriting data (the IIIT-INDIC-HW-WORDS dataset from
IIIT Hyderabad's CVIT lab — see the dataset discussion in project notes).
An earlier version used the base English TrOCR model and it hallucinated
fluent English words on non-English input — not just wrong guesses, but
structurally incapable of producing Devanagari characters at all, since
it was never trained on that script. This swap is the actual fix, not a
workaround. **Still unvalidated against real land-record handwriting** —
a community fine-tune's quality varies more than an official release;
test it yourself before trusting its accuracy in a demo.

**1. Install the extra dependencies:**
```bash
pip install torch "transformers==4.46.3" sentencepiece
```
(The pinned version matters — see the note in `handwriting_ocr.py` about
newer transformers releases failing to load TrOCR's tokenizer on Windows.)

**2. Enable it in `backend/main.py`:**
```python
ENABLE_HANDWRITING_OCR = True
```

**3. Restart the server and upload an image.**
First upload after enabling downloads the model (~1.3GB) from Hugging
Face — needs real internet, will take a few minutes. Subsequent runs use
the cached model.

**4. How it works:** `hybrid_ocr.py` runs Tesseract first, then re-checks
any word Tesseract was unsure about (confidence below 60%) by re-running
that crop through TrOCR, keeping whichever engine was more confident.
Full details and honest limitations are documented in the docstrings of
`hybrid_ocr.py` and `handwriting_ocr.py` — read those before demoing this
part.

**Report back what you see** when you test this on real handwritten
samples — accuracy, speed, and any errors — so we can tune the confidence
threshold and decide whether to feature it prominently in your demo or
describe it as "designed and implemented, tuning in progress."

## Regenerating or expanding the sample dataset
```bash
cd scripts
python3 gen_data.py --count 30 --out ../data/sample_records
```
This regenerates records with different random villages/owners/khasra
numbers and a random mix of light/medium/heavy scan degradation. See
`ground_truth.json` in the output folder for the true field values, used
by `evaluate.py` to score accuracy.

## What's real vs. simulated — be upfront about this in Q&A

| Component | This PoC | Production plan (as on your slides) |
|---|---|---|
| Printed-text OCR | Tesseract (`eng+hin`), with an adaptive two-pass preprocessing recovery step for low-confidence scans (see EVALUATION.md) | Tesseract/PaddleOCR, same idea |
| Handwriting OCR | TrOCR fine-tuned for Devanagari (`paudelanil/trocr-devanagari-2`, community fine-tune on IIIT-INDIC-HW-WORDS), confidence-triggered — off by default, see setup section above. Not yet tested against real handwritten land-record samples | CRNN or TrOCR variant fine-tuned specifically on Indian regional handwriting (e.g. against IIIT-HW) — largely achieved by the swap above; further fine-tuning on land-record-specific handwriting would be the next refinement |
| Field mapping | Regex + fuzzy label matching | IndicBERT/IndicNER |
| Layout/doc-type detection | Single hardcoded layout | LayoutLMv3/Donut |
| Cross-validation | Rule-based diff vs. DB records | ML-based anomaly/matching model |
| Bhoomi/DILRMP/e-Dharti integration | Not connected — no public sandbox exists | Real API integration once government access is granted |
| GIS (Bhu-Naksha) | Not implemented | PostGIS + Bhu-Naksha overlay |
| Active-learning retraining | Not implemented (officer corrections are saved, but nothing retrains on them yet) | Retrain OCR/NER on accumulated corrections |
| Deployment | Local dev server | Docker/Kubernetes hybrid-cloud |

**Why this is fine to say out loud:** every substitution sits behind the
exact same interface the real component would use (`run_ocr()`,
`extract_fields()`, `cross_validate()`) — swapping in PaddleOCR or a trained
NER model later doesn't change the API contract, the database schema, or
the UI. That's the point to make if a judge asks "is this the real thing."

## Known limitations to mention proactively
- Combined `eng+hin` Tesseract language mode occasionally emits stray
  Devanagari characters when OCR confidence is already very low (visible on
  the heavy-degradation samples) — a realistic and disclosable failure mode,
  not a bug to hide.
- Only one document layout is supported; a second layout (e.g. a sale deed
  format) would need its own `FIELD_LABELS` schema in `extract.py`.
- Cross-validation only checks for an existing record with the same
  khasra_no + village — no fuzzy/partial matching yet.
