"""
Combines stages 2 + 3: run Tesseract first, then re-check any word it was
unsure about using TrOCR as a second opinion. This is the module main.py
calls instead of ocr_engine.run_ocr() directly, once handwriting support
is wanted.

DESIGN NOTE - WHY CONFIDENCE-TRIGGERED, NOT A SEPARATE CLASSIFIER:
An earlier version of this module used a standalone printed-vs-handwritten
classifier (doc_classifier.py, kept in the repo for reference) based on
stroke-width/baseline heuristics. Testing it against our own known-printed
synthetic samples showed it confidently misclassifying clean printed text
as handwritten — without real labeled handwriting samples to calibrate
against, that heuristic is not trustworthy, and shipping it as the primary
router risked doing more harm than good (routing good printed text through
a slower, weaker-on-clean-text model).

Using Tesseract's own per-word confidence as the trigger instead is more
defensible: it doesn't need a separate untested model, and "OCR wasn't
confident about this word" is a reasonable, honestly-describable proxy for
"maybe try a different engine on this one" — whether the underlying cause
is handwriting, damage, or noise. Say exactly this if a judge asks why
it's not a trained classifier.

Returns the same OcrWord list shape as ocr_engine.run_ocr(), so
extract.py's field-mapping logic needs zero changes.
"""
from typing import List

import numpy as np

from ocr_engine import OcrWord, run_ocr


def run_hybrid_ocr(image: np.ndarray, use_handwriting: bool = True,
                    retry_conf_threshold: float = 60.0) -> List[OcrWord]:
    """
    1. Run Tesseract on the whole image (handles printed text well, cheap).
    2. For any word Tesseract returned with confidence below
       `retry_conf_threshold`, re-run just that crop through TrOCR.
    3. Keep whichever result reports the higher confidence for that word.

    Set use_handwriting=False to skip TrOCR entirely (e.g. if torch/
    transformers aren't installed yet, or for a fast printed-only demo) —
    this is main.py's default until handwriting support is installed and
    tested on real samples.
    """
    tesseract_words = run_ocr(image)

    if not use_handwriting or not tesseract_words:
        return tesseract_words

    # TrOCR import is deferred to here so the whole app doesn't fail to
    # start if torch/transformers aren't installed and handwriting is off.
    try:
        from handwriting_ocr import run_trocr
    except ImportError:
        # torch/transformers missing - silently fall back to printed-only.
        # main.py should surface a warning to the user in this case.
        return tesseract_words

    final_words: List[OcrWord] = []
    for w in tesseract_words:
        if w.conf >= retry_conf_threshold:
            final_words.append(w)
            continue

        crop = image[w.top:w.top + w.height, w.left:w.left + w.width]
        if crop.size == 0:
            final_words.append(w)
            continue

        hw_result = run_trocr(crop)
        # Keep whichever engine was more confident about this word.
        if hw_result.conf > w.conf:
            final_words.append(OcrWord(
                text=hw_result.text, conf=hw_result.conf,
                left=w.left, top=w.top, width=w.width, height=w.height,
            ))
        else:
            final_words.append(w)

    return final_words
