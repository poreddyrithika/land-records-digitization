"""
*** NOT CURRENTLY USED BY THE PIPELINE — kept for reference/future work ***
hybrid_ocr.py uses Tesseract's own per-word confidence to decide when to
retry with TrOCR instead of this module, because testing this heuristic
against our own known-printed synthetic samples showed it confidently
mislabeling clean printed text as handwritten (95% confidence, wrong) —
see hybrid_ocr.py's docstring for the full reasoning. Left in the repo
because the stroke-width/baseline approach is a reasonable starting point
if you get real labeled printed/handwritten samples to calibrate
thresholds against later; don't wire it back in un-calibrated.

Stage 2 of the pipeline: Detect type & layout
Classifies each text-bearing region of the document as PRINTED or
HANDWRITTEN, so the right OCR engine gets used per region — Tesseract for
printed labels/typed text, TrOCR for handwritten entries.

WHY A HEURISTIC, NOT A TRAINED CLASSIFIER:
A real production system would use a trained CNN (the proposal's slide
mentions LayoutLMv3/Donut for this). Training one needs labeled
printed-vs-handwritten data, which we don't have (see the dataset
discussion — no real samples were available at build time). This module
uses geometric heuristics on stroke shape instead: handwriting has more
irregular stroke width, less consistent baseline alignment, and higher
variance in connected-component spacing than typeset text. It is
noticeably weaker than a trained model and should be described as such —
say plainly in Q&A that this is a placeholder for LayoutLMv3/Donut, not a
replacement.

Swap point for later: replace `classify_region()`'s body with a real
model's predict call; the return type (a Region with a `.kind` field)
stays the same, so nothing downstream changes.
"""
from dataclasses import dataclass
from typing import List, Literal

import cv2
import numpy as np

RegionKind = Literal["printed", "handwritten"]


@dataclass
class Region:
    bbox: tuple           # (x, y, w, h)
    kind: RegionKind
    kind_confidence: float  # 0-100, heuristic-derived, not a calibrated probability


def _stroke_width_variance(crop: np.ndarray) -> float:
    """Printed fonts have near-uniform stroke width; handwriting varies a
    lot within the same letter. Approximate stroke width via distance
    transform on the binarized region and measure its variance."""
    if crop.size == 0:
        return 0.0
    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    stroke_px = dist[dist > 0]
    if stroke_px.size < 10:
        return 0.0
    return float(np.std(stroke_px) / (np.mean(stroke_px) + 1e-6))


def _baseline_irregularity(crop: np.ndarray) -> float:
    """Printed text sits on a near-perfectly straight baseline; handwriting
    wobbles. Approximate by looking at the row-wise center of mass of ink
    across the region's width and measuring how much it drifts."""
    if crop.size == 0:
        return 0.0
    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    col_sums = binary.sum(axis=0)
    ys = []
    h = binary.shape[0]
    for x in range(binary.shape[1]):
        col = binary[:, x]
        if col.sum() > 0:
            ys.append(np.average(np.arange(h), weights=col))
    if len(ys) < 5:
        return 0.0
    return float(np.std(ys))


def classify_region(gray_image: np.ndarray, bbox: tuple) -> Region:
    """
    Classify a single word/line region as printed or handwritten using
    stroke-width variance + baseline irregularity. Both signals are
    normalized against empirically-chosen thresholds tuned on our
    synthetic printed samples — these thresholds have NOT been validated
    against real handwritten samples and will need recalibration once
    real data is available.
    """
    x, y, w, h = bbox
    crop = gray_image[y:y + h, x:x + w]

    stroke_var = _stroke_width_variance(crop)
    baseline_irr = _baseline_irregularity(crop)

    # Higher variance/irregularity -> more likely handwritten.
    # These weights and thresholds are heuristic starting points, not
    # calibrated on labeled data.
    handwritten_score = min(100.0, (stroke_var * 120) + (baseline_irr * 8))

    kind: RegionKind = "handwritten" if handwritten_score > 40 else "printed"
    confidence = abs(handwritten_score - 40) + 50  # crude distance-from-boundary confidence
    confidence = min(95.0, confidence)

    return Region(bbox=bbox, kind=kind, kind_confidence=round(confidence, 1))


def classify_word_boxes(gray_image: np.ndarray, word_boxes: List[tuple]) -> List[Region]:
    """Classify a batch of (x, y, w, h) word boxes (e.g. from Tesseract's
    layout pass) as printed or handwritten."""
    return [classify_region(gray_image, box) for box in word_boxes]
