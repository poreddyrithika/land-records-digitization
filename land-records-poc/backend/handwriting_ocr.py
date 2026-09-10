"""
Handwriting OCR via TrOCR, fine-tuned for Devanagari script
(paudelanil/trocr-devanagari-2 — a community fine-tune of
microsoft/trocr-base-handwritten on the IIIT-INDIC-HW-WORDS dataset).

WHY NOT THE BASE ENGLISH MODEL:
An earlier version of this file used microsoft/trocr-base-handwritten
directly. Testing showed it hallucinating fluent English words on
non-English input — not just picking wrong words, but structurally unable
to produce Devanagari characters at all, since it was never trained on
that script. This model swaps in a checkpoint actually fine-tuned on
Devanagari handwriting data, which is the real fix, not a workaround.

STILL UNVALIDATED ON REAL DATA: this swap was made based on the model
card and dataset description, not by testing it against real handwritten
land-record samples (still don't have any — see project notes). Test it
on your machine against actual Devanagari handwriting before trusting the
accuracy; a community fine-tune's real-world quality varies a lot more
than an official Microsoft/Google release.

IMPORTANT, READ BEFORE RUNNING:
- Pin `transformers==4.46.3` (see requirements.txt) — newer transformers
  releases (5.x) failed to load TrOCR's tokenizer on Windows/Python 3.13
  during testing (a ValueError about instantiating the backend tokenizer).
  4.46.3 has a prebuilt Windows wheel and is confirmed working.
- Model is `paudelanil/trocr-devanagari-2`, NOT the base English model —
  a community fine-tune for Devanagari. If it's ever unavailable/removed
  from Hugging Face, fall back to `microsoft/trocr-base-handwritten`
  (English-only — expect hallucination on Devanagari input, see above).
- This needs `torch` and `transformers` installed (see requirements.txt).
- On first call, `load_model()` downloads ~1.3GB of model weights from
  Hugging Face. This requires a real internet connection on the machine
  running this code — it was NOT tested against the live model in the
  environment this project was built in, since that sandbox could not
  reach huggingface.co. Test this on your own machine and report back
  exactly what happens; treat first-run behavior as unverified until you
  confirm it.
- TrOCR is trained on ENGLISH handwriting. It was not fine-tuned on
  Devanagari or other Indian scripts. On Hindi/regional handwriting,
  expect low accuracy and low confidence — which is honest, not a bug:
  say so directly if asked, and note IndicTrOCR / fine-tuning on Indian
  handwriting datasets (e.g. IIIT-HW) as the production-grade next step.
- CPU inference is slow: expect a few seconds per line/word crop. Fine
  for a hackathon demo on a handful of images; do not claim this is
  fast in your pitch.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image

_model = None
_processor = None


MODEL_NAME = "paudelanil/trocr-devanagari-2"


def _lazy_load():
    """Load TrOCR only when first needed, so importing this module doesn't
    force a slow model load / download if handwriting OCR is never used."""
    global _model, _processor
    if _model is not None:
        return
    try:
        import torch  # noqa: F401
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError as e:
        raise ImportError(
            "TrOCR requires 'torch' and 'transformers'. Install with:\n"
            "  pip install torch transformers\n"
            f"Original error: {e}"
        )

    _processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    _model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)


@dataclass
class HandwritingResult:
    text: str
    conf: float  # 0-100, approximated from model generation score (see note in run_trocr)


def run_trocr(crop_bgr_or_gray: np.ndarray) -> HandwritingResult:
    """
    Run TrOCR on a single cropped image region (expected to contain one
    line or word of handwritten text — TrOCR is a line-level recognizer,
    not a full-page one, so the caller must crop to a single line first).

    Returns text plus a rough confidence score. TrOCR does not give a
    calibrated per-character confidence the way Tesseract does; here we
    approximate confidence from the model's sequence generation score,
    which is a weaker signal — treat it as directional, not precise, and
    lean on officer review more for handwritten fields than printed ones.
    """
    _lazy_load()

    if crop_bgr_or_gray.ndim == 2:
        pil_img = Image.fromarray(crop_bgr_or_gray).convert("RGB")
    else:
        pil_img = Image.fromarray(crop_bgr_or_gray[:, :, ::-1]).convert("RGB")  # BGR->RGB

    pixel_values = _processor(images=pil_img, return_tensors="pt").pixel_values

    import torch
    with torch.no_grad():
        outputs = _model.generate(
            pixel_values, output_scores=True, return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences
    text = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Approximate confidence: average of per-token max softmax probability.
    # This is a rough proxy, not a calibrated confidence — treat as
    # directional (higher = more likely correct), not as a precise percentage.
    try:
        scores = outputs.scores  # tuple of per-step logits
        import torch.nn.functional as F
        probs = [F.softmax(s, dim=-1).max().item() for s in scores]
        conf = float(np.mean(probs) * 100) if probs else 50.0
    except Exception:
        conf = 50.0  # fallback if score extraction fails for any reason

    return HandwritingResult(text=text.strip(), conf=round(conf, 1))


def run_trocr_batch(crops: List[np.ndarray]) -> List[HandwritingResult]:
    """Convenience wrapper for multiple line/word crops (e.g. all
    handwritten-classified regions from one document)."""
    return [run_trocr(c) for c in crops]
