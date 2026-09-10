"""
Stage 3 of the pipeline: Multi-Language OCR Engine
Extract raw text per field, with word-level confidence scores.

UPGRADED:
- Multi-language support for English, Telugu, Hindi, and Mixed documents.
- Automatic TESSDATA_PREFIX configuration using local fast tessdata models.
- Script and language detection (Telugu unicode block, Devanagari/Hindi, Latin/English, Mixed).
- Flexible language selection: "auto", "english", "telugu", "hindi", or "mixed".
"""
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Configure Tesseract binary path
if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configure Tessdata directory
LOCAL_TESSDATA = os.path.join(os.path.dirname(__file__), "tessdata")
if os.path.exists(LOCAL_TESSDATA):
    os.environ["TESSDATA_PREFIX"] = LOCAL_TESSDATA


@dataclass
class OcrWord:
    text: str
    conf: float   # 0-100
    left: int
    top: int
    width: int
    height: int


def detect_language_from_text(text: str) -> Tuple[str, dict]:
    """
    Detect script and language from raw text using Unicode code points.
    Returns detected language: 'telugu', 'hindi', 'english', or 'mixed',
    along with character count statistics.
    """
    if not text:
        return "english", {"english": 0, "telugu": 0, "hindi": 0}

    telugu_chars = len(re.findall(r'[\u0C00-\u0C7F]', text))
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))

    counts = {
        "english": english_chars,
        "telugu": telugu_chars,
        "hindi": hindi_chars,
    }

    indic_total = telugu_chars + hindi_chars
    total_alphabetic = indic_total + english_chars

    if total_alphabetic == 0:
        return "english", counts

    # Check for mixed language
    if indic_total > 5 and english_chars > 5:
        if telugu_chars > hindi_chars:
            return "mixed (telugu+english)", counts
        return "mixed (hindi+english)", counts

    if telugu_chars > hindi_chars and telugu_chars > english_chars:
        return "telugu", counts
    elif hindi_chars > telugu_chars and hindi_chars > english_chars:
        return "hindi", counts
    else:
        return "english", counts


def get_tesseract_lang_code(language_preference: str) -> str:
    """
    Map user selection / auto-detected language to Tesseract language codes.
    """
    pref = (language_preference or "auto").lower()
    if pref == "telugu":
        return "tel+eng"
    elif pref == "hindi":
        return "hin+eng"
    elif pref == "english":
        return "eng"
    elif "mixed" in pref:
        return "eng+tel+hin"
    else:
        # Auto: check multi-lingual mode
        return "eng+hin+tel"


def run_ocr(image: np.ndarray, lang: str = "auto") -> List[OcrWord]:
    """
    Run Tesseract OCR on a pre-processed image and return word-level text
    with bounding boxes and confidence scores.
    """
    tess_lang = get_tesseract_lang_code(lang)

    # Fallback to available languages if specified combo fails
    try:
        data = pytesseract.image_to_data(
            image,
            lang=tess_lang,
            output_type=pytesseract.Output.DICT,
            config="--oem 3 --psm 4",
        )
    except Exception:
        # Fallback to eng if compound fails
        try:
            data = pytesseract.image_to_data(
                image,
                lang="eng",
                output_type=pytesseract.Output.DICT,
                config="--oem 3 --psm 4",
            )
        except Exception:
            return []

    words = []
    n_boxes = len(data.get("text", []))
    for i in range(n_boxes):
        txt = data["text"][i].strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0

        if txt and conf >= 0:  # tesseract emits -1 for non-text regions
            words.append(OcrWord(
                text=txt,
                conf=conf,
                left=int(data["left"][i]),
                top=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
            ))
    return words


def words_to_lines(words: List[OcrWord], y_tolerance: int = 14) -> List[str]:
    """
    Group words into lines based on vertical proximity, then sort left-to-right.
    """
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: w.top)
    lines = []
    current_line = [sorted_words[0]]
    current_top = sorted_words[0].top

    for w in sorted_words[1:]:
        if abs(w.top - current_top) <= y_tolerance:
            current_line.append(w)
        else:
            lines.append(current_line)
            current_line = [w]
            current_top = w.top
    lines.append(current_line)

    return [
        " ".join(w.text for w in sorted(line, key=lambda w: w.left))
        for line in lines
    ]


def get_full_ocr_text(words: List[OcrWord]) -> str:
    """Return reconstructed full text as a single string."""
    return "\n".join(words_to_lines(words))
