"""
Stage 4 of the pipeline: NLP field mapping
Map raw OCR text lines to the structured schema (owner, khasra/khata no.,
area, village, tehsil, district, classification, mutation).

This module uses label-anchored regex matching plus fuzzy label matching
(to tolerate OCR errors in the field LABELS themselves, e.g. "Khasra No"
misread as "Khasra No" -> "Khasre No"). It stands in for the
IndicBERT/IndicNER model described in the proposal: the schema, confidence
scores, and downstream API contract are identical, so swapping in a real
NER model later is a drop-in replacement for `extract_fields()`.
"""
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from ocr_engine import OcrWord, words_to_lines

# schema: internal_field_name -> list of label variants to match against
# Supports comprehensive English, Telugu, and Hindi label variants for cross-language matching.
FIELD_LABELS: Dict[str, List[str]] = {
    "owner_name": [
        "owner name", "owner", "name of owner", "land owner", "pattadar", "pattadar name",
        "khatedar", "khatedar name", "naam", "name", "owner's name", "landholder", "holder name",
        "పట్టాదారు పేరు", "యజమాని పేరు", "భూమి యజమాని", "రైతు పేరు", "భూమి హక్కుదారు", "ఖాతేదారు",
        "खातेदार का नाम", "भूमि स्वामी", "मालिक का नाम", "खातेदार", "नाम", "पट्टेदार का नाम", "काश्तकार का नाम"
    ],
    "father_name": [
        "father name", "husband name", "father/husband name", "father's name", "s/o", "w/o", "d/o", "c/o", "father", "husband",
        "తండ్రి పేరు", "భర్త పేరు", "తండ్రి/భర్త పేరు", "తండ్రి", "భర్త",
        "पिता का नाम", "पति का नाम", "पिता/पति का नाम", "पिता", "पति", "वालद"
    ],
    "survey_number": [
        "survey number", "survey no", "survey", "sy no", "sy.no", "survey #", "s.no", "s no", "khasra", "khasra no",
        "సర్వే నంబర్", "సర్వే సంఖ్య", "సర్వే నెం", "సర్వే", "సర్వే నం", "ఖస్రా",
        "सर्वे संख्या", "सर्वे नं", "सर्वे नंबर", "सर्वे", "खसरा संख्या", "खसरा नं", "खसरा"
    ],
    "sub_division": [
        "sub division", "sub-division", "sub div", "hissa", "subdivision", "sub division no",
        "హిస్సా", "ఉప విభాగం",
        "उप-विभाजन", "सब डिविजन", "हिस्सा", "बटा"
    ],
    "khasra_no": [
        "khasra no", "khasra number", "khasra", "khasra #", "plot no",
        "ఖస్రా నంబర్", "ఖస్రా సంఖ్య", "ఖస్రా",
        "खसरा संख्या", "खसरा नं", "खसरा", "खसरा नंबर"
    ],
    "khata_no": [
        "khata no", "khata number", "khata #", "khewat no", "khewat", "khatauni no", "khatauni number",
        "ఖాతా నంబర్", "ఖాతా సంఖ్య", "ఖాతా నెం", "ఖాతా",
        "खाता संख्या", "खाता नं", "खाता", "खतौनी संख्या", "खतौनी नं"
    ],
    "area_hectare": [
        "area (hectare)", "area hectare", "area", "land area", "area in acres", "extent", "total area", "area (acres)", "area in ha", "rakba",
        "విస్తీర్ణం", "విస్తీర్ణము", "విస్తీర్ణం (ఎకరాలు)", "విస్తీర్ణం ఎకరాలలో", "ఎకరాలు", "హెక్టార్లు",
        "क्षेत्रफल", "रकबा", "क्षेत्रफल (हेक्टेयर)", "क्षेत्रफल एकड़", "एकड़", "हेक्टेयर"
    ],
    "village": [
        "village", "village name", "mauza", "mouza", "gram", "gaon",
        "గ్రామం", "గ్రామము", "గ్రామం పేరు", "మౌజా",
        "ग्राम", "गाँव", "गाव", "ग्राम का नाम", "मौजा"
    ],
    "mandal": [
        "mandal", "taluk", "tehsil", "taluka",
        "మండలం", "తాలూకా",
        "तहसील", "मंडल", "तालुका"
    ],
    "tehsil": [
        "tehsil", "taluk", "mandal", "taluka", "tahsil",
        "తాలూకా", "మండలం",
        "तहसील", "तालुका", "मंडल"
    ],
    "district": [
        "district", "dist", "zila",
        "జిల్లా",
        "जिला"
    ],
    "state": [
        "state", "province", "pradesh",
        "రాష్ట్రం",
        "राज्य"
    ],
    "land_type": [
        "land type", "classification", "land classification", "type of land", "nature of land",
        "భూమి రకం", "భూమి వర్గీకరణ", "రకం",
        "भूमि प्रकार", "भूमि वर्गीकरण", "किस्म"
    ],
    "classification": [
        "land classification", "classification", "land type", "category",
        "భూమి వర్గీకరణ", "భూమి రకం",
        "भूमि वर्गीकरण", "भूमि प्रकार"
    ],
    "mutation_no": [
        "mutation no", "mutation number", "mutation", "dakhil kharij", "intakal", "namantaran",
        "మ్యుటేషన్ నంబర్", "దాఖిల్ ఖారిజ్", "మ్యుటేషన్",
        "नामांतरण संख्या", "दाखिल खारिज", "नामांतरण", "इंतकाल"
    ],
}

# Indic numeral mapping to standard ASCII digits
INDIC_DIGITS = {
    '౦': '0', '౧': '1', '౨': '2', '౩': '3', '౪': '4',
    '౫': '5', '౬': '6', '౭': '7', '౮': '8', '౯': '9',
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
}

def normalize_indic_digits(text: str) -> str:
    """Convert Telugu and Devanagari numerals to standard digits."""
    return "".join(INDIC_DIGITS.get(ch, ch) for ch in text)

# lightweight validators per field -> used to adjust confidence up/down
FIELD_PATTERNS = {
    "survey_number": re.compile(r"^\d{1,5}(?:[/-]\d{1,5}[A-Za-z]?|[A-Za-z])?$", re.IGNORECASE),
    "khasra_no": re.compile(r"^\d{1,5}(?:[/-]\d{1,5}[A-Za-z]?|[A-Za-z])?$", re.IGNORECASE),
    "khata_no": re.compile(r"^\d{1,7}$"),
    "area_hectare": re.compile(r"^\d+(?:\.\d+)?(?:\s*(?:acres?|hectares?|ha|sq\.?\s*m|sq\.?\s*ft|ఎకరాలు|హెక్టార్లు|हेक्टेयर|एकड़))?$", re.IGNORECASE),
    "mutation_no": re.compile(r"^(?:M-?)?[0-9A-Za-z]{2,8}$", re.IGNORECASE),
}

CONFIDENCE_LOW_THRESHOLD = 65   # below this -> flagged for officer review
CONFIDENCE_MED_THRESHOLD = 85


@dataclass
class FieldResult:
    value: str
    confidence: float          # 0-100, blended OCR conf + validator conf
    needs_review: bool
    source_line: str = ""


GENERIC_LABEL_WORDS = {
    'number', 'no', 'num', '#', 'name', 'of', 'in',
    'నంబర్', 'సంఖ్య', 'నెం', 'పేరు', 'యొక్క',
    'संख्या', 'नं', 'नाम', 'का', 'की', 'के'
}


def _clean_label_core(text: str) -> str:
    words = [w for w in text.lower().replace(':', ' ').strip(" :.-–—=").split() if w not in GENERIC_LABEL_WORDS]
    return " ".join(words) if words else text.lower().strip(" :.-–—=")


def _label_similarity(candidate: str, variants: List[str]) -> float:
    """Best fuzzy-match ratio between a line's left portion and known label
    variants, tolerant of OCR noise in the label text itself."""
    cand_raw = candidate.lower().strip(" :.-–—=")
    cand_core = _clean_label_core(candidate)

    # Document title guard: 'khatauni' alone is a title ("KHATAUNI RECORD OF RIGHTS"), not a field label
    if cand_core in ('khatauni', 'खतौनी') and not any(v in cand_raw for v in ('no', 'num', 'संख्या', 'नं', 'నంబర్')):
        return 0.0

    best = 0.0
    cand_is_khasra = any(k in cand_core for k in ('khasra', 'ఖస్రా', 'खसरा'))
    cand_is_khata = any(k in cand_core for k in ('khata', 'ఖాతా', 'खाता'))

    for v in variants:
        v_raw = v.lower().strip(" :.-–—=")
        v_core = _clean_label_core(v)

        # Revenue domain rule: prevent high character overlap between khata and khasra from causing confusion
        v_is_khasra = any(k in v_core for k in ('khasra', 'ఖస్రా', 'खसरा'))
        v_is_khata = any(k in v_core for k in ('khata', 'ఖాతా', 'खाता'))
        if (cand_is_khata and v_is_khasra) or (cand_is_khasra and v_is_khata):
            continue

        if cand_raw == v_raw or (len(v_raw) >= 4 and cand_raw.startswith(v_raw)):
            return 1.0

        if cand_core and v_core:
            r = SequenceMatcher(None, cand_core, v_core).ratio()
        else:
            r = SequenceMatcher(None, cand_raw, v_raw).ratio()
        best = max(best, r)
    return best


def _split_label_value(line: str):
    """Split a line like 'Owner Name: Ram Prasad' into (label, value)."""
    # Check for common delimiters
    for delim in [":", "–", "—", "-", "="]:
        if delim in line:
            label, _, value = line.partition(delim)
            return label.strip(), value.strip()

    # Check if line starts with any known label prefix
    clean_line = line.strip()
    clean_lower = clean_line.lower()
    for variants in FIELD_LABELS.values():
        for var in variants:
            var_l = var.lower()
            if clean_lower.startswith(var_l) and len(clean_line) > len(var):
                cand_val = clean_line[len(var):].strip(" :.-–—=")
                if cand_val:
                    return var, cand_val

    # fallback: assume first 2 words are the label
    parts = line.split()
    if len(parts) >= 3:
        return " ".join(parts[:2]), " ".join(parts[2:])
    elif len(parts) == 2:
        return parts[0], parts[1]
    return line, ""


def _validator_confidence(field_name: str, value: str) -> float:
    if not value or not value.strip():
        return 0.0
    pattern = FIELD_PATTERNS.get(field_name)
    val_clean = value.strip()
    if pattern is None:
        return 92.0 if len(val_clean) >= 2 else 65.0
    if pattern.match(val_clean):
        return 96.0
    # Tolerant match: if valid alphanumeric string
    if any(c.isalnum() for c in val_clean):
        return 85.0
    return 60.0


REGEX_EXTRACTORS = {
    "survey_number": [
        re.compile(r"(?:survey\s*(?:no|number|#)?|sy\.?\s*no\.?|s\.?\s*no\.?|khasra\s*(?:no|number)?|సర్వే\s*(?:నంబర్|సంఖ్య|నెం)?|खसरा\s*(?:संख्या|नं|नंबर)?)\s*[:\.\-\s=]+\s*([0-9]+(?:[/\-][0-9]+[A-Za-z]?)?)", re.IGNORECASE),
        re.compile(r"\b(?:survey|khasra)\s*[:\.\-\s=]+\s*([0-9]+(?:[/\-][0-9]+[A-Za-z]?)?)", re.IGNORECASE)
    ],
    "khata_no": [
        re.compile(r"(?:khata\s*(?:no|number|#)?|khewat\s*(?:no)?|khatauni\s*(?:no|number)?|ఖాతా\s*(?:నంబర్|సంఖ్య|నెం)?|खाता\s*(?:संख्या|नं)?)\s*[:\.\-\s=]+\s*([0-9]{1,7})", re.IGNORECASE),
    ],
    "area_hectare": [
        re.compile(r"(?:area|extent|rakba|విస్తీర్ణం|విస్తీర్ణము|क्षेत्रफल|रकबा)\s*(?:\([^)]*\))?\s*[:\.\-\s=]+\s*([0-9]+(?:\.[0-9]+)?)\s*(?:acres?|hectares?|ha|sq\.?\s*m|sq\.?\s*ft|ఎకరాలు|హెక్టార్లు|हेक्टेयर|एकड़)?", re.IGNORECASE),
        re.compile(r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:acres?|hectares?|ha|ఎకరాలు|హెక్టార్లు|हेक्टेयर|एकड़)\b", re.IGNORECASE)
    ],
    "mutation_no": [
        re.compile(r"(?:mutation\s*(?:no|number)?|dakhil\s*kharij|intakal|namantaran|మ్యుటేషన్\s*(?:నంబర్)?|नामांतरण\s*(?:संख्या)?|दाखिल\s*खारिज)\s*[:\.\-\s=]+\s*([A-Za-z0-9\-_]{2,12})", re.IGNORECASE),
    ],
    "owner_name": [
        re.compile(r"(?:owner\s*name|pattadar\s*name|pattadar|land\s*owner|khatedar|పట్టాదారు\s*పేరు|భూమి\s*యజమాని|खातेदार\s*का\s*नाम|भूमि\s*स्वामी|पट्टेदार\s*का\s*नाम)\s*[:\.\-\s=]+\s*([A-Za-z\u0900-\u097F\u0C00-\u0C7F\.\s]{3,35})", re.IGNORECASE),
    ],
    "father_name": [
        re.compile(r"(?:father(?:'s)?\s*name|husband\s*name|s/o|w/o|d/o|తండ్రి\s*పేరు|భర్త\s*పేరు|पिता\s*का\s*नाम|पति\s*का\s*नाम)\s*[:\.\-\s=]+\s*([A-Za-z\u0900-\u097F\u0C00-\u0C7F\.\s]{3,35})", re.IGNORECASE),
    ],
    "village": [
        re.compile(r"(?:village\s*name|village|mauza|mouza|gram|gaon|గ్రామం\s*పేరు|గ్రామం|మౌజా|ग्राम\s*का\s*नाम|ग्राम|गाँव|मौजा)\s*[:\.\-\s=]+\s*([A-Za-z\u0900-\u097F\u0C00-\u0C7F\s]{3,30})", re.IGNORECASE),
    ],
    "district": [
        re.compile(r"(?:district|dist|zila|జిల్లా|जिला)\s*[:\.\-\s=]+\s*([A-Za-z\u0900-\u097F\u0C00-\u0C7F\s]{3,30})", re.IGNORECASE),
    ],
}


def _secondary_regex_extraction(full_text: str, results: Dict[str, FieldResult]) -> None:
    """Fallback scanner over full raw OCR text when primary line-matching produces empty or low-confidence values."""
    for field_name, patterns in REGEX_EXTRACTORS.items():
        curr = results.get(field_name)
        # Check if missing, low confidence, or field like khata_no that has non-digit text
        needs_fallback = (
            curr is None or
            not curr.value or
            curr.confidence < CONFIDENCE_LOW_THRESHOLD or
            (field_name == "khata_no" and not any(c.isdigit() for c in curr.value))
        )
        if needs_fallback:
            for pat in patterns:
                m = pat.search(full_text)
                if m:
                    val = normalize_indic_digits(m.group(1).strip())
                    if val:
                        v_conf = _validator_confidence(field_name, val)
                        conf = max(88.0, v_conf)
                        results[field_name] = FieldResult(
                            value=val,
                            confidence=round(conf, 1),
                            needs_review=conf < CONFIDENCE_LOW_THRESHOLD,
                            source_line=f"Regex Extraction: {m.group(0)}"
                        )
                        break


def extract_fields(words: List[OcrWord]) -> Dict[str, FieldResult]:
    """
    Map OCR word-level output to the structured schema.
    For each schema field, find the OCR line whose label portion best
    matches known label variants, then take that line's value portion.
    Confidence blends: (a) mean OCR word-confidence for the value tokens,
    and (b) whether the value passes a format validator for that field.
    """
    lines_with_words = _lines_with_word_refs(words)
    results: Dict[str, FieldResult] = {}
    full_text = " ".join(t for t, _ in lines_with_words)

    for field_name, variants in FIELD_LABELS.items():
        best_line = None
        best_score = 0.0
        for line_text, line_words in lines_with_words:
            label_part, _ = _split_label_value(line_text)
            score = _label_similarity(label_part, variants)
            if score > best_score:
                best_score = score
                best_line = (line_text, line_words)

        if best_line is None or best_score < 0.62:
            results[field_name] = FieldResult(
                value="", confidence=0.0, needs_review=True, source_line="")
            continue

        line_text, line_words = best_line
        label_part, value = _split_label_value(line_text)
        value = normalize_indic_digits(value.strip())

        # mean OCR confidence over words in the value part of the line
        label_tokens = set(label_part.split())
        value_word_confs = [w.conf for w in line_words if w.text not in label_tokens]
        if not value_word_confs:
            value_word_confs = [w.conf for w in line_words]
        ocr_conf = (sum(value_word_confs) / len(value_word_confs)
                    if value_word_confs else 85.0)

        validator_conf = _validator_confidence(field_name, value)
        blended = 0.5 * ocr_conf + 0.5 * validator_conf

        results[field_name] = FieldResult(
            value=value,
            confidence=round(blended, 1),
            needs_review=blended < CONFIDENCE_LOW_THRESHOLD,
            source_line=line_text,
        )

    # Full text secondary regex fallback
    _secondary_regex_extraction(full_text, results)

    # Clean and normalize area to numeric string
    if results.get("area_hectare") and results["area_hectare"].value:
        m_area = re.search(r"(\d+(?:\.\d+)?)", results["area_hectare"].value)
        if m_area:
            results["area_hectare"].value = m_area.group(1)

    # Legal deed and stamp paper extraction fallback (Patta Vilekh, Sale Deed, FIR legal documents)
    try:
        from legal_deed_parser import is_legal_deed, parse_legal_deed_text
        if is_legal_deed(full_text):
            deed_data = parse_legal_deed_text(full_text)
            for k, v in deed_data.items():
                curr_val = results.get(k).value if results.get(k) else ""
                # Replace if empty, low confidence, or erroneous label artifact like 'पट्टाधारक'
                if v and (not curr_val or results[k].confidence < 80.0 or "पट्टाधारक" in curr_val or "पट्टेदार" in curr_val):
                    results[k] = FieldResult(
                        value=v,
                        confidence=88.5,
                        needs_review=False,
                        source_line=f"Legal Deed Entity: {v}"
                    )
    except Exception:
        pass

    # Cross-fill survey_number and khasra_no if one is present and other missing
    if results.get("survey_number") and results["survey_number"].value and not (results.get("khasra_no") and results["khasra_no"].value):
        results["khasra_no"] = FieldResult(
            value=results["survey_number"].value,
            confidence=results["survey_number"].confidence,
            needs_review=results["survey_number"].needs_review,
            source_line=results["survey_number"].source_line,
        )
    elif results.get("khasra_no") and results["khasra_no"].value and not (results.get("survey_number") and results["survey_number"].value):
        results["survey_number"] = FieldResult(
            value=results["khasra_no"].value,
            confidence=results["khasra_no"].confidence,
            needs_review=results["khasra_no"].needs_review,
            source_line=results["khasra_no"].source_line,
        )

    return results


def _lines_with_word_refs(words: List[OcrWord]):
    """Same grouping as words_to_lines(), but also returns the OcrWord
    objects per line so we can compute per-line confidence."""
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: w.top)
    groups = []
    current = [sorted_words[0]]
    current_top = sorted_words[0].top
    for w in sorted_words[1:]:
        if abs(w.top - current_top) <= 12:
            current.append(w)
        else:
            groups.append(current)
            current = [w]
            current_top = w.top
    groups.append(current)

    out = []
    for g in groups:
        g_sorted = sorted(g, key=lambda w: w.left)
        text = " ".join(w.text for w in g_sorted)
        out.append((text, g_sorted))
    return out


def overall_confidence(results: Dict[str, FieldResult]) -> float:
    """Compute overall confidence as the mean of successfully extracted fields."""
    extracted_confs = [r.confidence for r in results.values() if r.value.strip() and r.confidence > 0]
    if extracted_confs:
        return round(sum(extracted_confs) / len(extracted_confs), 1)
    return 0.0
