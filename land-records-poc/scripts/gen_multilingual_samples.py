"""
Multilingual Land Record Sample Generator
------------------------------------------
Generates realistic land record images in multiple languages:
  1. Hindi (खतौनी / अधिकार अभिलेख)
  2. Telugu (పట్టాదారు పాస్ పుస్తకం / పహానీ)
  3. Mixed / Bilingual (English + Indic scripts)
  4. English (Record of Rights / Khatauni)

Both clean and degraded scans are produced, along with ground_truth.json.
Output directory: data/sample_records/multilingual/
"""
import json
import os
import random
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Deterministic seed for reproducible evaluation
random.seed(42)

def get_indic_font(size: int):
    """Attempt to find a font capable of rendering Indic and English glyphs."""
    candidates = [
        r"C:\Windows\Fonts\Nirmala.ttc",
        r"C:\Windows\Fonts\nirmala.ttf",
        r"C:\Windows\Fonts\mangal.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        "/usr/share/fonts/truetype/lohit-telugu/Lohit-Telugu.ttf",
        "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf"
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()

def render_multilingual_page(title, subtitle, fields, footer_text, seal_text):
    """Renders a standard formal revenue record sheet."""
    page_w, page_h = 1300, 1800
    img = Image.new("RGB", (page_w, page_h), color=(252, 250, 242))
    draw = ImageDraw.Draw(img)

    title_font = get_indic_font(34)
    subtitle_font = get_indic_font(22)
    label_font = get_indic_font(25)
    val_font = get_indic_font(25)
    footer_font = get_indic_font(20)

    # Header decorative border
    draw.rectangle([40, 40, page_w - 40, page_h - 40], outline=(70, 70, 70), width=3)
    draw.rectangle([48, 48, page_w - 48, page_h - 48], outline=(150, 150, 150), width=1)

    # Header
    draw.text((page_w // 2 - 280, 80), title, font=title_font, fill=(15, 15, 15))
    draw.text((page_w // 2 - 190, 135), subtitle, font=subtitle_font, fill=(80, 80, 80))
    draw.line([(70, 180), (page_w - 70, 180)], fill=(30, 30, 30), width=2)

    # Table of fields
    y = 220
    row_h = 110
    for label, val in fields:
        # Outer row box
        draw.rectangle([70, y, page_w - 70, y + row_h - 15], outline=(160, 160, 160), width=1, fill=(255, 255, 255))
        # Label separator column
        draw.line([(440, y), (440, y + row_h - 15)], fill=(180, 180, 180), width=1)
        # Text
        draw.text((90, y + 25), label, font=label_font, fill=(20, 20, 20))
        draw.text((470, y + 25), str(val), font=val_font, fill=(10, 20, 80))
        y += row_h

    # Footer
    draw.line([(70, y + 20), (page_w - 70, y + 20)], fill=(30, 30, 30), width=1)
    draw.text((75, y + 50), footer_text, font=footer_font, fill=(50, 50, 50))

    # Official Seal / Stamp
    seal_x1, seal_y1 = page_w - 320, y + 30
    seal_x2, seal_y2 = page_w - 120, y + 230
    draw.ellipse([seal_x1, seal_y1, seal_x2, seal_y2], outline=(160, 25, 25), width=3)
    draw.text((seal_x1 + 45, seal_y1 + 75), seal_text, font=subtitle_font, fill=(160, 25, 25))

    return img

def degrade_image(pil_img, severity="light"):
    """Simulate scan artifacts: slight rotation, subtle noise, blur, compression."""
    img = np.array(pil_img.convert("RGB"))
    angle = random.uniform(-1.5, 1.5) if severity == "light" else random.uniform(-3.5, 3.5)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=(250, 248, 240))

    if severity == "medium":
        img = cv2.GaussianBlur(img, (3, 3), 0)
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85 if severity == "light" else 65])
    return Image.fromarray(cv2.imdecode(enc, cv2.IMREAD_COLOR))


# =====================================================================
# MULTILINGUAL TEST SAMPLES DEFINITIONS
# =====================================================================

DATASETS = {
    "hindi": {
        "title": "खतौनी - अधिकार अभिलेख",
        "subtitle": "(राजस्व परिषद - उत्तर प्रदेश)",
        "footer": "हस्ताक्षर: राजस्व निरीक्षक / पटवारी",
        "seal": "राजकीय मुहर\nOFFICIAL",
        "ground_truth": {
            "language": "hindi",
            "owner_name": "राम प्रसाद शर्मा",
            "father_name": "श्याम सुंदर शर्मा",
            "khasra_no": "452",
            "khata_no": "1084",
            "area_hectare": "1.75",
            "village": "रामपुर",
            "tehsil": "सदर",
            "district": "मेरठ",
            "land_type": "कृषि",
            "mutation_no": "M-3842"
        },
        "fields": [
            ("खातेदार का नाम:", "राम प्रसाद शर्मा"),
            ("पिता का नाम:", "श्याम सुंदर शर्मा"),
            ("खसरा संख्या:", "452"),
            ("खाता संख्या:", "1084"),
            ("क्षेत्रफल (हेक्टेयर):", "1.75"),
            ("ग्राम:", "रामपुर"),
            ("तहसील:", "सदर"),
            ("जिला:", "मेरठ"),
            ("भूमि प्रकार:", "कृषि"),
            ("नामांतरण संख्या:", "M-3842")
        ]
    },
    "telugu": {
        "title": "పట్టాదారు పాస్ పుస్తకం & పహానీ",
        "subtitle": "(భూ పరిపాలన శాఖ - తెలంగాణ / ఆంధ్రప్రదేశ్)",
        "footer": "సంతకం: మండల రెవెన్యూ అధికారి (MRO)",
        "seal": "అధికారిక ముద్ర\nOFFICIAL",
        "ground_truth": {
            "language": "telugu",
            "owner_name": "వెంకటేశ్వర రావు",
            "father_name": "సత్యనారాయణ",
            "survey_number": "238/4A",
            "khata_no": "1205",
            "area_hectare": "3.25",
            "village": "రాంపూర్",
            "mandal": "చౌటుప్పల్",
            "district": "యాదాద్రి",
            "land_type": "వ్యవసాయం",
            "mutation_no": "M-6194"
        },
        "fields": [
            ("పట్టాదారు పేరు:", "వెంకటేశ్వర రావు"),
            ("తండ్రి పేరు:", "సత్యనారాయణ"),
            ("సర్వే నంబర్:", "238/4A"),
            ("ఖాతా నంబర్:", "1205"),
            ("విస్తీర్ణం (ఎకరాలు):", "3.25"),
            ("గ్రామం:", "రాంపూర్"),
            ("మండలం:", "చౌటుప్పల్"),
            ("జిల్లా:", "యాదాద్రి"),
            ("భూమి వర్గీకరణ:", "వ్యవసాయం"),
            ("మ్యుటేషన్ నెం:", "M-6194")
        ]
    },
    "mixed": {
        "title": "RECORD OF RIGHTS / రెవెన్యూ రికార్డు",
        "subtitle": "(Revenue Department - Bilingual Record)",
        "footer": "Signature of Tahsildar / తహసీల్దార్ సంతకం",
        "seal": "OFFICIAL SEAL\nముద్ర",
        "ground_truth": {
            "language": "mixed (telugu+english)",
            "owner_name": "కె సురేష్ కుమార్",
            "father_name": "K Ramanaiah",
            "survey_number": "312/1B",
            "khata_no": "4420",
            "area_hectare": "2.10",
            "village": "Rampur",
            "mandal": "Sadar",
            "district": "Bareilly",
            "classification": "Agricultural",
            "mutation_no": "M-9210"
        },
        "fields": [
            ("Owner Name / పట్టాదారు:", "కె సురేష్ కుమార్"),
            ("Father Name / తండ్రి:", "K Ramanaiah"),
            ("Survey Number / సర్వే:", "312/1B"),
            ("Khata No / ఖాతా:", "4420"),
            ("Area (Hectare) / విస్తీర్ణం:", "2.10"),
            ("Village / గ్రామం:", "Rampur"),
            ("Mandal / మండలం:", "Sadar"),
            ("District / జిల్లా:", "Bareilly"),
            ("Land Classification:", "Agricultural"),
            ("Mutation No / మ్యుటేషన్:", "M-9210")
        ]
    },
    "english": {
        "title": "KHATAUNI - RECORD OF RIGHTS",
        "subtitle": "(Revenue Department Extract)",
        "footer": "Signature of Patwari / Revenue Officer",
        "seal": "OFFICIAL SEAL\nVERIFIED",
        "ground_truth": {
            "language": "english",
            "owner_name": "Ramesh Kumar",
            "father_name": "Harish Chandra",
            "khasra_no": "142/3",
            "khata_no": "3892",
            "area_hectare": "1.84",
            "village": "Rampur",
            "tehsil": "Meerganj",
            "district": "Bareilly",
            "classification": "Agricultural",
            "mutation_no": "M-5421"
        },
        "fields": [
            ("Owner Name:", "Ramesh Kumar"),
            ("Father Name:", "Harish Chandra"),
            ("Khasra No.:", "142/3"),
            ("Khata No.:", "3892"),
            ("Area (Hectare):", "1.84"),
            ("Village:", "Rampur"),
            ("Tehsil:", "Meerganj"),
            ("District:", "Bareilly"),
            ("Land Classification:", "Agricultural"),
            ("Mutation No.:", "M-5421")
        ]
    }
}

def generate_all():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_records", "multilingual"))
    os.makedirs(out_dir, exist_ok=True)

    all_ground_truth = {}

    for lang_key, data in DATASETS.items():
        base_img = render_multilingual_page(
            title=data["title"],
            subtitle=data["subtitle"],
            fields=data["fields"],
            footer_text=data["footer"],
            seal_text=data["seal"]
        )

        # 1. Clean version
        clean_filename = f"{lang_key}_clean.jpg"
        clean_path = os.path.join(out_dir, clean_filename)
        base_img.save(clean_path, quality=95)
        all_ground_truth[clean_filename] = {**data["ground_truth"], "_version": "clean"}

        # 2. Scanned / degraded version (light scan effect)
        scanned_img = degrade_image(base_img, severity="light")
        scanned_filename = f"{lang_key}_scanned.jpg"
        scanned_path = os.path.join(out_dir, scanned_filename)
        scanned_img.save(scanned_path, quality=80)
        all_ground_truth[scanned_filename] = {**data["ground_truth"], "_version": "scanned_light"}

        print(f"  [CREATED] {clean_filename} and {scanned_filename}")

    # Write ground truth JSON
    gt_path = os.path.join(out_dir, "ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(all_ground_truth, f, ensure_ascii=False, indent=2)

    print(f"\nMultilingual dataset generated successfully in:\n  {out_dir}")
    print(f"Ground truth saved to:\n  {gt_path}")

if __name__ == "__main__":
    generate_all()
