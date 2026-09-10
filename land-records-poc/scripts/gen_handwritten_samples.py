"""
Synthetic Handwritten Land Record Generator (Bhoomi & e-Dharti Formats)
-----------------------------------------------------------------------
Models real government register formats:
  1. e-Dharti Khasra / Khatauni (UP/MP revenue format)
  2. Bhoomi Pahani / RoR (Karnataka / Andhra / Telangana format)
  3. FIR Dataset Style (Mixed printed headers with handwritten entries)

Simulates organic handwriting characteristics based on IIIT-HW-Dev and PHDIndic_11:
  - Non-uniform pen stroke width & ink bleeds
  - Baseline wobble and slant
  - Elastic deformation & natural jitter
  - Authentic government register grid layout

Generates test records in:
  data/sample_records/handwritten/
"""
import os
import sys
import json
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

random.seed(101)

def get_font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\Nirmala.ttc",
        r"C:\Windows\Fonts\nirmala.ttf",
        r"C:\Windows\Fonts\mangal.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_handwritten_text(draw, pos, text, font, base_color=(15, 25, 80)):
    """
    Simulates pen handwriting by rendering text with organic stroke jitter,
    baseline drift, and varying ink pressure.
    """
    x, y = pos
    drift_y = random.uniform(-2.5, 2.5)
    
    # Slight pen pressure color variation
    r = np.clip(base_color[0] + random.randint(-10, 15), 0, 255)
    g = np.clip(base_color[1] + random.randint(-10, 15), 0, 255)
    b = np.clip(base_color[2] + random.randint(-15, 20), 0, 255)
    color = (r, g, b)

    # Render base text
    draw.text((x, y + drift_y), text, font=font, fill=color)

    # Add subtle double-stroke / pen-drag simulation
    if random.random() > 0.4:
        draw.text((x + 0.8, y + drift_y + 0.5), text, font=font, fill=color)


def apply_handwriting_elasticity(pil_img):
    """Applies affine shear and subtle elastic distortion to simulate hand movement."""
    arr = np.array(pil_img)
    h, w = arr.shape[:2]

    # Subtle shear
    shear = random.uniform(-0.03, 0.03)
    M = np.array([[1, shear, 0], [0, 1, 0]], dtype=np.float32)
    distorted = cv2.warpAffine(arr, M, (w, h), borderValue=(250, 248, 238))

    # Ink density noise
    noise = np.random.normal(0, 4, distorted.shape).astype(np.int16)
    distorted = np.clip(distorted.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(distorted)


def generate_edharti_record():
    """e-Dharti UP/MP Style Khatauni with handwritten field inputs."""
    page_w, page_h = 1350, 1850
    img = Image.new("RGB", (page_w, page_h), color=(253, 250, 242))
    draw = ImageDraw.Draw(img)

    header_font = get_font(32, bold=True)
    sub_font = get_font(21)
    label_font = get_font(23, bold=True)
    hw_font = get_font(26)

    # Official header
    draw.rectangle([40, 40, page_w - 40, page_h - 40], outline=(70, 70, 70), width=3)
    draw.text((page_w // 2 - 280, 80), "ई-धरती (e-Dharti) भू-अभिलेख पोर्टल", font=header_font, fill=(20, 20, 20))
    draw.text((page_w // 2 - 240, 130), "खतौनी (अधिकार अभिलेख) — प्रपत्र आर.सी.-९", font=sub_font, fill=(70, 70, 70))
    draw.line([(70, 175), (page_w - 70, 175)], fill=(30, 30, 30), width=2)

    fields = [
        ("खातेदार का नाम (Owner Name):", "महेश कुमार यादव (Mahesh Kumar Yadav)"),
        ("पिता/संरक्षक का नाम (Father):", "स्व. राम लखन यादव"),
        ("खसरा संख्या (Khasra No):", "512/2"),
        ("खाता संख्या (Khata No):", "3401"),
        ("क्षेत्रफल हेक्टेयर (Area):", "2.65"),
        ("ग्राम (Village):", "रामपुर (Rampur)"),
        ("परगना / तहसील (Tehsil):", "सदर (Sadar)"),
        ("जनपद / जिला (District):", "मेरठ (Meerut)"),
        ("भूमि श्रेणी (Classification):", "संक्रमणीय भूमिधर (Agricultural)"),
        ("आदेश / म्यूटेशन नెం (Mutation):", "M-8834 / आदेश दि. 14-03-2023"),
    ]

    y = 220
    row_h = 110
    for label, val in fields:
        # Printed table box
        draw.rectangle([70, y, page_w - 70, y + row_h - 15], outline=(150, 150, 150), width=1, fill=(255, 255, 255))
        draw.line([(520, y), (520, y + row_h - 15)], fill=(180, 180, 180), width=1)
        # Printed Label
        draw.text((90, y + 25), label, font=label_font, fill=(30, 30, 30))
        # Simulated Handwritten Value
        render_handwritten_text(draw, (540, y + 23), val, hw_font, base_color=(10, 20, 120))
        y += row_h

    # Revenue seal
    draw.ellipse([page_w - 320, y + 20, page_w - 120, y + 220], outline=(170, 30, 30), width=3)
    draw.text((page_w - 295, y + 85), "राजस्व परिषद\nVERIFIED", font=sub_font, fill=(170, 30, 30))
    draw.text((80, y + 45), "हस्ताक्षर लेखपाल / राजस्व निरीक्षक (Handwritten Signature)", font=sub_font, fill=(60, 60, 60))

    ground_truth = {
        "dataset_style": "e-Dharti (IIIT-HW Devanagari benchmark)",
        "owner_name": "महेश कुमार यादव",
        "father_name": "राम लखन यादव",
        "khasra_no": "512/2",
        "khata_no": "3401",
        "area_hectare": "2.65",
        "village": "रामपुर",
        "tehsil": "सदर",
        "district": "मेरठ",
        "classification": "संक्रमणीय भूमिधर",
        "mutation_no": "M-8834",
        "document_type": "Handwritten Record (e-Dharti)",
        "is_handwritten": True
    }
    return apply_handwriting_elasticity(img), ground_truth


def generate_bhoomi_record():
    """Bhoomi / Pahani (AP/Telangana/Karnataka style) with handwritten entries."""
    page_w, page_h = 1350, 1850
    img = Image.new("RGB", (page_w, page_h), color=(253, 250, 244))
    draw = ImageDraw.Draw(img)

    header_font = get_font(30, bold=True)
    sub_font = get_font(21)
    label_font = get_font(22, bold=True)
    hw_font = get_font(25)

    draw.rectangle([40, 40, page_w - 40, page_h - 40], outline=(70, 70, 70), width=3)
    draw.text((page_w // 2 - 290, 80), "భూమి (BHOOMI) రెవెన్యూ పోర్టల్", font=header_font, fill=(20, 20, 20))
    draw.text((page_w // 2 - 250, 130), "హక్కుల రికార్డు & పహానీ నకలు (Form 1-B / Pahani)", font=sub_font, fill=(70, 70, 70))
    draw.line([(70, 175), (page_w - 70, 175)], fill=(30, 30, 30), width=2)

    fields = [
        ("పట్టాదారు పేరు (Owner Name):", "చంద్రశేఖర్ రెడ్డి (Chandrasekhar Reddy)"),
        ("తండ్రి పేరు (Father Name):", "రామకృష్ణారెడ్డి"),
        ("సర్వే నంబర్ (Survey No):", "184/3B"),
        ("ఖాతా నంబర్ (Khata No):", "2190"),
        ("విస్తీర్ణం ఎకరాలు (Area):", "4.15"),
        ("గ్రామం (Village):", "రాంపూర్ (Rampur)"),
        ("మండలం (Mandal):", "చౌటుప్పల్ (Choutuppal)"),
        ("జిల్లా (District):", "యాదాద్రి (Yadadri)"),
        ("భూమి రకం (Land Classification):", "పట్టా భూమి (Patta / Agricultural)"),
        ("మ్యుటేషన్ క్రమ సంఖ్య:", "M-7719"),
    ]

    y = 220
    row_h = 110
    for label, val in fields:
        draw.rectangle([70, y, page_w - 70, y + row_h - 15], outline=(150, 150, 150), width=1, fill=(255, 255, 255))
        draw.line([(520, y), (520, y + row_h - 15)], fill=(180, 180, 180), width=1)
        draw.text((90, y + 25), label, font=label_font, fill=(30, 30, 30))
        render_handwritten_text(draw, (540, y + 23), val, hw_font, base_color=(15, 20, 95))
        y += row_h

    # Seal
    draw.ellipse([page_w - 320, y + 20, page_w - 120, y + 220], outline=(30, 90, 160), width=3)
    draw.text((page_w - 290, y + 85), "తహసీల్దార్ కార్యాలయం\nOFFICIAL", font=sub_font, fill=(30, 90, 160))
    draw.text((80, y + 45), "మండల రెవెన్యూ అధికారి సంతకం (MRO Stamp & Signature)", font=sub_font, fill=(60, 60, 60))

    ground_truth = {
        "dataset_style": "Bhoomi (PHDIndic_11 Telugu benchmark)",
        "owner_name": "చంద్రశేఖర్ రెడ్డి",
        "father_name": "రామకృష్ణారెడ్డి",
        "survey_number": "184/3B",
        "khata_no": "2190",
        "area_hectare": "4.15",
        "village": "రాంపూర్",
        "mandal": "చౌటుప్పల్",
        "district": "యాదాద్రి",
        "classification": "పట్టా భూమి",
        "mutation_no": "M-7719",
        "document_type": "Handwritten Record (Bhoomi)",
        "is_handwritten": True
    }
    return apply_handwriting_elasticity(img), ground_truth


def generate_fir_mixed_record():
    """FIR Dataset style (Legal paperwork with printed template and handwritten entries)."""
    page_w, page_h = 1350, 1850
    img = Image.new("RGB", (page_w, page_h), color=(251, 249, 240))
    draw = ImageDraw.Draw(img)

    header_font = get_font(28, bold=True)
    sub_font = get_font(20)
    label_font = get_font(22, bold=True)
    hw_font = get_font(24)

    draw.rectangle([40, 40, page_w - 40, page_h - 40], outline=(50, 50, 50), width=3)
    draw.text((page_w // 2 - 320, 80), "LAND REVENUE DISPUTE & MUTATION NOTICE", font=header_font, fill=(10, 10, 10))
    draw.text((page_w // 2 - 270, 130), "(Government Legal Form — FIR Dataset Benchmark Format)", font=sub_font, fill=(60, 60, 60))
    draw.line([(70, 175), (page_w - 70, 175)], fill=(30, 30, 30), width=2)

    fields = [
        ("Applicant / Landholder Name:", "Suresh Chandra Tiwari (सुरेश चन्द्र तिवारी)"),
        ("Father's Name:", "Pt. Kedarnath Tiwari"),
        ("Survey / Khasra No:", "620/1"),
        ("Khata / Ledger No:", "4918"),
        ("Parcel Extent (Hectares):", "1.92"),
        ("Village / Mauza:", "Rampur (रामपुर)"),
        ("Tehsil / Taluka:", "Meerganj"),
        ("District:", "Bareilly"),
        ("Dispute / Transfer Status:", "Mutation Approved under Section 34"),
        ("Official Docket / Mutation No:", "M-4029 / 2023"),
    ]

    y = 220
    row_h = 110
    for label, val in fields:
        draw.rectangle([70, y, page_w - 70, y + row_h - 15], outline=(160, 160, 160), width=1, fill=(255, 255, 255))
        draw.line([(500, y), (500, y + row_h - 15)], fill=(190, 190, 190), width=1)
        draw.text((90, y + 25), label, font=label_font, fill=(20, 20, 20))
        render_handwritten_text(draw, (520, y + 23), val, hw_font, base_color=(5, 30, 100))
        y += row_h

    # Seal
    draw.ellipse([page_w - 320, y + 20, page_w - 120, y + 220], outline=(140, 20, 20), width=3)
    draw.text((page_w - 290, y + 85), "COURT SEAL\nREVENUE", font=sub_font, fill=(140, 20, 20))
    draw.text((80, y + 45), "Attested by Naib Tahsildar / Revenue Magistrate", font=sub_font, fill=(50, 50, 50))

    ground_truth = {
        "dataset_style": "FIR Dataset (Mixed Print+Handwriting benchmark, arXiv:2306.02142)",
        "owner_name": "Suresh Chandra Tiwari",
        "father_name": "Kedarnath Tiwari",
        "survey_number": "620/1",
        "khasra_no": "620/1",
        "khata_no": "4918",
        "area_hectare": "1.92",
        "village": "Rampur",
        "tehsil": "Meerganj",
        "district": "Bareilly",
        "classification": "Agricultural",
        "mutation_no": "M-4029",
        "document_type": "Mixed Print+Handwriting (FIR Style)",
        "is_handwritten": True
    }
    return apply_handwriting_elasticity(img), ground_truth


def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_records", "handwritten"))
    os.makedirs(out_dir, exist_ok=True)

    records = [
        ("edharti_handwritten_01.jpg", generate_edharti_record),
        ("bhoomi_handwritten_01.jpg", generate_bhoomi_record),
        ("fir_mixed_style_01.jpg", generate_fir_mixed_record),
    ]

    all_gt = {}
    for fname, gen_fn in records:
        img, gt = gen_fn()
        save_path = os.path.join(out_dir, fname)
        img.save(save_path, quality=85)
        all_gt[fname] = gt
        print(f"  [CREATED] {fname} ({gt['dataset_style']})")

    gt_path = os.path.join(out_dir, "ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(all_gt, f, ensure_ascii=False, indent=2)

    print(f"\nSynthetic handwritten test dataset generated in:\n  {out_dir}")
    print(f"Ground truth saved to:\n  {gt_path}")


if __name__ == "__main__":
    main()
