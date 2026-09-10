"""
Synthetic Land Record Generator
--------------------------------
Generates realistic-looking khata/khasra register page images (with known
ground truth) and applies scan-like degradation (noise, blur, skew, fading)
so the OCR pipeline can be tested against something that behaves like a
real scanned document, since no public dataset of actual land records
exists.

Usage:
    python3 gen_data.py --count 20 --out ../data/sample_records
"""
import argparse
import json
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

random.seed(42)

VILLAGES = ["Rampur", "Sultanpur", "Devgaon", "Kishangarh", "Madhopur",
            "Bhagwanpur", "Nandgaon", "Chandpur", "Lakhanpur", "Govindpur"]
TEHSILS = ["Sadar", "Baraut", "Chhata", "Kotwali", "Meerganj"]
DISTRICTS = ["Meerut", "Bareilly", "Aligarh", "Etawah", "Kanpur Dehat"]
OWNER_FIRST = ["Ram", "Shyam", "Suresh", "Rajesh", "Mahesh", "Sunita", "Geeta",
               "Anita", "Vinod", "Ramesh", "Kamla", "Radha", "Om", "Suresh Chandra"]
OWNER_LAST = ["Prasad", "Kumar", "Singh", "Yadav", "Verma", "Sharma", "Devi",
              "Gupta", "Tiwari", "Chaudhary"]
CLASSIFICATION = ["Agricultural", "Residential", "Barren", "Irrigated", "Non-Irrigated"]


def rand_owner():
    return f"{random.choice(OWNER_FIRST)} {random.choice(OWNER_LAST)}"


def make_record():
    """Return a dict of ground-truth field values for one record."""
    return {
        "owner_name": rand_owner(),
        "khasra_no": str(random.randint(100, 999)),
        "khata_no": str(random.randint(1000, 9999)),
        "area_hectare": f"{round(random.uniform(0.1, 5.0), 2)}",
        "village": random.choice(VILLAGES),
        "tehsil": random.choice(TEHSILS),
        "district": random.choice(DISTRICTS),
        "classification": random.choice(CLASSIFICATION),
        "mutation_no": f"M-{random.randint(1000, 9999)}",
    }


def render_page(record, page_w=1240, page_h=1754):
    """Render one record as a register-page image (PIL), table-style,
    similar to a khatauni extract layout."""
    img = Image.new("RGB", (page_w, page_h), color=(250, 248, 240))
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
        label_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        value_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    except OSError:
        title_font = label_font = value_font = ImageFont.load_default()

    draw.text((page_w // 2 - 260, 60), "KHATAUNI - RECORD OF RIGHTS",
               font=title_font, fill=(20, 20, 20))
    draw.text((page_w // 2 - 150, 105), "(Revenue Department Extract)",
               font=value_font, fill=(60, 60, 60))
    draw.line([(80, 150), (page_w - 80, 150)], fill=(0, 0, 0), width=2)

    fields = [
        ("Owner Name", record["owner_name"]),
        ("Khasra No.", record["khasra_no"]),
        ("Khata No.", record["khata_no"]),
        ("Area (Hectare)", record["area_hectare"]),
        ("Village", record["village"]),
        ("Tehsil", record["tehsil"]),
        ("District", record["district"]),
        ("Land Classification", record["classification"]),
        ("Mutation No.", record["mutation_no"]),
    ]

    y = 210
    row_h = 90
    for label, value in fields:
        draw.rectangle([80, y, page_w - 80, y + row_h - 10], outline=(120, 120, 120), width=1)
        draw.text((100, y + 15), f"{label}:", font=label_font, fill=(10, 10, 10))
        draw.text((450, y + 15), value, font=value_font, fill=(0, 0, 80))
        y += row_h

    draw.line([(80, y + 20), (page_w - 80, y + 20)], fill=(0, 0, 0), width=1)
    draw.text((80, y + 40), "Signature of Patwari / Revenue Officer",
               font=value_font, fill=(30, 30, 30))
    draw.ellipse([page_w - 300, y + 30, page_w - 150, y + 180], outline=(150, 0, 0), width=2)
    draw.text((page_w - 280, y + 90), "OFFICIAL\nSEAL", font=value_font, fill=(150, 0, 0))

    return img


def degrade(pil_img, severity="medium"):
    """Apply scan-like artifacts: rotation/skew, gaussian noise, blur,
    contrast fade, and jpeg-style compression artifacts."""
    img = np.array(pil_img.convert("RGB"))

    # random skew
    angle = random.uniform(-3, 3) if severity != "heavy" else random.uniform(-8, 8)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=(245, 245, 235))

    # gaussian blur (simulates poor focus)
    if severity in ("medium", "heavy"):
        k = 3 if severity == "medium" else 5
        img = cv2.GaussianBlur(img, (k, k), 0)

    # noise
    noise_level = 8 if severity == "light" else (15 if severity == "medium" else 30)
    noise = np.random.normal(0, noise_level, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # contrast fade (old paper look)
    fade = random.uniform(0.75, 0.95) if severity != "light" else random.uniform(0.9, 1.0)
    img = np.clip(img.astype(np.float32) * fade + 255 * (1 - fade) * 0.5, 0, 255).astype(np.uint8)

    # jpeg re-encode to add compression artifacts
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(35, 70)])
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return Image.fromarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--out", type=str, default="../data/sample_records")
    args = ap.parse_args()

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.out))
    os.makedirs(out_dir, exist_ok=True)

    ground_truth = {}
    severities = ["light", "medium", "medium", "heavy"]  # weighted mix

    for i in range(args.count):
        record = make_record()
        page = render_page(record)
        severity = random.choice(severities)
        degraded = degrade(page, severity=severity)

        fname = f"record_{i:03d}.jpg"
        degraded.save(os.path.join(out_dir, fname), quality=80)
        ground_truth[fname] = {**record, "_severity": severity}

    with open(os.path.join(out_dir, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Generated {args.count} synthetic records in {out_dir}")


if __name__ == "__main__":
    main()
