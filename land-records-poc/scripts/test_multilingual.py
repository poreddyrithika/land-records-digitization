"""
Multilingual Test Runner for Land Records Digitization Pipeline
----------------------------------------------------------------
Validates:
  1. Script & Language Detection (Hindi, Telugu, English, Mixed)
  2. Text Input Parsing & Regex/NLP Field Extraction across languages
  3. Image OCR Pipeline (End-to-end preprocessing, OCR, extraction)
  4. Accuracy comparison against Ground Truth
"""
import os
import sys
import json
import cv2

# Set stdout encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add backend to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from ocr_engine import run_ocr, words_to_lines, get_full_ocr_text, detect_language_from_text, OcrWord
from extract import extract_fields, overall_confidence
from preprocess import preprocess_with_fallback

# Sample text inputs for direct NLP & Field extraction testing
TEXT_TEST_INPUTS = {
    "hindi": {
        "text": """
खतौनी - अधिकार अभिलेख
खातेदार का नाम: राम प्रसाद शर्मा
पिता का नाम: श्याम सुंदर शर्मा
खसरा संख्या: 452
खाता संख्या: 1084
क्षेत्रफल (हेक्टेयर): 1.75
ग्राम: रामपुर
तहसील: सदर
जिला: मेरठ
भूमि प्रकार: कृषि
नामांतरण संख्या: M-3842
""",
        "expected_lang": "hindi",
        "expected_fields": {
            "owner_name": "राम प्रसाद शर्मा",
            "khasra_no": "452",
            "khata_no": "1084",
            "area_hectare": "1.75",
            "village": "रामपुर",
            "tehsil": "सदर",
            "district": "मेरठ"
        }
    },
    "telugu": {
        "text": """
పట్టాదారు పాస్ పుస్తకం & పహానీ
పట్టాదారు పేరు: వెంకటేశ్వర రావు
తండ్రి పేరు: సత్యనారాయణ
సర్వే నంబర్: 238/4A
ఖాతా నంబర్: 1205
విస్తీర్ణం (ఎకరాలు): 3.25
గ్రామం: రాంపూర్
మండలం: చౌటుప్పల్
జిల్లా: యాదాద్రి
భూమి వర్గీకరణ: వ్యవసాయం
మ్యుటేషన్ నెం: M-6194
""",
        "expected_lang": "telugu",
        "expected_fields": {
            "owner_name": "వెంకటేశ్వర రావు",
            "survey_number": "238/4A",
            "khata_no": "1205",
            "area_hectare": "3.25",
            "village": "రాంపూర్",
            "mandal": "చౌటుప్పల్",
            "district": "యాదాద్రి"
        }
    },
    "mixed": {
        "text": """
RECORD OF RIGHTS / రెవెన్యూ రికార్డు
Owner Name / పట్టాదారు: కె సురేష్ కుమార్
Father Name / తండ్రి: K Ramanaiah
Survey Number / సర్వే: 312/1B
Khata No / ఖాతా: 4420
Area (Hectare) / విస్తీర్ణం: 2.10
Village / గ్రామం: Rampur
Mandal / మండలం: Sadar
District / జిల్లా: Bareilly
Land Classification: Agricultural
Mutation No / మ్యుటేషన్: M-9210
""",
        "expected_lang": "mixed (telugu+english)",
        "expected_fields": {
            "owner_name": "కె సురేష్ కుమార్",
            "survey_number": "312/1B",
            "khata_no": "4420",
            "area_hectare": "2.10",
            "village": "Rampur"
        }
    },
    "english": {
        "text": """
KHATAUNI - RECORD OF RIGHTS
Owner Name: Ramesh Kumar
Father Name: Harish Chandra
Khasra No.: 142/3
Khata No.: 3892
Area (Hectare): 1.84
Village: Rampur
Tehsil: Meerganj
District: Bareilly
Land Classification: Agricultural
Mutation No.: M-5421
""",
        "expected_lang": "english",
        "expected_fields": {
            "owner_name": "Ramesh Kumar",
            "khasra_no": "142/3",
            "khata_no": "3892",
            "area_hectare": "1.84",
            "village": "Rampur",
            "tehsil": "Meerganj",
            "district": "Bareilly"
        }
    }
}


def test_text_inputs():
    print("=" * 70)
    print("TEST SUITE 1: TEXT-BASED MULTILINGUAL INPUT TESTING")
    print("=" * 70)

    for lang_name, payload in TEXT_TEST_INPUTS.items():
        text = payload["text"].strip()
        expected_lang = payload["expected_lang"]
        expected_fields = payload["expected_fields"]

        # 1. Test script detection
        detected_lang, stats = detect_language_from_text(text)
        lang_match = expected_lang in detected_lang
        print(f"\n[Language: {lang_name.upper()}]")
        print(f"  Detected Script/Language: {detected_lang} (Expected: {expected_lang}) -> {'PASS' if lang_match else 'FAIL'}")
        print(f"  Character Counts: {stats}")

        # 2. Test field extraction by converting text lines to mock OcrWord list
        mock_words = []
        y = 100
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            x = 50
            for word in line.split():
                mock_words.append(OcrWord(text=word, conf=95.0, left=x, top=y, width=len(word)*15, height=25))
                x += len(word) * 15 + 10
            y += 40

        extracted = extract_fields(mock_words)
        print("  Extracted Key Fields:")
        for k, exp_val in expected_fields.items():
            act_val = extracted[k].value if k in extracted else ""
            status = "MATCH" if (exp_val.lower() in act_val.lower() or act_val.lower() in exp_val.lower()) else "DIFF"
            print(f"    - {k:15s}: Expected '{exp_val}' | Extracted '{act_val}' [{status}]")


def test_image_inputs():
    print("\n" + "=" * 70)
    print("TEST SUITE 2: IMAGE-BASED OCR & PIPELINE VERIFICATION")
    print("=" * 70)

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_records", "multilingual"))
    gt_file = os.path.join(data_dir, "ground_truth.json")

    if not os.path.exists(gt_file):
        print(f"Ground truth not found at {gt_file}. Please run gen_multilingual_samples.py first.")
        return

    with open(gt_file, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Test clean images
    clean_images = [k for k in ground_truth.keys() if "_clean" in k]

    for fname in clean_images:
        img_path = os.path.join(data_dir, fname)
        gt = ground_truth[fname]
        expected_lang = gt.get("language", "english")

        print(f"\n[IMAGE TEST: {fname}] (Target: {expected_lang})")
        if not os.path.exists(img_path):
            print(f"  File not found: {img_path}")
            continue

        try:
            clean_img, _ = preprocess_with_fallback(img_path)
            ocr_words = run_ocr(clean_img, lang=expected_lang)
            full_text = get_full_ocr_text(ocr_words)
            detected_lang, counts = detect_language_from_text(full_text)

            print(f"  Detected Language : {detected_lang}")
            print(f"  OCR Words Count   : {len(ocr_words)}")
            
            extracted = extract_fields(ocr_words)
            conf = overall_confidence(extracted)
            print(f"  Extraction Conf   : {conf:.1f}%")

            # Check a few primary fields
            fields_to_check = ["owner_name", "khasra_no", "survey_number", "khata_no", "area_hectare", "village"]
            for fld in fields_to_check:
                if fld in gt and gt[fld]:
                    val = extracted[fld].value if fld in extracted else ""
                    print(f"    - {fld:15s}: GT = '{gt[fld]}' | OCR = '{val}'")

        except Exception as e:
            print(f"  Error processing image: {e}")

    print("\n" + "=" * 70)
    print("MULTILINGUAL TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_text_inputs()
    test_image_inputs()
