import cv2
import numpy as np
import json
import sys
from ocr_engine import run_ocr
from extract import extract_fields

gt = json.load(open('../data/sample_records/ground_truth.json'))
FIELDS = ["owner_name","khasra_no","khata_no","area_hectare","village","tehsil","district","classification","mutation_no"]

def preprocess_variant(image_path, upscale=1.0, use_clahe=False, denoise_h=12, sharpen=False, denoise_method='nlmeans'):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if upscale != 1.0:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    if denoise_method == 'nlmeans':
        denoised = cv2.fastNlMeansDenoising(gray, h=denoise_h)
    elif denoise_method == 'bilateral':
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    elif denoise_method == 'median':
        denoised = cv2.medianBlur(gray, 3)
    else:
        denoised = gray
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        denoised = clahe.apply(denoised)
    if sharpen:
        blurred = cv2.GaussianBlur(denoised, (0,0), 2)
        denoised = cv2.addWeighted(denoised, 1.3, blurred, -0.3, 0)
    binarized = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return binarized

def score(files, **kwargs):
    total_match = 0; total_fields = 0
    for fname in files:
        path = f'../data/sample_records/{fname}'
        truth = gt[fname]
        clean = preprocess_variant(path, **kwargs)
        words = run_ocr(clean)
        fields = extract_fields(words)
        for k in FIELDS:
            total_fields += 1
            if fields[k].value.strip().lower() == str(truth[k]).strip().lower():
                total_match += 1
    return total_match, total_fields

if __name__ == "__main__":
    import ast
    cfg = ast.literal_eval(sys.argv[1])
    severity = sys.argv[2]
    files = [f for f,v in gt.items() if v['_severity']==severity]
    m, t = score(files, **cfg)
    print(f"{severity}: {100*m/t:.1f}% ({m}/{t})  cfg={cfg}")
