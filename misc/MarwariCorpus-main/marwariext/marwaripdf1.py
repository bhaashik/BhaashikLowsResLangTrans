import csv
import re
import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract

pdf_path = "Rajasthani03.pdf"
output_csv = "Rajasthani03_output.csv"

def clean_text(text):
    """Clean OCR text: remove junk, normalize Devanagari output."""
    text = re.sub(r'\s+', ' ', text)  # collapse whitespace
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)  # keep only Devanagari + spaces
    return text.strip()

def preprocess_image(pil_image):
    """Convert PIL image to OpenCV format and preprocess for better OCR."""
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Binarization
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    return thresh

def ocr_page(image):
    """Perform OCR with multiple PSM modes and return best text."""
    processed = preprocess_image(image)
    best_text = ""
    best_len = 0

    # Try multiple PSM modes to capture more text
    for psm in [3, 4, 6, 11, 12]:
        config = f"--oem 1 --psm {psm}"
        text = pytesseract.image_to_string(processed, lang="hin", config=config)
        text = clean_text(text)
        if len(text) > best_len:
            best_len = len(text)
            best_text = text

    return best_text

def extract_pdf_to_csv(pdf_path, output_csv):
    pages = convert_from_path(pdf_path, dpi=400)  # higher DPI = better OCR
    rows = []

    for i, page in enumerate(pages, start=1):
        print(f"OCR processing page {i}...")
        text = ocr_page(page)
        rows.append([pdf_path, i, text])

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pdf_file", "page", "text"])
        writer.writerows(rows)

    print(f"OCR completed. Output saved in {output_csv}")

# Run
extract_pdf_to_csv(pdf_path, output_csv)
