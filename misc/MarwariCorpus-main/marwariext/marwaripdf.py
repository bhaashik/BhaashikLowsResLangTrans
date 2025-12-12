import csv
import re
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output

pdf_path = "RJmarwari.pdf"
output_csv = "RJmarwari_output.csv"

def clean_text(text):
    """Clean OCR text: remove junk, normalize Devanagari output."""
    # Remove extra spaces, line breaks
    text = re.sub(r'\s+', ' ', text)
    # Remove junk ASCII-like symbols
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)  
    return text.strip()

def ocr_page(image):
    """Perform OCR on image with Hindi language model."""
    try:
        text = pytesseract.image_to_string(image, lang='hin', config="--psm 6")
        return clean_text(text)
    except Exception as e:
        print(f"OCR failed: {e}")
        return ""

def extract_pdf_to_csv(pdf_path, output_csv):
    pages = convert_from_path(pdf_path, dpi=300)  # high DPI for clarity
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
