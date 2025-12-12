import re

INPUT_FILE = "marwari_poems_only.txt"
OUTPUT_FILE = "pure_marwari_poems_cleaned.txt"

unwanted = [
    "हिन्दी", "उर्दू", "अंगिका", "अवधी", "गुजराती", "नेपाली",
    "भोजपुरी", "मैथिली", "राजस्थानी", "हरियाणवी",
    "कविता कोश", "श्रेणी", "लेखक", "भाषा", "Category"
]

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

cleaned_lines = []
for line in lines:
    text = line.strip()

    # skip completely empty or unwanted lines
    if not text or any(word in text for word in unwanted):
        continue

    # remove web links and English
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[A-Za-z0-9]", "", text)

    # keep only Devanagari chars and space/newline
    text = re.sub(r"[^ \u0900-\u097F]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text:
        cleaned_lines.append(text)

# join with real line breaks — poems will stay multiline
final_text = "\n".join(cleaned_lines)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(final_text)

print(f"✅ Cleaned file saved as: {OUTPUT_FILE}")
