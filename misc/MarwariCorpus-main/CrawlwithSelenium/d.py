from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import re
import csv
import time
from urllib.parse import urljoin, urlparse

# ----------------------
# CONFIGURATION
# ----------------------
SEED_URL = "https://www.aapanorajasthan.org/aapani%20bhasa.php"
OUTPUT_FILE = "clean_marwari_text.csv"
MAX_PAGES = 5  # limit crawling depth for safety

# ----------------------
# DEVANAGARI PATTERN
# ----------------------
# Match full Devanagari text including punctuation
DEVNAGARI_PATTERN = re.compile(r'[।\u0900-\u097F\s,.;:!?“”"\'-]+')

# Optional stopword filter (common Hindi words to exclude)
COMMON_HINDI_WORDS = {
    "है", "हैं", "और", "से", "के", "का", "की", "को", "पर", "में",
    "यह", "कि", "तो", "ही", "था", "थे", "थे", "थे", "था", "था", "था",
    "एक", "नहीं", "जो", "ने", "पर", "रहा", "रही", "था", "थी", "था",
    "भी", "अब", "तक", "पर", "का", "की", "के", "में", "से"
}

# ----------------------
# SELENIUM SETUP
# ----------------------
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

visited = set()
clean_paragraphs = []

# ----------------------
# HELPERS
# ----------------------
def extract_devanagari_paragraphs(text):
    """Extract continuous Devanagari text blocks and clean them."""
    blocks = DEVNAGARI_PATTERN.findall(text)
    results = []
    for b in blocks:
        b = re.sub(r'\s+', ' ', b).strip()
        if len(b) < 25:  # skip very short fragments
            continue

        # check if paragraph is mostly Devanagari (>=80% chars)
        devanagari_chars = sum(1 for c in b if '\u0900' <= c <= '\u097F')
        ratio = devanagari_chars / max(len(b), 1)
        if ratio < 0.8:
            continue

        # filter out common Hindi-heavy text
        if not any(stop in b.split() for stop in COMMON_HINDI_WORDS):
            results.append(b)
    return results

def get_internal_links(soup, base_url):
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        abs_href = urljoin(base_url, href)
        if urlparse(abs_href).netloc == urlparse(base_url).netloc:
            links.add(abs_href)
    return links

# ----------------------
# MAIN CRAWLER
# ----------------------
def crawl(url):
    if len(visited) >= MAX_PAGES or url in visited:
        return
    try:
        driver.get(url)
        time.sleep(2)
        visited.add(url)
        print(f"🌐 Crawling: {url}")

        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        # Extract text only from visible content areas
        main_tags = soup.find_all(["p", "div", "span"], recursive=True)
        text = " ".join(tag.get_text(separator=" ") for tag in main_tags)
        blocks = extract_devanagari_paragraphs(text)
        clean_paragraphs.extend(blocks)

        # Follow internal links if needed
        for link in get_internal_links(soup, SEED_URL):
            if link not in visited:
                crawl(link)

    except Exception as e:
        print(f"❌ Error at {url}: {e}")

# ----------------------
# RUN
# ----------------------
print("🚀 Extracting Marwari paragraphs...")
crawl(SEED_URL)
driver.quit()

# ----------------------
# SAVE OUTPUT
# ----------------------
unique_clean = list(dict.fromkeys(clean_paragraphs))
print(f"\n✅ Found {len(unique_clean)} Marwari paragraphs.")

with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Marwari_Paragraph"])
    for p in unique_clean:
        writer.writerow([p])

print(f"📁 Saved to {OUTPUT_FILE}")
