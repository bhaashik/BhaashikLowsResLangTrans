from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
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
MAX_PAGES = 10
OUTPUT_FILE = "marwari_paragraphs.csv"

# Regex for Devanagari text blocks (includes punctuation and spaces)
DEVNAGARI_PARAGRAPH_PATTERN = re.compile(r'[\u0900-\u097F\s।.,!?;:\-–—()]+')

# ----------------------
# SELENIUM SETUP
# ----------------------
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ----------------------
# STORAGE
# ----------------------
visited = set()
paragraphs = []

# ----------------------
# HELPERS
# ----------------------
def extract_devanagari_paragraphs(text):
    """Return a list of Devanagari text blocks (paragraphs)."""
    blocks = DEVNAGARI_PARAGRAPH_PATTERN.findall(text)
    clean_blocks = []
    for b in blocks:
        b = b.strip()
        if len(b) > 20:  # only keep full sentences / paragraphs
            clean_blocks.append(b)
    return clean_blocks

def get_internal_links(soup, base_url):
    """Extract internal links from page."""
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
def crawl(url, depth=0):
    if len(visited) >= MAX_PAGES:
        return
    if url in visited:
        return

    try:
        driver.get(url)
        time.sleep(1.5)
        visited.add(url)
        print(f"[{len(visited)}] Crawling: {url}")

        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        text = soup.get_text(separator="\n")
        blocks = extract_devanagari_paragraphs(text)
        paragraphs.extend(blocks)

        links = get_internal_links(soup, SEED_URL)
        for link in links:
            if link not in visited:
                crawl(link, depth + 1)

    except Exception as e:
        print(f"Error accessing {url}: {e}")

# ----------------------
# RUN
# ----------------------
print("🚀 Starting paragraph extraction ...")
crawl(SEED_URL)
driver.quit()

# ----------------------
# SAVE RESULTS
# ----------------------
unique_paragraphs = list(dict.fromkeys(paragraphs))  # remove duplicates
print(f"\n✅ Extracted {len(unique_paragraphs)} Marwari paragraphs.")

with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Paragraph"])
    for p in unique_paragraphs:
        writer.writerow([p])

print(f"📁 Saved to {OUTPUT_FILE}")
