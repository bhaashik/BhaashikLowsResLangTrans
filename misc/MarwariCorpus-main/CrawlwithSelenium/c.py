from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import re
import csv
import time
from urllib.parse import urljoin, urlparse

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
SEED_URLS = [
    "https://en.wikipedia.org/wiki/Marwari_language",
    "https://www.marwari-literature.com/en",
    "https://rajasthani1.blogspot.com/",
    "https://www.rajsabadkosh.org/sabadkosh-rajasthani-bhasha.aspx",
    "https://hindi.rajras.in/rajasthan/sanskriti/bhasha-va-boliyan/",
    "https://learnrajasthani.wordpress.com/",
    "https://www.marwaripathshala.com/introduction-to-marwari-language-and-script"
]

MAX_PAGES_PER_SITE = 5  # crawl limit per domain
OUTPUT_FILE = "marwari_paragraphs_multi.csv"

# Regex for Devanagari paragraphs (includes punctuation/spaces)
DEVNAGARI_BLOCK_PATTERN = re.compile(r'[\u0900-\u097F\s।,;:–—"\'\-()]+')

# -------------------------------------------------------
# SELENIUM SETUP
# -------------------------------------------------------
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# -------------------------------------------------------
# STORAGE
# -------------------------------------------------------
visited = set()
paragraphs = []  # stores tuples: (url, paragraph)

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
def extract_devanagari_paragraphs(text):
    """Extract large blocks of Devanagari text (sentences/paragraphs)."""
    blocks = DEVNAGARI_BLOCK_PATTERN.findall(text)
    clean_blocks = []
    for b in blocks:
        b = b.strip()
        if len(b) > 30:  # only keep full sentences / paragraphs
            clean_blocks.append(b)
    return clean_blocks

def get_internal_links(soup, base_url):
    """Return internal links for the same domain."""
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        abs_href = urljoin(base_url, href)
        if urlparse(abs_href).netloc == urlparse(base_url).netloc:
            links.add(abs_href)
    return links

# -------------------------------------------------------
# MAIN CRAWLER FUNCTION
# -------------------------------------------------------
def crawl(url, max_pages=MAX_PAGES_PER_SITE):
    site_domain = urlparse(url).netloc
    site_visited = set()
    to_visit = [url]

    while to_visit and len(site_visited) < max_pages:
        current = to_visit.pop(0)
        if current in visited:
            continue

        try:
            driver.get(current)
            time.sleep(2)
            visited.add(current)
            site_visited.add(current)
            print(f"[{len(visited)}] Crawling: {current}")

            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")

            # Extract text from <p> and <div>
            for tag in soup.find_all(["p", "div"]):
                text = tag.get_text(separator=" ").strip()
                if not text:
                    continue
                blocks = extract_devanagari_paragraphs(text)
                for b in blocks:
                    paragraphs.append((current, b))

            # Get internal links for further crawl
            for link in get_internal_links(soup, url):
                if link not in site_visited and len(site_visited) < max_pages:
                    to_visit.append(link)

        except Exception as e:
            print(f"Error on {current}: {e}")

# -------------------------------------------------------
# RUN CRAWLER
# -------------------------------------------------------
print("🚀 Starting multi-site Marwari paragraph extraction ...")
for seed in SEED_URLS:
    print(f"\n🌐 Crawling site: {seed}")
    crawl(seed, MAX_PAGES_PER_SITE)

driver.quit()

# -------------------------------------------------------
# SAVE RESULTS
# -------------------------------------------------------
# Remove duplicates
unique = list({(url, para): None for url, para in paragraphs}.keys())
print(f"\n✅ Extracted {len(unique)} paragraphs from {len(SEED_URLS)} sites.")

with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Source_URL", "Paragraph"])
    for url, para in unique:
        writer.writerow([url, para])

print(f"📁 Saved all results to {OUTPUT_FILE}")
