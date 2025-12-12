"""
crawl_marwari_sites.py

Crawls multiple seed websites and extracts clean Marwari (Devanagari-script) paragraphs.
Requires: selenium, webdriver-manager, beautifulsoup4
Install: pip install selenium webdriver-manager beautifulsoup4
"""

import time
import re
import csv
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# ---------------------
# CONFIGURATION
# ---------------------
SEED_URLS = [
    "https://www.aapanorajasthan.org/aapani%20bhasa.php",
    "https://www.marwaripathshala.com/introduction-to-marwari-language-and-script",
    "https://www.marwari-literature.com/en",
    "https://1iwd.com/marwadi-bhasha/",
    "https://marwaribaatein.com/marwari-language/",
    "https://hattai.page.tl/marwari-origin-etc.htm",
    "https://www.sidhimarwadi.com/home"
]

OUTPUT_CSV = "marwari_pgfs_all_sites.csv"
MAX_PAGES_PER_SITE = 10        # max pages to follow per domain (keeps crawl bounded)
SLEEP_BETWEEN_REQUESTS = 1.2   # polite delay
HEADLESS = True

# ---------------------
# PATTERNS & FILTERS
# ---------------------
# A block-match pattern (will capture contiguous Devanagari + punctuation + spaces)
DEVANAGARI_BLOCK_PATTERN = re.compile(r'[।\u0900-\u097F0-9\s,.;:!?«»“”"\'\-–—()]+')

# Minimum length of paragraph to keep (characters)
MIN_PARAGRAPH_LEN = 30

# Minimum fraction of characters that must be Devanagari
MIN_DEVANAGARI_RATIO = 0.80

# Small list of Marwari / Rajasthani specific tokens to prefer (heuristic)
# You can expand this list as you collect more genuine Marwari words
MARWARI_TOKENS = {
    "थारो", "थारी", "म्हारो", "म्हारी", "म्हाने", "हुवै", "हुवैजी", "बाई", "घर", "रै",
    "कैल", "खम्मा", "सुणो", "किण", "कु", "जेठा", "बाट", "रैव", "रामदोहाई", "हुवै"
}

# Common Hindi words that, if dominating text, likely indicate generic Hindi (filter heuristic)
COMMON_HINDI_WORDS = {
    "है", "हैं", "और", "से", "के", "का", "की", "को", "पर", "में",
    "यह", "कि", "तो", "ही", "था", "थे", "एक", "नहीं", "जो", "ने", "भी"
}

# ---------------------
# SELENIUM SETUP
# ---------------------
options = Options()
if HEADLESS:
    options.add_argument("--headless=new")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--lang=hi-IN")  # hint to servers to serve Hindi/Devanagari pages

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ---------------------
# UTIL FUNCTIONS
# ---------------------
def is_same_domain(base, url):
    return urlparse(base).netloc == urlparse(url).netloc

def clean_whitespace(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def devanagari_ratio(s: str) -> float:
    if not s:
        return 0.0
    dev_count = sum(1 for ch in s if '\u0900' <= ch <= '\u097F')
    return dev_count / len(s)

def contains_marwari_token(s: str) -> bool:
    # lowercase-ish: tokens in Devanagari (no case folding)
    words = re.findall(r'[\u0900-\u097F]+', s)
    return any(w in MARWARI_TOKENS for w in words)

def likely_hindi_dominant(s: str) -> bool:
    words = [w for w in re.findall(r'[\u0900-\u097F]+', s)]
    if not words:
        return False
    count_common = sum(1 for w in words if w in COMMON_HINDI_WORDS)
    return (count_common / len(words)) > 0.6

def extract_paragraph_candidates(soup: BeautifulSoup):
    # focus on visible content tags, in order of content likeliness
    tags = []
    tags.extend(soup.find_all("article"))
    tags.extend(soup.find_all(["p", "div", "span", "h1", "h2", "h3", "h4", "h5"]))
    seen = set()
    texts = []
    for t in tags:
        # avoid duplicates
        try:
            txt = t.get_text(separator=" ").strip()
        except Exception:
            continue
        txt = clean_whitespace(txt)
        if not txt or txt in seen:
            continue
        seen.add(txt)
        texts.append(txt)
    return texts

def filter_devanagari_paragraphs(texts):
    kept = []
    for t in texts:
        # extract contiguous devanagari-containing blocks (the pattern may return many small fragments)
        blocks = DEVANAGARI_BLOCK_PATTERN.findall(t)
        for b in blocks:
            b = clean_whitespace(b)
            if len(b) < MIN_PARAGRAPH_LEN:
                continue
            ratio = devanagari_ratio(b)
            if ratio < MIN_DEVANAGARI_RATIO:
                continue
            # heuristic: reject if overly Hindi-dominant unless contains marwari token
            if likely_hindi_dominant(b) and not contains_marwari_token(b):
                continue
            # finally, prefer paragraphs that contain at least one marwari token OR pass general checks
            # (we keep both; token presence increases precision)
            kept.append(b)
    return kept

# ---------------------
# CRAWLING LOGIC
# ---------------------
import collections

def crawl_site(seed_url, max_pages=MAX_PAGES_PER_SITE):
    to_visit = collections.deque([seed_url])
    visited = set()
    results = []  # list of tuples (source_url, paragraph)
    pages_visited = 0

    while to_visit and pages_visited < max_pages:
        url = to_visit.popleft()
        if url in visited:
            continue
        try:
            print(f"[{pages_visited+1}/{max_pages}] Visiting: {url}")
            driver.get(url)
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")

            # extract and filter paragraph candidates
            candidates = extract_paragraph_candidates(soup)
            keep = filter_devanagari_paragraphs(candidates)
            for p in keep:
                results.append((url, p))

            # collect internal links to queue
            # only simple a[href] same-domain HTTP(S) links
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href:
                    continue
                # build absolute
                abs_href = urljoin(url, href)
                parsed = urlparse(abs_href)
                if parsed.scheme not in ("http", "https"):
                    continue
                if is_same_domain(seed_url, abs_href) and abs_href not in visited:
                    to_visit.append(abs_href)

            visited.add(url)
            pages_visited += 1
        except Exception as e:
            print("  ❌ Error visiting", url, ":", e)
            visited.add(url)
            pages_visited += 1
    return results

# ---------------------
# RUN FOR ALL SEEDS
# ---------------------
all_results = []
seen_paragraphs = set()  # dedup by paragraph text

for seed in SEED_URLS:
    try:
        print("\n" + "="*60)
        print("Starting site:", seed)
        site_results = crawl_site(seed, max_pages=MAX_PAGES_PER_SITE)
        # deduplicate and append
        for src, para in site_results:
            if para in seen_paragraphs:
                continue
            seen_paragraphs.add(para)
            all_results.append((src, para))
    except Exception as e:
        print("Failed to crawl seed:", seed, ":", e)

driver.quit()

# ---------------------
# SAVE OUTPUT CSV
# ---------------------
print(f"\nSaving {len(all_results)} unique paragraphs to {OUTPUT_CSV} ...")
with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["source_url", "marwari_paragraph"])
    for src, para in all_results:
        writer.writerow([src, para])

print("Done.")
