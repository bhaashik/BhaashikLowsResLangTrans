import csv
import hashlib
import os
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

# Additional for PDF extraction
import pdfplumber  # You need to install this: pip install pdfplumber

SEED_URLS = [
    "https://www.aapanorajasthan.org",
]
OUTPUT_FILE = "ma2wari6.csv"
MAX_PAGES_TOTAL = 1000
MAX_PAGES_PER_DOMAIN = 200
REQUEST_TIMEOUT = 15
REQUEST_DELAY_SEC = 1.0

ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "application/pdf")
EXCLUDE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp",
                ".zip", ".rar", ".7z", ".gz", ".tar", ".mp3", ".mp4",
                ".avi", ".mkv", ".webm", ".css", ".js", ".ico", ".woff",
                ".woff2", ".ttf", ".otf")

# HTTP Session
session = requests.Session()
session.headers.update({"User-Agent": "MarwariCrawler/1.3 (+contact: youremail@example.com)"})


# Helpers

def normalize_url(url: str) -> str:
    url, _frag = urldefrag(url)
    parsed = urlparse(url)
    norm = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
    ).geturl()
    return norm


def looks_like_file(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in EXCLUDE_EXTS)


def is_html_like(resp: requests.Response) -> bool:
    ct = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
    return any(ct.startswith(t) for t in ALLOWED_CONTENT_TYPES if t != "application/pdf")


def is_pdf_like(resp: requests.Response) -> bool:
    ct = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
    return ct == "application/pdf"


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")

def is_devanagari_text(text: str, min_dev_chars: int = 80, min_ratio: float = 0.3) -> bool:
    dev_chars = DEVANAGARI_RE.findall(text)
    dev_count = sum(len(x) for x in dev_chars)
    ratio = dev_count / max(len(text), 1)
    return dev_count >= min_dev_chars and ratio >= min_ratio

def split_sentences(text: str) -> list[str]:
    # Basic split by danda or punctuation marks
    sentences = re.split(r'[।?!\n]+', text)
    return [s.strip() for s in sentences if s.strip()]

def is_marwari_sentence(sentence: str) -> bool:
    # Check if sentence contains mostly Devanagari characters and minimal latin letters
    dev_chars = DEVANAGARI_RE.findall(sentence)
    dev_count = sum(len(x) for x in dev_chars)
    if dev_count < 5:
        return False
    latin_count = len(re.findall(r'[A-Za-z]', sentence))
    # Return true if Devanagari chars > latin chars
    return dev_count > latin_count

def extract_marwari_sentences(text: str) -> str:
    sentences = split_sentences(text)
    marwari_sentences = [s for s in sentences if is_marwari_sentence(s)]
    return "। ".join(marwari_sentences)  # Join with danda


def clean_soup(soup: BeautifulSoup):
    for tag in soup(["script", "style", "noscript", "header", "footer", "aside", "nav"]):
        tag.decompose()


def extract_main_text(url: str, html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    clean_soup(soup)
    title = soup.title.get_text(strip=True) if soup.title else ""

    # Wikimedia/Wikipedia-specific
    if "wikimedia.org" in url or "wikipedia.org" in url:
        main = soup.find("div", {"class": "mw-parser-output"})
        if main:
            ps = main.find_all("p")
            text = " ".join(p.get_text(" ", strip=True) for p in ps)
            text = re.sub(r"\[\d+\]", "", text)
            return title, text

    # Fallback
    ps = soup.find_all("p")
    text = " ".join(p.get_text(" ", strip=True) for p in ps)
    return title, text


def extract_text_from_pdf(content: bytes) -> str:
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"[error] PDF extraction failed: {e}")
    return text


def crawl():
    to_visit = deque([normalize_url(u) for u in SEED_URLS])
    visited = set()
    per_domain_counts = {}
    seen_hashes = set()
    allowed_domains = {urlparse(u).netloc for u in SEED_URLS}

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "title", "text"])
        writer.writeheader()

    total_saved = 0

    while to_visit and len(visited) < MAX_PAGES_TOTAL:
        current = to_visit.popleft()
        if current in visited:
            continue
        visited.add(current)

        cur_domain = urlparse(current).netloc
        if cur_domain not in allowed_domains:
            continue

        per_domain_counts.setdefault(cur_domain, 0)
        if per_domain_counts[cur_domain] >= MAX_PAGES_PER_DOMAIN:
            continue

        if looks_like_file(current) and not current.endswith(".pdf"):
            continue

        try:
            resp = session.get(current, timeout=REQUEST_TIMEOUT)
        except Exception as e:
            print(f"[error] GET fail: {current} :: {e}")
            continue

        if resp.status_code != 200:
            print(f"[http {resp.status_code}] Skip: {current}")
            continue

        # PDF processing
        if is_pdf_like(resp) or current.endswith(".pdf"):
            pdf_text = extract_text_from_pdf(resp.content)
            if is_devanagari_text(pdf_text):
                marwari_text = extract_marwari_sentences(pdf_text)
                if len(marwari_text) > 100:  # Threshold for meaningful text
                    h = hashlib.sha256(marwari_text.encode("utf-8")).hexdigest()
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=["url", "title", "text"])
                            writer.writerow({"url": current, "title": "", "text": marwari_text})
                        total_saved += 1
                        per_domain_counts[cur_domain] += 1
                        print(f"[saved PDF] ({total_saved}) {current}")
            time.sleep(REQUEST_DELAY_SEC)
            continue

        # HTML processing
        if not is_html_like(resp):
            continue

        title, text = extract_main_text(current, resp.text)
        if text and is_devanagari_text(text):
            marwari_text = extract_marwari_sentences(text)
            if len(marwari_text) > 100:
                h = hashlib.sha256(marwari_text.encode("utf-8")).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=["url", "title", "text"])
                        writer.writerow({"url": current, "title": title, "text": marwari_text})
                    total_saved += 1
                    per_domain_counts[cur_domain] += 1
                    print(f"[saved] ({total_saved}) {current}")

        # Extract and filter links
        try:
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                link = urljoin(current, a["href"])
                link = normalize_url(link)
                # Skip malformed concatenated URLs
                if "http" in link[5:]:
                    continue
                # Only crawl links within allowed domains
                domain = urlparse(link).netloc
                if domain not in allowed_domains:
                    continue
                if looks_like_file(link) and not link.endswith(".pdf"):
                    continue
                if link not in visited and link not in to_visit:
                    to_visit.append(link)
        except Exception as e:
            print(f"[error] Link extraction failed: {current} :: {e}")

        time.sleep(REQUEST_DELAY_SEC)

    print(f"✅ Done. Saved {total_saved} pages to {OUTPUT_FILE}")


if __name__ == "__main__":
    import io  # Needed for pdfplumber BytesIO wrapper
    crawl()
