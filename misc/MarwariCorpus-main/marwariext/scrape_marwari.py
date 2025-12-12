import os
import asyncio
import csv
import re
from urllib.parse import urljoin, urlparse

from zenrows import ZenRowsClient
from bs4 import BeautifulSoup

API_KEY = 'b25216417b4ac3a6fcb62fd004524791607e7c70'
client = ZenRowsClient(API_KEY, concurrency=5, retries=2)

BASE_URL = "https://www.aapanorajasthan.org"
DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

visited_urls = set()

def is_internal_link(href):
    if not href or href.startswith("#"):
        return False
    parsed = urlparse(href)
    return not parsed.netloc or parsed.netloc == urlparse(BASE_URL).netloc

def extract_internal_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if is_internal_link(href):
            full_url = urljoin(base_url, href)
            if full_url.startswith(BASE_URL):
                links.add(full_url)
    return links

def extract_marwari_text(html):
    soup = BeautifulSoup(html, "html.parser")
    texts = soup.stripped_strings
    marwari_lines = []
    for line in texts:
        if DEVANAGARI_RE.search(line) and not re.search(r'[A-Za-z]', line):
            marwari_lines.append(line)
    return "\n".join(marwari_lines)

async def fetch_url(url):
    if url in visited_urls:
        return None
    visited_urls.add(url)
    try:
        params = {
            "url": url,
            "js_render": "false",
            "premium_proxy": "false"
        }
        resp = await client.get_async(url, params=params)
        return url, resp.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

async def crawl_site(start_url):
    to_visit = {start_url}
    results = []

    while to_visit:
        current_batch = list(to_visit)[:10]
        to_visit -= set(current_batch)

        tasks = [fetch_url(url) for url in current_batch]
        responses = await asyncio.gather(*tasks)

        for result in responses:
            if result:
                url, html = result
                text = extract_marwari_text(html)
                if text.strip():
                    results.append((url, text))
                # Add new internal links to queue
                new_links = extract_internal_links(html, url)
                to_visit.update(new_links - visited_urls)

    return results

async def main():
    print("Crawling and extracting...")
    data = await crawl_site(BASE_URL)

    with open("aapanorajasthan_marwari.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "marwari_text"])
        for url, text in data:
            writer.writerow([url, text])

    print(f"Done! Extracted {len(data)} pages to 'aapanorajasthan_marwari.csv'")

if __name__ == "__main__":
    asyncio.run(main())
