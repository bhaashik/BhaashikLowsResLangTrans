from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import os

# ====== CONFIGURATION ======
START_URL = "http://kavitakosh.org/kk/%E0%A4%B9%E0%A4%B0%E0%A4%BF_%E0%A4%B6%E0%A4%82%E0%A4%95%E0%A4%B0_%E0%A4%86%E0%A4%9A%E0%A4%BE%E0%A4%B0%E0%A5%8D%E0%A4%AF"
OUTPUT_FILE = "1.txt"
# ============================

# Setup Chrome options
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--log-level=3")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

print(f"🌐 Opening main page: {START_URL}")
driver.get(START_URL)
time.sleep(3)

soup = BeautifulSoup(driver.page_source, "html.parser")
body = soup.find("div", {"id": "bodyContent"}) or soup.find("div", {"class": "content"})  # fallback if structure differs

# Collect all internal links
links = []
if body:
    for a in body.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/kk/"):
            full_url = "http://kavitakosh.org" + href
            links.append(full_url)

links = sorted(list(set(links)))
print(f"✅ Found {len(links)} sub-pages to crawl.\n")

works_data = []
for i, link in enumerate(links, 1):
    print(f"[{i}/{len(links)}] Extracting: {link}")
    driver.get(link)
    time.sleep(2)

    sub_soup = BeautifulSoup(driver.page_source, "html.parser")
    content_div = sub_soup.find("div", {"id": "bodyContent"}) or sub_soup.find("div", {"class": "content"})

    if not content_div:
        print("❌ No content found.")
        continue

    paragraphs = [p.get_text(" ", strip=True) for p in content_div.find_all("p") if p.get_text(strip=True)]
    final_text = "\n".join(paragraphs)

    if not final_text:
        print("⚠️ No valid content.")
        continue

    works_data.append((link, final_text))
    print(f"✅ Extracted {len(final_text)} characters")

# Save results
if works_data:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for link, text in works_data:
            f.write(f"URL: {link}\n{text}\n\n{'='*80}\n\n")
    print(f"\n🎉 All extracted text saved to {OUTPUT_FILE}")
else:
    print("\n⚠️ No valid content found on any page.")

driver.quit()
print("🚀 Done!")
