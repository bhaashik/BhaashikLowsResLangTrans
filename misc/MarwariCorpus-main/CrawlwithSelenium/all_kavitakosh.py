from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import os

# ---------------- CONFIG ----------------
BASE_URL = "http://kavitakosh.org"
START_URL = "http://kavitakosh.org/kk/%E0%A4%B0%E0%A4%BE%E0%A4%9C%E0%A4%B8%E0%A5%8D%E0%A4%A5%E0%A4%BE%E0%A4%A8%E0%A5%80"
OUTPUT_FILE = "rajasthani_poems.txt"
# ----------------------------------------

# ✅ Setup Chrome (headless)
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

print(f"🌐 Opening: {START_URL}")
driver.get(START_URL)
time.sleep(3)

# ✅ Parse author list
soup = BeautifulSoup(driver.page_source, "html.parser")
body = soup.find("div", {"id": "bodyContent"}) or soup.find("div", {"class": "content"})

authors = []
if body:
    for a in body.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/kk/") and not any(x in href for x in ["Category", "index.php"]):
            authors.append(BASE_URL + href)

authors = sorted(list(set(authors)))
print(f"✅ Found {len(authors)} author pages.")

poems = []

# ✅ Crawl each author and their poems
for idx, author_url in enumerate(authors, 1):
    print(f"\n👤 [{idx}/{len(authors)}] Author page: {author_url}")
    driver.get(author_url)
    time.sleep(2)

    asoup = BeautifulSoup(driver.page_source, "html.parser")
    abody = asoup.find("div", {"id": "bodyContent"}) or asoup.find("div", {"class": "content"})

    poem_links = []
    if abody:
        for a in abody.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/kk/") and not any(x in href for x in ["Category", "index.php"]):
                poem_links.append(BASE_URL + href)

    poem_links = sorted(list(set(poem_links)))

    # ✅ Extract poem text
    for poem_url in poem_links:
        driver.get(poem_url)
        time.sleep(1)
        psoup = BeautifulSoup(driver.page_source, "html.parser")
        content = psoup.find("div", {"id": "mw-content-text"}) or psoup.find("div", {"id": "bodyContent"})
        if not content:
            continue

        # Extract only paragraphs and poem lines
        poem_texts = []
        for p in content.find_all(["p", "div"]):
            text = p.get_text("\n", strip=True)
            if text and len(text) > 20 and not text.startswith("यह लेख"):
                poem_texts.append(text)

        if poem_texts:
            poem = "\n".join(poem_texts)
            poems.append(poem)
            print(f"📝 Poem extracted ({len(poem)} chars)")

# ✅ Save only poem text — nothing else
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for poem in poems:
        f.write(poem.strip() + "\n\n" + "="*80 + "\n\n")

driver.quit()

print(f"\n🎉 Extracted {len(poems)} poems.")
print(f"💾 Saved to: {OUTPUT_FILE}")
