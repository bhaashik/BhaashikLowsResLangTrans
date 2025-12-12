from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import re
import time

BASE_URL = "http://kavitakosh.org"
START_URL = "http://kavitakosh.org/kk/%E0%A4%B0%E0%A4%BE%E0%A4%9C%E0%A4%B8%E0%A5%8D%E0%A4%A5%E0%A4%BE%E0%A4%A8%E0%A5%80"
OUTPUT_FILE = "marwari_poems_only.txt"

# ✅ Headless Chrome setup
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

print(f"🌐 Opening Rajasthani section: {START_URL}")
driver.get(START_URL)
time.sleep(3)

# ✅ Parse main authors page
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

all_poems = []

# ✅ Go through each author and extract poem text
for idx, author_url in enumerate(authors, 1):
    print(f"\n👤 [{idx}/{len(authors)}] Visiting author page: {author_url}")
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

    for poem_url in poem_links:
        driver.get(poem_url)
        time.sleep(1)
        psoup = BeautifulSoup(driver.page_source, "html.parser")
        content = psoup.find("div", {"id": "mw-content-text"}) or psoup.find("div", {"id": "bodyContent"})
        if not content:
            continue

        # Extract only poem text
        text = content.get_text("\n", strip=True)

        # Remove site junk, numbers, English, punctuation
        text = re.sub(r"[A-Za-z0-9]", "", text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^ \u0900-\u097F\n]", " ", text)
        text = re.sub(r"\s+", " ", text)

        # Skip empty or non-poem pages
        if len(text.strip()) < 30:
            continue

        all_poems.append(text.strip())
        print(f"📝 Extracted poem ({len(text)} chars)")

# ✅ Save all Marwari poems
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for poem in all_poems:
        f.write(poem.strip() + "\n\n")

driver.quit()

print(f"\n🎉 Extracted {len(all_poems)} poems.")
print(f"💾 Saved clean Marwari-only poems to: {OUTPUT_FILE}")
