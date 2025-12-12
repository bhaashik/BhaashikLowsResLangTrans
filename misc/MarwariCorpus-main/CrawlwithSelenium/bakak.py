from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import os

# ====== CONFIGURATION ======
START_URL = "http://gadyakosh.org/gk/%E0%A4%B8%E0%A4%A4%E0%A5%8B%E0%A4%B3%E0%A4%BF%E0%A4%AF%E0%A5%8B_/_%E0%A4%B9%E0%A4%B0%E0%A5%80%E0%A4%B6_%E0%A4%AC%E0%A5%80._%E0%A4%B6%E0%A4%B0%E0%A5%8D%E0%A4%AE%E0%A4%BE"
OUTPUT_FILE = "gadyakosh_harish_sharma.txt"
# ============================

# ✅ Setup Chrome options
options = Options()
options.add_argument("--headless")  # run invisibly
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--log-level=3")

# ✅ Launch browser automatically
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

print(f"🌐 Opening main page: {START_URL}")
driver.get(START_URL)
time.sleep(3)

# ✅ Parse main page
soup = BeautifulSoup(driver.page_source, "html.parser")
body = soup.find("div", {"id": "bodyContent"})

# ✅ Collect all /gk/ internal links
links = []
if body:
    for a in body.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/gk/"):
            full_url = "http://gadyakosh.org" + href
            links.append(full_url)

links = sorted(list(set(links)))
print(f"✅ Found {len(links)} sub-pages to crawl.\n")

# ✅ Visit each link and extract text
poems_data = []
for i, link in enumerate(links, 1):
    print(f"[{i}/{len(links)}] Extracting: {link}")
    driver.get(link)
    time.sleep(2)

    sub_soup = BeautifulSoup(driver.page_source, "html.parser")
    content_div = sub_soup.find("div", {"id": "bodyContent"})

    if not content_div:
        print("❌ No content found.")
        continue

    paragraphs = [p.get_text(" ", strip=True) for p in content_div.find_all("p") if p.get_text(strip=True)]
    final_text = "\n".join(paragraphs)

    if not final_text or "फ़िलहाल इस पृष्ठ पर कोई सामग्री नहीं है" in final_text:
        print("⚠️ No valid content.")
        continue

    poems_data.append((link, final_text))
    print(f"✅ Extracted {len(final_text)} characters")

# ✅ Save all extracted text to file
if poems_data:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for link, text in poems_data:
            f.write(f"URL: {link}\n{text}\n\n{'='*80}\n\n")
    print(f"\n🎉 All extracted text saved to {OUTPUT_FILE}")
else:
    print("\n⚠️ No valid content found on any page.")

driver.quit()
print("🚀 Done!")
