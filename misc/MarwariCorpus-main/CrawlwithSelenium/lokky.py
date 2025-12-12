from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import os

# ====== CONFIGURATION ======
START_URL = "http://gadyakosh.org/gk/%E0%A4%A7%E0%A4%BE%E0%A4%A8_%E0%A4%95%E0%A4%A5%E0%A4%BE%E0%A4%B5%E0%A4%BE%E0%A4%82_/_%E0%A4%B8%E0%A4%A4%E0%A5%8D%E0%A4%AF%E0%A4%A8%E0%A4%BE%E0%A4%B0%E0%A4%BE%E0%A4%AF%E0%A4%A3_%E0%A4%B8%E0%A5%8B%E0%A4%A8%E0%A5%80"
OUTPUT_FILE = "gadyakosh_output.txt"
# ============================

# ✅ Setup Chrome options
options = Options()
options.add_argument("--headless")  # run without showing browser window
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--log-level=3")  # suppress Selenium logs

# ✅ Automatically install the correct ChromeDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

print(f"🌐 Opening main page: {START_URL}")
driver.get(START_URL)
time.sleep(3)

# ✅ Parse the main page
soup = BeautifulSoup(driver.page_source, "html.parser")
body = soup.find("div", {"id": "bodyContent"})

# ✅ Collect all sub-links
links = []
if body:
    for a in body.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/gk/"):
            full_url = "http://gadyakosh.org" + href
            links.append(full_url)

links = sorted(list(set(links)))
print(f"✅ Found {len(links)} sub-pages.")

# ✅ Visit each link and extract content
poems_data = []
for i, link in enumerate(links, 1):
    print(f"\n[{i}/{len(links)}] Extracting: {link}")
    driver.get(link)
    time.sleep(2)

    sub_soup = BeautifulSoup(driver.page_source, "html.parser")
    content_div = sub_soup.find("div", {"id": "bodyContent"})

    if not content_div:
        print("❌ No body content found.")
        continue

    paragraphs = [p.get_text(" ", strip=True) for p in content_div.find_all("p") if p.get_text(strip=True)]
    final_text = "\n".join(paragraphs)

    # ✅ Skip empty pages or default placeholder text
    if not final_text or "फ़िलहाल इस पृष्ठ पर कोई सामग्री नहीं है" in final_text:
        print("⚠️ No valid content.")
        continue

    poems_data.append((link, final_text))
    print(f"✅ Extracted {len(final_text)} characters")

# ✅ Save all results to a text file
if poems_data:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for link, text in poems_data:
            f.write(f"URL: {link}\n{text}\n\n{'='*80}\n\n")
    print(f"\n🎉 All extracted text saved to {OUTPUT_FILE}")
else:
    print("\n⚠️ No content found on any sub-page.")

driver.quit()
print("🚀 Done!")
