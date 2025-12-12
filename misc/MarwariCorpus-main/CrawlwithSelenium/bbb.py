from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import os

# ====== CONFIGURATION ======
LINKS = [
    "http://gadyakosh.org/gk/%E0%A4%A0%E0%A4%B9%E0%A4%B0%E0%A4%BE_%E0%A4%B9%E0%A5%81%E0%A4%86_%E0%A4%B8%E0%A4%AE%E0%A4%AF_/_%E0%A4%B8%E0%A4%BE%E0%A4%82%E0%A4%B5%E0%A4%B0_%E0%A4%A6%E0%A4%87%E0%A4%AF%E0%A4%BE",
    "http://gadyakosh.org/gk/%E0%A4%85%E0%A4%B0%E0%A5%87,_%E0%A4%87%E0%A4%A4%E0%A4%A8%E0%A4%BE_%E0%A4%A6%E0%A5%81%E0%A4%83%E0%A4%96_!_/_%E0%A4%B8%E0%A4%BE%E0%A4%82%E0%A4%B5%E0%A4%B0_%E0%A4%A6%E0%A4%87%E0%A4%AF%E0%A4%BE",
    "http://gadyakosh.org/gk/%E0%A4%A7%E0%A4%B0%E0%A4%A4%E0%A5%80_%E0%A4%95%E0%A4%AC_%E0%A4%A4%E0%A4%95_%E0%A4%98%E0%A5%82%E0%A4%AE%E0%A5%87%E0%A4%97%E0%A5%80_/_%E0%A4%B8%E0%A4%BE%E0%A4%82%E0%A4%B5%E0%A4%B0_%E0%A4%A6%E0%A4%87%E0%A4%AF%E0%A4%BE"
]
OUTPUT_FILE = "gadyakosh_sanwar_daiya.txt"
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

print(f"🟢 Starting extraction for {len(LINKS)} pages...\n")

poems_data = []

for i, link in enumerate(LINKS, 1):
    print(f"[{i}/{len(LINKS)}] Extracting: {link}")
    driver.get(link)
    time.sleep(2)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    content_div = soup.find("div", {"id": "bodyContent"})

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

# ✅ Save results to text file
if poems_data:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for link, text in poems_data:
            f.write(f"URL: {link}\n{text}\n\n{'='*80}\n\n")
    print(f"\n🎉 All extracted text saved to {OUTPUT_FILE}")
else:
    print("\n⚠️ No valid content found on any page.")

driver.quit()
print("🚀 Done!")
