from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

# ---------------- Chrome Setup ----------------
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ---------------- All URLs You Provided ----------------
urls = [
    "http://gadyakosh.org/gk/%E0%A4%86%E0%A4%AB%E0%A4%B3_/_%E0%A4%AE%E0%A4%A6%E0%A4%A8_%E0%A4%97%E0%A5%8B%E0%A4%AA%E0%A4%BE%E0%A4%B2_%E0%A4%B2%E0%A4%A2%E0%A4%BC%E0%A4%BE",
    "http://gadyakosh.org/gk/%E0%A4%A6%E0%A5%8B%E0%A4%B2%E0%A4%A1%E0%A4%BC%E0%A5%80_%E0%A4%9C%E0%A5%82%E0%A4%A3_/_%E0%A4%AE%E0%A4%A6%E0%A4%A8_%E0%A4%97%E0%A5%8B%E0%A4%AA%E0%A4%BE%E0%A4%B2_%E0%A4%B2%E0%A4%A7%E0%A4%BC%E0%A4%BE",
    "http://gadyakosh.org/gk/%E0%A4%9B%E0%A4%BF%E0%A4%82%E0%A4%95%E0%A5%80_/_%E0%A4%B8%E0%A4%A4%E0%A5%8D%E0%A4%AF%E0%A4%A8%E0%A4%BE%E0%A4%B0%E0%A4%BE%E0%A4%AF%E0%A4%A3_%E0%A4%B8%E0%A5%8B%E0%A4%A8%E0%A5%80",
    "http://gadyakosh.org/gk/%E0%A4%9B%E0%A5%87%E0%A4%95%E0%A4%A1%E0%A4%BC%E0%A4%B2%E0%A5%80_%E0%A4%B8%E0%A4%BE%E0%A4%82%E0%A4%B8_/_%E0%A4%B8%E0%A4%A4%E0%A5%8D%E0%A4%AF%E0%A4%A8%E0%A4%BE%E0%A4%B0%E0%A4%BE%E0%A4%AF%E0%A4%A3_%E0%A4%B8%E0%A5%8B%E0%A4%A8%E0%A5%80",
    "http://gadyakosh.org/gk/%E0%A4%9B%E0%A5%8B%E0%A4%9F%E0%A5%80-%E0%A4%B8%E0%A5%80_%E0%A4%AC%E0%A4%BE%E0%A4%A4_/_%E0%A4%B8%E0%A4%A4%E0%A5%8D%E0%A4%AF%E0%A4%A8%E0%A4%BE%E0%A4%B0%E0%A4%BE%E0%A4%AF%E0%A4%A3_%E0%A4%B8%E0%A5%8B%E0%A4%A8%E0%A5%80",
    "http://gadyakosh.org/gk/%E0%A4%9C%E0%A4%B2%E0%A4%AE-%E0%A4%A6%E0%A4%BF%E0%A4%A8_/_%E0%A4%B8%E0%A4%A4%E0%A5%8D%E0%A4%AF%E0%A4%A8%E0%A4%BE%E0%A4%B0%E0%A4%BE%E0%A4%AF%E0%A4%A3_%E0%A4%B8%E0%A5%8B%E0%A4%A8%E0%A5%80",
    "http://gadyakosh.org/gk/%E0%A4%9C%E0%A5%80%E0%A4%B5%E0%A4%A3_%E0%A4%B0%E0%A5%8C_%E0%A4%9C%E0%A4%A5%E0%A4%BE%E0%A4%B0%E0%A4%A5_/_%E0%A4%95%E0%A4%A8%E0%A5%8D%E0%A4%B9%E0%A5%88%E0%A4%AF%E0%A4%BE%E0%A4%B2%E0%A4%BE%E0%A4%B2_%E0%A4%AD%E0%A4%BE%E0%A4%9F%E0%A5%80",
    "http://gadyakosh.org/gk/%E0%A4%9C%E0%A5%81%E0%A4%A6%E0%A5%8D%E0%A4%A7_/_%E0%A4%B0%E0%A4%BE%E0%A4%AE%E0%A4%B8%E0%A5%8D%E0%A4%B5%E0%A4%B0%E0%A5%82%E0%A4%AA_%E0%A4%95%E0%A4%BF%E0%A4%B8%E0%A4%BE%E0%A4%A8",
    "http://gadyakosh.org/gk/%E0%A4%A1%E0%A4%B0_/_%E0%A4%B0%E0%A4%BE%E0%A4%AE%E0%A4%B8%E0%A5%8D%E0%A4%B5%E0%A4%B0%E0%A5%82%E0%A4%AA_%E0%A4%95%E0%A4%BF%E0%A4%B8%E0%A4%BE%E0%A4%A8",
    "http://gadyakosh.org/gk/%E0%A4%A4%E0%A5%80%E0%A4%96%E0%A5%80_%E0%A4%A7%E0%A4%BE%E0%A4%B0.._/_%E0%A4%B0%E0%A4%BE%E0%A4%AE%E0%A4%B8%E0%A5%8D%E0%A4%B5%E0%A4%B0%E0%A5%82%E0%A4%AA_%E0%A4%95%E0%A4%BF%E0%A4%B8%E0%A4%BE%E0%A4%A8",
    "http://gadyakosh.org/gk/%E0%A4%A6%E0%A4%B2%E0%A4%BE%E0%A4%B2_/_%E0%A4%B0%E0%A4%BE%E0%A4%AE%E0%A4%B8%E0%A5%8D%E0%A4%B5%E0%A4%B0%E0%A5%82%E0%A4%AA_%E0%A4%95%E0%A4%BF%E0%A4%B8%E0%A4%BE%E0%A4%A8",
    "http://gadyakosh.org/gk/%E0%A4%A6%E0%A5%80%E0%A4%AA%E0%A4%B2%E0%A4%BE%E0%A4%A3%E0%A5%88_%E0%A4%B0%E0%A5%8B_%E0%A4%A6%E0%A4%BE%E0%A4%A4%E0%A4%BE%E0%A4%B0_/_%E0%A4%B0%E0%A4%BE%E0%A4%AE%E0%A4%B8%E0%A5%8D%E0%A4%B5%E0%A4%B0%E0%A5%82%E0%A4%AA_%E0%A4%95%E0%A4%BF%E0%A4%B8%E0%A4%BE%E0%A4%A8",
    "http://gadyakosh.org/gk/%E0%A4%A7%E0%A5%8B%E0%A4%B3%E0%A5%8B_%E0%A4%A6%E0%A4%BF%E0%A4%A8_/_%E0%A4%B0%E0%A4%BE%E0%A4%AE%E0%A4%B8%E0%A5%8D%E0%A4%B5%E0%A4%B0%E0%A5%82%E0%A4%AA_%E0%A4%95%E0%A4%BF%E0%A4%B8%E0%A4%BE%E0%A4%A8",
    "http://gadyakosh.org/gk/%E0%A4%A8%E0%A4%B5%E0%A5%80_%E0%A4%B0%E0%A5%80%E0%A4%A4_%E0%A4%A8%E0%A5%88_%E0%A4%AA%E0%A4%B0%E0%A5%8B%E0%A4%9F%E0%A4%A4%E0%A5%80_%E0%A4%85%E0%A4%B0_%E0%A4%85%E0%A4%82%E0%A4%97%E0%A5%87%E0%A4%9C%E0%A4%A4%E0%A5%80_%E0%A4%85%E0%A4%97%E0%A4%BE%E0%A4%A1%E0%A4%BC%E0%A5%80_%E0%A4%95%E0%A4%B9%E0%A4%BE%E0%A4%A3%E0%A4%BF%E0%A4%AF%E0%A4%BE%E0%A4%82_/_%E0%A4%A8%E0%A5%80%E0%A4%B0%E0%A4%9C_%E0%A4%A6%E0%A4%87%E0%A4%AF%E0%A4%BE",
    "http://gadyakosh.org/gk/%E0%A4%A8%E0%A4%BE%E0%A4%9C%E0%A4%95_/_%E0%A4%B8%E0%A4%A4%E0%A5%8D%E0%A4%AF%E0%A4%A8%E0%A4%BE%E0%A4%B0%E0%A4%BE%E0%A4%AF%E0%A4%A3_%E0%A4%B8%E0%A5%8B%E0%A4%A8%E0%A5%80",
    "http://gadyakosh.org/gk/%E0%A4%A8%E0%A5%8C%E0%A4%95%E0%A4%B0_/_%E0%A4%A8%E0%A5%80%E0%A4%B0%E0%A4%9C_%E0%A4%A6%E0%A4%87%E0%A4%AF%E0%A4%BE",
    "http://gadyakosh.org/gk/%E0%A4%AA%E0%A4%BE%E0%A4%82%E0%A4%97%E0%A4%B3%E0%A5%80_%E0%A4%B8%E0%A4%82%E0%A4%B5%E0%A5%87%E0%A4%A6%E0%A4%A8%E0%A4%BE_/_%E0%A4%B0%E0%A4%BE%E0%A4%AE%E0%A4%B8%E0%A5%8D%E0%A4%B5%E0%A4%B0%E0%A5%82%E0%A4%AA_%E0%A4%95%E0%A4%BF%E0%A4%B8%E0%A4%BE%E0%A4%A8",
    "http://gadyakosh.org/gk/%E0%A4%AA%E0%A5%82%E0%A4%81%E0%A4%9C%E0%A5%80_/_%E0%A4%B0%E0%A4%BE%E0%A4%AE%E0%A4%B8%E0%A5%8D%E0%A4%B5%E0%A4%B0%E0%A5%82%E0%A4%AA_%E0%A4%95%E0%A4%BF%E0%A4%B8%E0%A4%BE%E0%A4%A8",
    "http://gadyakosh.org/gk/%E0%A4%AC%E0%A4%BE_%E0%A4%AC%E0%A4%BE%E0%A4%A4_/_%E0%A4%B0%E0%A4%BE%E0%A4%AE%E0%A4%B8%E0%A5%8D%E0%A4%B5%E0%A4%B0%E0%A5%82%E0%A4%AA_%E0%A4%95%E0%A4%BF%E0%A4%B8%E0%A4%BE%E0%A4%A8"
]

# ---------------- Helper Function ----------------
def clean_text(text):
    text = re.sub(r"(—|–|-)?\s*(लेखक|कवि|रचनाकार|द्वारा|मदन गोपाल|रामस्वरूप किसान|नीरज|सत्यनारायण|भाटी)\s*[^\n]*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_poems(url):
    driver.get(url)
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    main_div = soup.find("div", {"id": "bodyContent"})
    poems = []
    if not main_div:
        return []

    links = [a["href"] for a in main_div.find_all("a", href=True) if a["href"].startswith("/gk/")]

    if not links:
        paragraphs = main_div.find_all(["p", "div"])
        combined = "\n".join([p.get_text(" ", strip=True) for p in paragraphs])
        cleaned = clean_text(combined)
        if cleaned and "फ़िलहाल इस पृष्ठ पर कोई सामग्री नहीं है" not in cleaned:
            poems.append({"url": url, "text": cleaned})
        return poems

    for link in links:
        full_url = "http://gadyakosh.org" + link
        driver.get(full_url)
        time.sleep(2)
        inner_soup = BeautifulSoup(driver.page_source, "html.parser")
        inner = inner_soup.find("div", {"id": "bodyContent"})
        if inner:
            paragraphs = inner.find_all(["p", "div"])
            combined = "\n".join([p.get_text(" ", strip=True) for p in paragraphs])
            cleaned = clean_text(combined)
            if cleaned and "फ़िलहाल इस पृष्ठ पर कोई सामग्री नहीं है" not in cleaned:
                poems.append({"url": full_url, "text": cleaned})
    return poems

# ---------------- Main ----------------
all_poems = []
for u in urls:
    print(f"🔹 Extracting: {u}")
    all_poems.extend(extract_poems(u))

# ---------------- Save Results ----------------
df = pd.DataFrame(all_poems)
df.to_csv("gadyakosh_all_poems.csv", index=False, encoding="utf-8-sig")

print("\n✅ All poems extracted successfully!")
print("📁 Saved as: gadyakosh_all_poems.csv")

driver.quit()
