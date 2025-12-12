from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import pandas as pd
import time

# -------- CONFIGURE CHROMEDRIVER PATH --------
# Update this path to your local chromedriver.exe location
chrome_driver_path = r"C:\chromedriver-win64\chromedriver.exe"

# -------- WEBSITE LINKS --------
urls = [
    "http://gadyakosh.org/gk/%E0%A4%AE%E0%A4%BE%E0%A4%82_/_%E0%A4%B5%E0%A4%BF%E0%A4%9C%E0%A4%AF%E0%A4%A6%E0%A4%BE%E0%A4%A8_%E0%A4%A6%E0%A5%87%E0%A4%A5%E0%A4%BE_%27%E0%A4%AC%E0%A4%BF%E0%A4%9C%E0%A5%8D%E2%80%8D%E0%A4%9C%E0%A5%80%27",
    "http://gadyakosh.org/gk/%E0%A4%86%E0%A4%AB%E0%A4%B3_/_%E0%A4%AE%E0%A4%A6%E0%A4%A8_%E0%A4%97%E0%A5%8B%E0%A4%AA%E0%A4%BE%E0%A4%B2_%E0%A4%B2%E0%A4%A2%E0%A4%BC%E0%A4%BE",
    "http://gadyakosh.org/gk/%E0%A4%A6%E0%A5%8B%E0%A4%B2%E0%A4%A1%E0%A4%BC%E0%A5%80_%E0%A4%9C%E0%A5%82%E0%A4%A3_/_%E0%A4%AE%E0%A4%A6%E0%A4%A8_%E0%A4%97%E0%A5%8B%E0%A4%AA%E0%A4%BE%E0%A4%B2_%E0%A4%B2%E0%A4%A2%E0%A4%BC%E0%A4%BE"
]

# -------- SELENIUM SETUP --------
options = Options()
options.add_argument("--headless")  # Run browser invisibly
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)

# -------- DATA EXTRACTION --------
data = []

for url in urls:
    driver.get(url)
    time.sleep(3)

    # Check if "no content" message exists
    if "फ़िलहाल इस पृष्ठ पर कोई सामग्री नहीं है" in driver.page_source:
        print(f"⚠️ No content found at: {url}")
        continue

    # Extract title (optional)
    try:
        title = driver.find_element(By.ID, "firstHeading").text
    except:
        title = "Unknown Title"

    # Extract all <p> tag text
    paragraphs = driver.find_elements(By.TAG_NAME, "p")
    text_content = "\n".join([p.text for p in paragraphs if p.text.strip()])

    if text_content:
        data.append({
            "URL": url,
            "Title": title,
            "Text": text_content
        })
        print(f"✅ Extracted content from: {url}")
    else:
        print(f"⚠️ No paragraph text found at: {url}")

# -------- SAVE TO CSV --------
df = pd.DataFrame(data)
df.to_csv("marwadi_texts.csv", index=False, encoding="utf-8-sig")

driver.quit()

print("\n✅ Extraction complete! Data saved to marwadi_texts.csv")
