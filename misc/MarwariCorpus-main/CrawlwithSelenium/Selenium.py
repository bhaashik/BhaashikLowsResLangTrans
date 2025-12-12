import time
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# -----------------------------
SEARCH_QUERIES = ["Marwari dictionary", "Marwari words", "मारवाड़ी शब्द"]
MAX_URLS_PER_QUERY = 15
OUTPUT_FILE = "marwari_actual_words.csv"
word_data = []

# Optional stopwords (common Hindi/Sanskrit words to exclude)
MARWARI_STOPWORDS = set([
    "और", "के", "है", "की", "में", "से", "का", "यह", "था", "थे", "हैं", "पर", "वह"
])

# -----------------------------
# Initialize Selenium driver
# -----------------------------
def init_driver():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

# -----------------------------
# Scroll to load all content
# -----------------------------
def scroll_to_bottom(driver, pause_time=1, max_scrolls=50):
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause_time)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

# -----------------------------
# Check if text contains Devanagari characters
# -----------------------------
def contains_devanagari(text):
    return bool(re.search(r'[\u0900-\u097F]', text))

# -----------------------------
# Heuristic check for actual Marwari word
# -----------------------------
def is_marwari_word(text):
    text = text.strip()
    if not text or len(text) < 2 or len(text) > 12:
        return False
    if " " in text:  # ignore phrases/sentences
        return False
    if text in MARWARI_STOPWORDS:
        return False
    return contains_devanagari(text)

# -----------------------------
# Extract only Marwari words
# -----------------------------
def extract_marwari_words(driver):
    results = []
    elements = driver.find_elements(By.XPATH, "//li | //p | //td | //div | //span")
    for el in elements:
        text = el.text.strip()
        if is_marwari_word(text):
            results.append(text)
    # Handle iframes recursively
    iframes = driver.find_elements(By.TAG_NAME, "iframe")
    for iframe in iframes:
        try:
            driver.switch_to.frame(iframe)
            results.extend(extract_marwari_words(driver))
            driver.switch_to.parent_frame()
        except:
            continue
    return results

# -----------------------------
# Search Bing using Selenium
# -----------------------------
def search_bing_selenium(driver, query, max_results=15):
    driver.get("https://www.bing.com")
    search_box = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "q"))
    )
    search_box.clear()
    search_box.send_keys(query)
    search_box.submit()
    time.sleep(2)  # wait for results

    urls = []
    results = driver.find_elements(By.XPATH, "//li[@class='b_algo']/h2/a")
    for r in results[:max_results]:
        url = r.get_attribute("href")
        urls.append(url)
    return urls

# -----------------------------
# Main function
# -----------------------------
def main():
    driver = init_driver()
    all_urls = []

    # Step 1: Search Bing for queries
    for query in SEARCH_QUERIES:
        print(f"🔍 Searching Bing for: {query}")
        urls = search_bing_selenium(driver, query, max_results=MAX_URLS_PER_QUERY)
        print(f"🌐 Found {len(urls)} URLs")
        all_urls.extend(urls)

    all_urls = list(set(all_urls))  # remove duplicates

    # Step 2: Crawl each URL
    for url in all_urls:
        print(f"🌍 Crawling: {url}")
        try:
            driver.get(url)
            time.sleep(2)
            scroll_to_bottom(driver, pause_time=1, max_scrolls=50)
            extracted = extract_marwari_words(driver)
            print(f"✅ Extracted {len(extracted)} words from {url}")
            word_data.extend(extracted)
        except Exception as e:
            print(f"❌ Failed to crawl {url}: {e}")

    driver.quit()

    # Step 3: Remove duplicates and save CSV
    unique_words = list(sorted(set(word_data)))
    if unique_words:
        df = pd.DataFrame({"Marwari Word (Devanagari)": unique_words})
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        print(f"\n🎉 SUCCESS! Extracted {len(unique_words)} unique Marwari words.")
        print(f"💾 Saved to {OUTPUT_FILE}")
    else:
        print("⚠️ No Marwari words extracted. Try increasing MAX_URLS_PER_QUERY or adjust queries.")

if __name__ == "__main__":
    main()
