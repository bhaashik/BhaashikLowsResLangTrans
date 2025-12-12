from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

# Target page (the one you gave)
url = "http://gadyakosh.org/gk/%E0%A4%AE%E0%A4%BE%E0%A4%82_/_%E0%A4%B5%E0%A4%BF%E0%A4%9C%E0%A4%AF%E0%A4%A6%E0%A4%BE%E0%A4%A8_%E0%A4%A6%E0%A5%87%E0%A4%A5%E0%A4%BE_%27%E0%A4%AC%E0%A4%BF%E0%A4%9C%E0%A5%8D%E2%80%8D%E0%A4%9C%E0%A5%80%27"

# Setup Chrome options
options = Options()
options.add_argument("--headless")  # No browser window
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# Auto-manage ChromeDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Load the page
driver.get(url)
time.sleep(5)  # Let the dynamic content load

# Parse the page source
soup = BeautifulSoup(driver.page_source, "html.parser")

# Find the main content (poem)
content_div = soup.find("div", {"id": "bodyContent"})

if content_div:
    # Extract text while removing author and links
    for tag in content_div.find_all(["a", "sup", "small"]):
        tag.decompose()

    poem_text = content_div.get_text(separator="\n", strip=True)
    print("\n=== Extracted Poem Text ===\n")
    print(poem_text)

    # Save to file
    with open("poem_ma.txt", "w", encoding="utf-8") as f:
        f.write(poem_text)

    print("\n✅ Poem saved as 'poem_ma.txt'")
else:
    print("❌ Could not find the poem content on the page.")

# Close the browser
driver.quit()
