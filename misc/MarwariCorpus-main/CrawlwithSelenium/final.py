import requests
from bs4 import BeautifulSoup

# ✅ All URLs you provided
urls = [
    "http://gadyakosh.org/gk/%E0%A4%86%E0%A4%AB%E0%A4%B3_/_%E0%A4%AE%E0%A4%A6%E0%A4%A8_%E0%A4%97%E0%A5%8B%E0%A4%AA%E0%A4%BE%E0%A4%B2_%E0%A4%B2%E0%A4%A7%E0%A4%BC%E0%A4%BE",
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
    "http://gadyakosh.org/gk/%E0%A4%AC%E0%A4%BE_%E0%A4%AC%E0%A4%BE%E0%A4%A4_/_%E0%A4%B0%E0%A4%BE%E0%A4%AE%E0%A4%B8%E0%A5%8D%E0%A4%B5%E0%A4%B0%E0%A5%82%E0%A4%AA_%E0%A4%95%E0%A4%BF%E0%A4%B8%E0%A4%BE%E0%A4%A8",
]

all_poems = []

for url in urls:
    print(f"🔹 Fetching: {url}")
    try:
        res = requests.get(url, timeout=15)
        res.encoding = 'utf-8'
        soup = BeautifulSoup(res.text, 'html.parser')

        # Find content div
        content_div = soup.find("div", {"id": "bodyContent"})
        if not content_div:
            print("❌ No content section found.")
            continue

        # Extract all paragraph-like text
        paragraphs = [
            p.get_text(separator=" ", strip=True)
            for p in content_div.find_all(['p', 'div'])
            if p.get_text(strip=True)
            and "फ़िलहाल इस पृष्ठ पर कोई सामग्री नहीं है" not in p.get_text()
            and "मदन गोपाल लढ़ा" not in p.get_text()
            and "रामस्वरूप किसान" not in p.get_text()
            and "नीरज दइया" not in p.get_text()
            and "सत्यनारायण सोनी" not in p.get_text()
        ]

        poem_text = " ".join(paragraphs).strip()

        if poem_text:
            all_poems.append(poem_text)
            print("✅ Extracted poem.")
        else:
            print("⚠️ No poem found.")

    except Exception as e:
        print(f"❌ Error: {e}")

# Save all to file
with open("poems.txt", "w", encoding="utf-8") as f:
    for poem in all_poems:
        f.write(poem + "\n\n" + "="*80 + "\n\n")

print(f"\n✅ Done! Extracted {len(all_poems)} poems saved to poems.txt")
