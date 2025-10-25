import requests
from bs4 import BeautifulSoup
import json
import os

#Data path to store a scrappping ddata as json in data folder
DATA_PATH = os.path.join("data", "articles.json")

def scrape_equity_news():
    urls = [
        "https://www.moneycontrol.com/news/business/markets/",
        "https://economictimes.indiatimes.com/markets",
        "https://www.reuters.com/markets/",
    ]

    articles = []

    for url in urls:
        try:
            res = requests.get(url, timeout=10)
            soup =  BeautifulSoup(res.text, "html.parser")
            links = soup.find_all("a", href = True)

            for link in links:
                text = link.get_text(strip=True)
                href = link["href"]
                if(len(text) > 25 and
                   "http" in href and
                   ("market" in href or "equity" in href or "stocks" in href)):
                    articles.append({"title":text , "url": href})
        except Exception as e:
            print(f"Error Scrapping url {url}: {e}")
        
    # Remove duplicates
    seen = set()
    unique_articles = []

    for art in articles:
        if art["url"] not in  seen:
            seen.add(art["url"])
            unique_articles.append(art)
    
    os.makedirs('data', exist_ok= True)
    with open(DATA_PATH, "w", encoding="utf-8") as  f:
        json.dump(unique_articles, f, indent=2)

    print(f"✅ Scraped and saved {len(unique_articles)} news articles.")
