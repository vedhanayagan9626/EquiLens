# run_refresh.py
import schedule, time
from scraper.scraper import scrape_equity_news
from rag.embed_store import build_vectorstore

def job():
    scrape_equity_news()
    build_vectorstore()

schedule.every(3).hours.do(job)

print("🕒 Scheduler running every 3 hours...")
while True:
    schedule.run_pending()
    time.sleep(60)
