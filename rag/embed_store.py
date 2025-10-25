# rag/embed_store.py
import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_PATH = os.path.join("data", "articles.json")
CHROMA_PATH = os.path.join("data", "chroma")

def build_vectorstore():
    """Build or update the Chroma vector store from scraped articles."""
    if not os.path.exists(DATA_PATH):
        print("❌ No scraped data found. Run scraper first.")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)

    if not articles:
        print("⚠️ No articles found in data file.")
        return

    # Combine title and URL for better context
    texts = [f"{a['title']}\nURL: {a.get('url', '')}" for a in articles]
    metadatas = [{"source": a["url"], "title": a["title"]} for a in articles]

    # Instantiate embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build or recreate vectorstore
    # Remove old collection if exists
    if os.path.exists(CHROMA_PATH):
        import shutil
        shutil.rmtree(CHROMA_PATH)
        print("🗑️ Removed old vector store")

    # Create new vectorstore
    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_PATH,
        collection_name="equity_news"
    )
    
    print(f"✅ Vector store built with {len(articles)} articles at {CHROMA_PATH}")

if __name__ == "__main__":
    build_vectorstore()