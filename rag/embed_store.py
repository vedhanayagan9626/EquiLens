# embed_store.py
import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma


DATA_PATH = os.path.join("data", "articles.json")
CHROMA_PATH = os.path.join("data", "chroma")

def build_vectorstore():
    if not os.path.exists(DATA_PATH):
        print("❌ No scraped data found. Run scraper first.")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)

    texts = [a["title"] + "\n" + a.get("url","") for a in articles]  # include URL if you like
    metadatas = [{"source": a["url"], "title": a["title"]} for a in articles]

    # instantiate embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # build or load vectorstore
    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_PATH,
        collection_name="equity_news"
    )
    vectordb.persist()
    print("✅ Vector store built and persisted at", CHROMA_PATH)
