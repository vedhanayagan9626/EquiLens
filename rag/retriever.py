# retriever.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma

CHROMA_PATH = "data/chroma"

def load_retriever(k: int = 5):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding=embeddings, collection_name="equity_news")
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return retriever
