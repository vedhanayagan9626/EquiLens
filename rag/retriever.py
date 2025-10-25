# rag/retriever.py
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = os.path.join("data", "chroma")

def load_retriever(k: int = 5):
    """Load the Chroma retriever for similarity search."""
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(
            f"Vector store not found at {CHROMA_PATH}. "
            "Please run 'python rag/embed_store.py' first to build it."
        )
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectordb = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="equity_news"
    )
    
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    return retriever