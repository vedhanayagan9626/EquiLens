# llm/llm_model.py
import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

def load_llm(model_name="mistral"):
    """
    Load LLM based on model_name.
    Options: 'mistral', 'openai', 'ollama'
    """
    
    if model_name == "mistral":
        # Using Groq for Mistral (fast and free tier available)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        return ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0.3,
            groq_api_key=api_key
        )
    
    elif model_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            openai_api_key=api_key
        )
    
    elif model_name == "ollama":
        # For local Ollama instance
        return ChatOllama(
            model="mistral",
            temperature=0.3
        )
    
    else:
        raise ValueError(f"Unknown model_name: {model_name}")