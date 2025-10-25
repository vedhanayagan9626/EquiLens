# app.py
import streamlit as st
from retriever import load_retriever
from llm.llm_model import load_llm

# Import chain creation API (newer pattern)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="EquiLens – Equity News Assistant", layout="wide")

st.title("🧠 EquiLens – Equity News Research Assistant")
st.caption("Ask questions about equity market news and get insights from recently scraped data.")

query = st.text_input("Ask about the latest equity/market news:")

if query:
    with st.spinner("Processing your query..."):
        retriever = load_retriever(k=5)
        llm = load_llm(model_name="mistral")  # or whichever you use

        # Define a system prompt & human prompt
        system_prompt = (
            "You are an expert financial news analyst. "
            "Use the following context from recent equity-market news to answer the question. "
            "If you don't know, say you don't know."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        # create chain to combine docs
        docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        # build retrieval chain
        chain = create_retrieval_chain(retriever=retriever, combine_documents_chain=docs_chain)

        result = chain.invoke({"input": query})
        answer = result["text"] if isinstance(result, dict) and "text" in result else result

        st.markdown(f"### 📰 Answer:\n{answer}")
