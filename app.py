# app.py
import streamlit as st
from rag.retriever import load_retriever
from llm.llm_model import load_llm

# Updated LangChain imports - using manual chain approach (most compatible)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="EquiLens – Equity News Assistant", layout="wide")

st.title("🧠 EquiLens – Equity News Research Assistant")
st.caption("Ask questions about equity market news and get insights from recently scraped data.")

query = st.text_input("Ask about the latest equity/market news:")

if query:
    with st.spinner("Processing your query..."):
        retriever = load_retriever(k=5)
        llm = load_llm(model_name="mistral")

        # Define a system prompt & human prompt
        system_prompt = (
            "You are an expert financial news analyst. "
            "Use the following context from recent equity-market news to answer the question. "
            "If you don't know, say you don't know.\n\n"
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        # Create chain to combine docs
        docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        # Build retrieval chain
        chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=docs_chain)

        result = chain.invoke({"input": query})
        answer = result.get("answer", "No answer found")

        st.markdown(f"### 📰 Answer:\n{answer}")
        
        # Optionally show sources
        if "context" in result:
            with st.expander("📚 View Sources"):
                for i, doc in enumerate(result["context"], 1):
                    st.markdown(f"**Source {i}:** {doc.metadata.get('title', 'Unknown')}")
                    st.markdown(f"URL: {doc.metadata.get('source', 'N/A')}")
                    st.markdown("---")