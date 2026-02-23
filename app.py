"""
app.py

Streamlit frontend for RAG Website Chatbot
"""

import streamlit as st
from scraper import scrape_website
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="RAG Website Chatbot")
st.title("🌐 RAG-Powered Website Chatbot")

st.markdown("Enter a website URL and ask questions based on its content.")

# Sidebar
st.sidebar.title("Configuration")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

url = st.text_input("Website URL")

if st.button("Build Knowledge Base"):
    if not url or not groq_api_key:
        st.warning("Please provide both URL and Groq API key.")
    else:
        with st.spinner("Scraping and building knowledge base..."):
            content = scrape_website(url)
            pipeline = RAGPipeline(groq_api_key=groq_api_key)
            pipeline.build_knowledge_base(content)
            st.session_state.pipeline = pipeline
        st.success("Knowledge base built successfully!")

# Chat section
if st.session_state.pipeline:
    user_query = st.chat_input("Ask a question about the website...")

    if user_query:
        st.session_state.chat_history.append(("user", user_query))

        with st.spinner("Generating answer..."):
            answer = st.session_state.pipeline.generate_answer(user_query)

        st.session_state.chat_history.append(("assistant", answer))

    # Display chat
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(message)
        else:
            st.chat_message("assistant").write(message)
