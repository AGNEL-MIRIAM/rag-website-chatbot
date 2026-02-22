"""
app.py

Main entry point for Streamlit application.
Handles user interface and chatbot interaction.
"""

import streamlit as st


def main():
    st.set_page_config(page_title="RAG Website Chatbot")

    st.title("RAG-Powered Website Chatbot")

    st.markdown("Enter a website URL and ask questions based on its content.")

    url = st.text_input("Website URL")
    query = st.text_input("Ask a question")

    if st.button("Submit"):
        if not url or not query:
            st.warning("Please provide both URL and question.")
        else:
            st.info("RAG pipeline will process the request here.")


if __name__ == "__main__":
    main()
