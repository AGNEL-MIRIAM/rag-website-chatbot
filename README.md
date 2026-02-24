# 🌐 RAG-Powered Website Chatbot

An intelligent chatbot that can answer questions about any website by dynamically scraping its content and applying Retrieval-Augmented Generation (RAG).

---

## 🚀 Problem Statement

Large Language Models (LLMs) cannot directly access live website content and may produce hallucinated or outdated information.

This project solves that problem by:

- Dynamically scraping website content
- Converting content into semantic embeddings
- Indexing content using FAISS
- Retrieving relevant context
- Generating grounded answers using Groq Llama-3

The chatbot answers questions strictly based on the scraped website content.

---

## 🧠 Solution Overview

This system implements a Retrieval-Augmented Generation (RAG) pipeline:

1. Website scraping (multi-page internal crawl)
2. Text chunking
3. Embedding generation using Sentence Transformers
4. Vector indexing using FAISS
5. Similarity-based retrieval
6. Answer generation using Groq Llama-3

This ensures:
- Accurate
- Context-grounded
- Up-to-date responses

---

## 🏗 Architecture

User Browser  
⬇  
Flask Backend  
⬇  
Website Scraper  
⬇  
Text Chunking  
⬇  
Sentence Transformer Embeddings  
⬇  
FAISS Vector Store  
⬇  
Groq Llama-3 LLM  
⬇  
Response Returned to User  

---

## 🛠 Tech Stack

- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector Search:** FAISS (cosine similarity)
- **LLM:** Groq Llama-3
- **Web Scraping:** Requests + BeautifulSoup

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/AGNEL-MIRIAM/rag-website-chatbot.git
cd rag-website-chatbot
```

---

### 2️⃣ Create virtual environment

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Set Groq API Key

Windows:
```bash
setx GROQ_API_KEY "your_groq_api_key_here"
```

Mac/Linux:
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

Restart terminal after setting environment variable.

---

### 5️⃣ Run the application

```bash
python app.py
```

Open browser:
```
http://127.0.0.1:5000
```

---

## 💬 How to Use

1. Enter a website URL  
2. Choose number of pages to scrape  
3. Click **Build**  
4. Wait for knowledge base to be created  
5. Ask questions related to the website  

The chatbot will answer based strictly on the scraped content.

---

## 🔍 How It Works (Technical Flow)

1. The scraper collects text from internal website pages.
2. The content is split into overlapping chunks.
3. Each chunk is converted into a semantic embedding.
4. FAISS indexes embeddings using cosine similarity.
5. When a user asks a question:
   - The query is embedded
   - Top relevant chunks are retrieved
   - Context is sent to Groq Llama-3
   - A grounded response is generated

This prevents hallucination and ensures contextual answers.

---

## 📈 Key Features

- Multi-page website crawling
- Semantic search using embeddings
- Cosine similarity retrieval
- Real-time LLM answer generation
- Persistent backend pipeline
- Clean custom chat UI
- Secure API key handling via environment variables

---

## 🚧 Future Improvements

- Deployment to cloud
- Authentication system
- Caching vector stores
- Streaming responses
- Support for PDFs and documents

---

## 👨‍💻 Author

Developed as part of claysys AI Hackathon project.

