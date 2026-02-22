from typing import List
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np
import torch


class RAGPipeline:
    """
    Complete RAG System:
    - Text chunking
    - Embedding generation
    - FAISS vector indexing
    - Semantic retrieval
    - LLM answer generation
    """

    def __init__(self):
        # Embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # LLM model
        self.llm_name = "google/flan-t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
        self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_name)

        self.chunks = []
        self.embeddings = None
        self.index = None

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []

        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = words[start:end]
            chunks.append(" ".join(chunk))
            start += chunk_size - overlap

        self.chunks = chunks
        return chunks

    def generate_embeddings(self):
        if not self.chunks:
            raise ValueError("No chunks available. Run chunk_text first.")

        self.embeddings = self.embedding_model.encode(self.chunks)
        return self.embeddings

    def build_vector_store(self):
        if self.embeddings is None:
            raise ValueError("No embeddings available. Run generate_embeddings first.")

        embeddings = np.array(self.embeddings).astype("float32")
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def retrieve(self, query: str, k: int = 3):
        if self.index is None:
            raise ValueError("Vector store not built. Run build_vector_store first.")

        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, k)
        retrieved_chunks = [self.chunks[i] for i in indices[0]]

        return retrieved_chunks

    def generate_answer(self, query: str, k: int = 3):
        retrieved_chunks = self.retrieve(query, k)
        context = " ".join(retrieved_chunks)

        prompt = f"""
        Answer the question based only on the context below.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=150
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
