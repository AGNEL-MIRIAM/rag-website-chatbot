from typing import List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class RAGPipeline:
    """
    Core RAG system:
    - Text chunking
    - Embedding generation
    - FAISS vector indexing
    - Semantic retrieval
    """

    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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

        return retrieved_chunks, distances
