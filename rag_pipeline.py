from typing import List
from sentence_transformers import SentenceTransformer


class RAGPipeline:
    """
    Core RAG system:
    - Text chunking
    - Embedding generation
    - (Next: Vector indexing)
    - (Next: Retrieval)
    - (Next: Answer generation)
    """

    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Splits text into overlapping word-based chunks.
        """
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
        """
        Generate embeddings for stored chunks.
        """
        if not self.chunks:
            raise ValueError("No chunks available. Run chunk_text first.")

        self.embeddings = self.embedding_model.encode(self.chunks)
        return self.embeddings
