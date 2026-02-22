from sentence_transformers import SentenceTransformer
from groq import Groq
import faiss
import numpy as np

class RAGPipeline:
    def __init__(self, groq_api_key: str):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = Groq(api_key=groq_api_key)
        self.chunks = []
        self.index = None

    def chunk_text(self, text: str, chunk_size: int = 150, overlap: int = 30):
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunks.append(" ".join(words[start:end]))
            start += chunk_size - overlap
        self.chunks = [c for c in chunks if len(c.strip()) > 40]

    def build_knowledge_base(self, text: str):
        self.chunk_text(text)
        print(f"Built knowledge base with {len(self.chunks)} chunks")

        embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        print("FAISS index ready.")

    def generate_answer(self, query: str, k: int = 5) -> str:
        # Embed and retrieve
        query_emb = self.embedding_model.encode([query])
        query_emb = np.array(query_emb).astype("float32")
        faiss.normalize_L2(query_emb)

        distances, indices = self.index.search(query_emb, k)
        retrieved = [
            self.chunks[idx]
            for idx, dist in zip(indices[0], distances[0])
            if dist > 0.2 and idx < len(self.chunks)
        ]

        if not retrieved:
            return "Could not find relevant content for this question."

        context = "\n\n".join(retrieved)

        # Call Groq LLM
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer questions based only on the provided context. Be concise and accurate."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based only on the context above:"
                }
            ],
            temperature=0.2,
            max_tokens=300
        )

        return response.choices[0].message.content
