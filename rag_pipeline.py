from typing import List


class RAGPipeline:
    """
    Core RAG system:
    - Text chunking
    - (Next: Embedding generation)
    - (Next: Vector indexing)
    - (Next: Retrieval)
    - (Next: Answer generation)
    """

    def __init__(self):
        self.chunks = []

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
