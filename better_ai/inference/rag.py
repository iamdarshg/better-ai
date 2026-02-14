"""
Simple RAG (Retrieval-Augmented Generation) system for Better AI.
"""

from typing import List, Dict, Any
import numpy as np

class SimpleRAG:
    """
    Retrieves relevant document chunks to augment generation.
    """
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.index = [] # List of {text: str, vector: np.array}

    def add_documents(self, documents: List[str]):
        """
        Chunks and indexes documents.
        """
        for doc in documents:
            # Simple chunking by newline
            chunks = doc.split("\n\n")
            for chunk in chunks:
                if len(chunk.strip()) > 50:
                    vector = self._embed(chunk)
                    self.index.append({"text": chunk, "vector": vector})
        print(f"Indexed {len(self.index)} chunks.")

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieves top-k relevant chunks.
        """
        if not self.index:
            return []

        query_vec = self._embed(query)

        # Calculate cosine similarity
        scores = []
        for item in self.index:
            sim = self._cosine_similarity(query_vec, item["vector"])
            scores.append(sim)

        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.index[i]["text"] for i in top_indices]

    def _embed(self, text: str) -> np.array:
        """Mock embedding function"""
        # In real use, we'd use model.encode(text)
        return np.random.randn(128)

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def augment_prompt(query: str, rag: SimpleRAG) -> str:
    """
    Adds retrieved context to the user query.
    """
    context_chunks = rag.retrieve(query)
    if not context_chunks:
        return query

    context_str = "\n".join([f"- {c}" for c in context_chunks])
    return f"[CONTEXT]\n{context_str}\n[/CONTEXT]\n\nQuestion: {query}"
