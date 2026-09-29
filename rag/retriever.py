"""
rag/retriever.py

Given a natural-language description of a bug/symptom, retrieves the most
relevant code chunks from the vector store built by indexer.py.
"""

from __future__ import annotations
from dataclasses import dataclass
import chromadb
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    file_path: str
    start_line: int
    end_line: int
    content: str
    distance: float

    def as_prompt_text(self) -> str:
        return f"--- {self.file_path} (lines {self.start_line}-{self.end_line}) ---\n{self.content}"


class CodebaseRetriever:
    def __init__(self, vector_db_path: str, collection_name: str, embedding_model: str):
        self.embedder = SentenceTransformer(embedding_model)
        self.chroma = chromadb.PersistentClient(path=vector_db_path)
        self.collection = self.chroma.get_or_create_collection(collection_name)

    def search(self, query: str, repo_label: str, top_k: int = 8) -> list[RetrievedChunk]:
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where={"repo": repo_label},
        )

        chunks = []
        if not results["ids"] or not results["ids"][0]:
            return chunks

        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            chunks.append(
                RetrievedChunk(
                    file_path=meta["file_path"],
                    start_line=meta["start_line"],
                    end_line=meta["end_line"],
                    content=doc,
                    distance=dist,
                )
            )
        return chunks
