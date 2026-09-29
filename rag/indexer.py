"""
rag/indexer.py

Language-agnostic codebase indexer. Since different production repos are in
different languages (Python, Java, Go, PHP, etc.), we deliberately avoid a
single-language AST parser and instead chunk files by line windows with
overlap. This is simpler, works identically across any language, and is
good enough for retrieval -- the LLM does the actual reasoning once it has
the right chunk in front of it.

If you later want higher-precision chunking for a specific language (e.g.
splitting exactly on function boundaries), you can plug a tree-sitter based
chunker in here per-extension without changing the rest of the pipeline.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import chromadb
from sentence_transformers import SentenceTransformer


@dataclass
class CodeChunk:
    file_path: str
    start_line: int
    end_line: int
    content: str

    @property
    def chunk_id(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


class CodebaseIndexer:
    def __init__(
        self,
        vector_db_path: str,
        collection_name: str,
        embedding_model: str,
        chunk_lines: int = 120,
        chunk_overlap_lines: int = 20,
        include_extensions: list[str] | None = None,
        exclude_dirs: list[str] | None = None,
    ):
        self.chunk_lines = chunk_lines
        self.chunk_overlap_lines = chunk_overlap_lines
        self.include_extensions = set(include_extensions or [])
        self.exclude_dirs = set(exclude_dirs or [])

        self.embedder = SentenceTransformer(embedding_model)
        self.chroma = chromadb.PersistentClient(path=vector_db_path)
        self.collection = self.chroma.get_or_create_collection(collection_name)

    def _iter_source_files(self, repo_path: Path):
        for path in repo_path.rglob("*"):
            if not path.is_file():
                continue
            if any(part in self.exclude_dirs for part in path.parts):
                continue
            if self.include_extensions and path.suffix not in self.include_extensions:
                continue
            yield path

    def _chunk_file(self, path: Path) -> list[CodeChunk]:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return []

        chunks = []
        step = max(self.chunk_lines - self.chunk_overlap_lines, 1)
        for start in range(0, len(lines), step):
            end = min(start + self.chunk_lines, len(lines))
            content = "\n".join(lines[start:end])
            if content.strip():
                chunks.append(
                    CodeChunk(
                        file_path=str(path),
                        start_line=start + 1,
                        end_line=end,
                        content=content,
                    )
                )
            if end == len(lines):
                break
        return chunks

    def index_repo(self, repo_path: str | Path, repo_label: str) -> int:
        """
        Indexes (or re-indexes) a repo into the vector store.
        repo_label namespaces chunk IDs so multiple repos can share one
        collection without colliding.
        Returns number of chunks indexed.
        """
        repo_path = Path(repo_path)
        all_chunks: list[CodeChunk] = []
        for file_path in self._iter_source_files(repo_path):
            all_chunks.extend(self._chunk_file(file_path))

        if not all_chunks:
            return 0

        ids = [f"{repo_label}::{c.chunk_id}" for c in all_chunks]
        documents = [c.content for c in all_chunks]
        metadatas = [
            {
                "repo": repo_label,
                "file_path": Path(c.file_path).relative_to(repo_path).as_posix(),
                "start_line": c.start_line,
                "end_line": c.end_line,
            }
            for c in all_chunks
        ]
        embeddings = self.embedder.encode(documents, show_progress_bar=False).tolist()

        # Upsert in batches to avoid oversized single calls on large repos.
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
            )

        return len(all_chunks)
