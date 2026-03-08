"""Chunking engine with semantic, hierarchical, and code-aware strategies.

Implements multiple chunking approaches:
- Fixed-size: simple character/token-based chunking with overlap
- Semantic: split on topic/meaning boundaries using embedding similarity
- Hierarchical: multi-level chunking (document → section → paragraph → sentence)
- Code-aware: respects code structure (functions, classes, blocks)
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with metadata about its position and hierarchy."""

    content: str
    index: int = 0
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None  # for hierarchical chunking
    level: int = 0  # hierarchy level (0 = top)


class ChunkingStrategy(ABC):
    """Abstract chunking strategy."""

    @abstractmethod
    def chunk(self, text: str, **kwargs: Any) -> list[Chunk]: ...


class FixedSizeChunker(ChunkingStrategy):
    """Simple fixed-size chunking with overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, **kwargs: Any) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind(". ")
                last_newline = chunk_text.rfind("\n")
                break_point = max(last_period, last_newline)
                if break_point > self.chunk_size * 0.5:
                    end = start + break_point + 1
                    chunk_text = text[start:end]

            chunks.append(Chunk(
                content=chunk_text.strip(),
                index=idx,
                start_char=start,
                end_char=end,
            ))
            start = end - self.overlap
            idx += 1

        return [c for c in chunks if c.content]


class SemanticChunker(ChunkingStrategy):
    """Semantic chunking — splits text at meaning boundaries.

    Uses sentence-level similarity to find natural breakpoints.
    When consecutive sentences have low similarity, a chunk boundary is inserted.
    """

    def __init__(
        self,
        embedder: Any = None,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1500,
    ) -> None:
        self._embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Handle common abbreviations and edge cases
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        # Also split on double newlines (paragraph boundaries)
        result: list[str] = []
        for sent in sentences:
            parts = sent.split("\n\n")
            result.extend(p.strip() for p in parts if p.strip())
        return result

    def chunk(self, text: str, **kwargs: Any) -> list[Chunk]:
        """Chunk text semantically. Falls back to fixed-size if no embedder."""
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [Chunk(content=text, index=0, start_char=0, end_char=len(text))]

        if self._embedder is None:
            # Fallback: use paragraph-based chunking
            return self._paragraph_chunk(text, sentences)

        # With embedder, compute similarity between consecutive sentence groups
        return self._semantic_split(text, sentences)

    def _paragraph_chunk(self, text: str, sentences: list[str]) -> list[Chunk]:
        """Fallback: group sentences into chunks by paragraph boundaries."""
        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0
        char_pos = 0

        for sent in sentences:
            if current_len + len(sent) > self.max_chunk_size and current_len >= self.min_chunk_size:
                chunk_text = " ".join(current)
                chunks.append(Chunk(
                    content=chunk_text,
                    index=len(chunks),
                    start_char=char_pos - current_len,
                    end_char=char_pos,
                ))
                current = []
                current_len = 0

            current.append(sent)
            current_len += len(sent) + 1
            char_pos += len(sent) + 1

        if current:
            chunk_text = " ".join(current)
            chunks.append(Chunk(
                content=chunk_text,
                index=len(chunks),
                start_char=char_pos - current_len,
                end_char=char_pos,
            ))

        return chunks

    def _semantic_split(self, text: str, sentences: list[str]) -> list[Chunk]:
        """Split using embedding similarity between sentence windows."""
        import asyncio
        import numpy as np

        # Get embeddings for each sentence
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a future
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    embeddings = pool.submit(
                        lambda: asyncio.run(self._embedder.embed(sentences))
                    ).result()
            else:
                embeddings = loop.run_until_complete(self._embedder.embed(sentences))
        except RuntimeError:
            embeddings = asyncio.run(self._embedder.embed(sentences))

        # Compute cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            a = np.array(embeddings[i])
            b = np.array(embeddings[i + 1])
            sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            similarities.append(float(sim))

        # Find breakpoints where similarity drops below threshold
        breakpoints: list[int] = []
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                breakpoints.append(i + 1)

        # Build chunks from breakpoints
        chunks: list[Chunk] = []
        start_idx = 0
        char_pos = 0

        for bp in breakpoints + [len(sentences)]:
            chunk_sentences = sentences[start_idx:bp]
            chunk_text = " ".join(chunk_sentences)

            # Enforce size limits
            if len(chunk_text) > self.max_chunk_size:
                # Sub-chunk if too large
                sub_chunker = FixedSizeChunker(self.max_chunk_size, overlap=64)
                sub_chunks = sub_chunker.chunk(chunk_text)
                for sc in sub_chunks:
                    sc.index = len(chunks)
                    sc.start_char += char_pos
                    sc.end_char += char_pos
                    chunks.append(sc)
            elif len(chunk_text) >= self.min_chunk_size or not chunks:
                chunks.append(Chunk(
                    content=chunk_text,
                    index=len(chunks),
                    start_char=char_pos,
                    end_char=char_pos + len(chunk_text),
                ))
            else:
                # Merge with previous chunk if too small
                if chunks:
                    chunks[-1].content += " " + chunk_text
                    chunks[-1].end_char = char_pos + len(chunk_text)

            char_pos += len(chunk_text) + 1
            start_idx = bp

        return chunks


class HierarchicalChunker(ChunkingStrategy):
    """Multi-level hierarchical chunking.

    Level 0: Full document summary
    Level 1: Section-level chunks (split by headers/major breaks)
    Level 2: Paragraph-level chunks
    """

    def __init__(
        self,
        section_pattern: str = r'\n#{1,3}\s+|\n\n\n+',
        max_section_size: int = 2000,
        max_paragraph_size: int = 500,
    ) -> None:
        self.section_pattern = section_pattern
        self.max_section_size = max_section_size
        self.max_paragraph_size = max_paragraph_size

    def chunk(self, text: str, **kwargs: Any) -> list[Chunk]:
        chunks: list[Chunk] = []
        doc_id = f"doc_{id(text)}"

        # Level 0: Document summary (first ~500 chars or abstract)
        summary = text[:500].strip()
        if len(text) > 500:
            summary += "..."
        chunks.append(Chunk(
            content=summary,
            index=0,
            start_char=0,
            end_char=min(500, len(text)),
            level=0,
            metadata={"type": "summary", "doc_id": doc_id},
        ))

        # Level 1: Section-level
        sections = re.split(self.section_pattern, text)
        sections = [s.strip() for s in sections if s.strip()]

        char_pos = 0
        for i, section in enumerate(sections):
            section_id = f"{doc_id}_s{i}"
            chunks.append(Chunk(
                content=section[:self.max_section_size],
                index=len(chunks),
                start_char=char_pos,
                end_char=char_pos + len(section),
                level=1,
                parent_id=doc_id,
                metadata={"type": "section", "section_index": i, "doc_id": doc_id},
            ))

            # Level 2: Paragraph-level within each section
            paragraphs = section.split("\n\n")
            para_pos = char_pos
            for j, para in enumerate(paragraphs):
                para = para.strip()
                if not para or len(para) < 20:
                    para_pos += len(para) + 2
                    continue

                if len(para) > self.max_paragraph_size:
                    # Sub-chunk long paragraphs
                    sub_chunker = FixedSizeChunker(self.max_paragraph_size, overlap=50)
                    for sub in sub_chunker.chunk(para):
                        sub.level = 2
                        sub.parent_id = section_id
                        sub.start_char += para_pos
                        sub.end_char += para_pos
                        sub.index = len(chunks)
                        sub.metadata = {"type": "paragraph", "section_index": i, "doc_id": doc_id}
                        chunks.append(sub)
                else:
                    chunks.append(Chunk(
                        content=para,
                        index=len(chunks),
                        start_char=para_pos,
                        end_char=para_pos + len(para),
                        level=2,
                        parent_id=section_id,
                        metadata={"type": "paragraph", "section_index": i, "doc_id": doc_id},
                    ))
                para_pos += len(para) + 2

            char_pos += len(section) + 1

        return chunks


class CodeAwareChunker(ChunkingStrategy):
    """Code-aware chunking that respects code structure.

    Splits code into logical units: functions, classes, imports, top-level blocks.
    Preserves docstrings and comments with their associated code.
    """

    def __init__(self, max_chunk_size: int = 1000) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str, **kwargs: Any) -> list[Chunk]:
        language = kwargs.get("language", self._detect_language(text))

        if language == "python":
            return self._chunk_python(text)
        else:
            return self._chunk_generic_code(text)

    def _detect_language(self, text: str) -> str:
        """Simple language detection based on content."""
        if re.search(r'^(def |class |import |from .+ import )', text, re.MULTILINE):
            return "python"
        if re.search(r'^(function |const |let |var |import .+ from )', text, re.MULTILINE):
            return "javascript"
        return "generic"

    def _chunk_python(self, text: str) -> list[Chunk]:
        """Chunk Python code by top-level constructs."""
        chunks: list[Chunk] = []
        lines = text.split("\n")
        current_block: list[str] = []
        block_start = 0
        char_pos = 0
        in_class_or_func = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Detect top-level definitions
            is_top_level_def = (
                re.match(r'^(def |class |async def )', line) and not line.startswith(" ")
            )

            if is_top_level_def and current_block:
                # Save current block
                block_text = "\n".join(current_block)
                if block_text.strip():
                    chunks.append(Chunk(
                        content=block_text,
                        index=len(chunks),
                        start_char=char_pos - len(block_text),
                        end_char=char_pos,
                        metadata={"type": "code_block", "language": "python"},
                    ))
                current_block = []
                block_start = char_pos

            current_block.append(line)
            char_pos += len(line) + 1

            # Check chunk size limit
            block_text = "\n".join(current_block)
            if len(block_text) > self.max_chunk_size and not is_top_level_def:
                if block_text.strip():
                    chunks.append(Chunk(
                        content=block_text,
                        index=len(chunks),
                        start_char=block_start,
                        end_char=char_pos,
                        metadata={"type": "code_block", "language": "python"},
                    ))
                current_block = []
                block_start = char_pos

        # Last block
        if current_block:
            block_text = "\n".join(current_block)
            if block_text.strip():
                chunks.append(Chunk(
                    content=block_text,
                    index=len(chunks),
                    start_char=block_start,
                    end_char=char_pos,
                    metadata={"type": "code_block", "language": "python"},
                ))

        return chunks

    def _chunk_generic_code(self, text: str) -> list[Chunk]:
        """Chunk code by blank-line-separated blocks."""
        blocks = re.split(r'\n\s*\n', text)
        chunks: list[Chunk] = []
        char_pos = 0

        current_block: list[str] = []
        current_len = 0

        for block in blocks:
            block = block.strip()
            if not block:
                char_pos += 1
                continue

            if current_len + len(block) > self.max_chunk_size and current_block:
                chunk_text = "\n\n".join(current_block)
                chunks.append(Chunk(
                    content=chunk_text,
                    index=len(chunks),
                    start_char=char_pos - len(chunk_text),
                    end_char=char_pos,
                    metadata={"type": "code_block", "language": "generic"},
                ))
                current_block = []
                current_len = 0

            current_block.append(block)
            current_len += len(block) + 2
            char_pos += len(block) + 2

        if current_block:
            chunk_text = "\n\n".join(current_block)
            chunks.append(Chunk(
                content=chunk_text,
                index=len(chunks),
                start_char=char_pos - len(chunk_text),
                end_char=char_pos,
                metadata={"type": "code_block"},
            ))

        return chunks


def create_chunker(
    strategy: str = "semantic",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    embedder: Any = None,
    **kwargs: Any,
) -> ChunkingStrategy:
    """Factory to create a chunking strategy."""
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_overlap)
    elif strategy == "semantic":
        return SemanticChunker(
            embedder=embedder,
            min_chunk_size=kwargs.get("min_chunk_size", 100),
            max_chunk_size=kwargs.get("max_chunk_size", chunk_size * 3),
            similarity_threshold=kwargs.get("similarity_threshold", 0.5),
        )
    elif strategy == "hierarchical":
        return HierarchicalChunker(
            max_section_size=kwargs.get("max_section_size", chunk_size * 4),
            max_paragraph_size=chunk_size,
        )
    elif strategy == "code":
        return CodeAwareChunker(max_chunk_size=chunk_size * 2)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
