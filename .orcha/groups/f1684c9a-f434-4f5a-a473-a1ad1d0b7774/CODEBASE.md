# Yoda - Local Personal AI Agent

## About This Project
Yoda is a Python-based personal AI assistant that runs locally. It features persistent memory with semantic search, tool integrations (filesystem, web search), a CLI interface, and a web chat UI. All async, Python 3.12+.

## Tech Stack
- **Python 3.12+** with async/await throughout
- **aiosqlite** for persistent memory storage in SQLite
- **sentence-transformers** (all-MiniLM-L6-v2) for local text embeddings
- **numpy** for cosine similarity computation
- **pydantic** + **pydantic-settings** for types and config
- **OpenAI-compatible API** (defaults to Ollama) for LLM inference

## What This Branch Does
The `feature/memory-system` branch implements the full memory and context management subsystem:
- `SQLiteMemoryStore`: async CRUD + semantic search over memories
- `EmbeddingPipeline`: local embedding generation with hash-based fallback
- `ContextWindow`: assembles LLM prompts within token budgets
- `ConversationSummarizer`: compacts long conversations into summaries
- `MemoryRetriever`: semantic retrieval with recency boosting

## Key Files
- `yoda/memory/store.py` — SQLite-backed memory store (main implementation)
- `yoda/memory/embeddings.py` — embedding pipeline with caching
- `yoda/memory/context.py` — context window, summarizer, retriever
- `yoda/memory/base.py` — abstract base class (from Architect)
- `yoda/types.py` — shared data models: MemoryEntry, Conversation, Message
- `yoda/config.py` — MemorySettings with db_path, embedding_model, thresholds
- `tests/test_memory.py` — 34 tests covering all memory components
