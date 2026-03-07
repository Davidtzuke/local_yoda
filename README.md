# Yoda

A local personal AI agent with long-term memory, tool use, and multiple chat interfaces.

## Features

- **Local-first**: Runs entirely on your machine using Ollama or any OpenAI-compatible API
- **Long-term memory**: SQLite-backed storage with semantic search via sentence-transformers
- **Tool use**: File system access, web search, and an extensible tool interface
- **Dual UI**: Interactive CLI chat and a web-based chat UI (FastAPI + WebSocket)
- **Async throughout**: Built on Python 3.12+ with full async/await architecture

## Quick Start

```bash
# 1. Install (editable mode recommended for development)
pip install -e ".[dev]"

# 2. Make sure Ollama is running with a model pulled
ollama pull llama3.1:8b

# 3. Start a CLI chat
yoda chat

# 4. Or start the web UI
yoda serve
# Then open http://127.0.0.1:8420
```

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `YODA_DEBUG` | `false` | Enable debug logging |
| `YODA_DATA_DIR` | `~/.yoda` | Data storage directory |
| `YODA_LLM_BASE_URL` | `http://localhost:11434/v1` | LLM API endpoint |
| `YODA_LLM_MODEL` | `llama3.1:8b` | Model name |
| `YODA_LLM_API_KEY` | `ollama` | API key |
| `YODA_MEMORY_DB_PATH` | `~/.yoda/memory.db` | Memory database path |
| `YODA_MEMORY_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `YODA_SERVER_HOST` | `127.0.0.1` | Web server bind host |
| `YODA_SERVER_PORT` | `8420` | Web server bind port |

## Project Structure

```
yoda/
  __init__.py          # Package root
  __main__.py          # CLI entry point (click)
  types.py             # Shared data models (Message, Conversation, ToolResult, etc.)
  config.py            # Configuration schema (pydantic-settings)
  core/
    __init__.py
    agent.py           # Agent loop: LLM reasoning + tool dispatch
  memory/
    __init__.py
    base.py            # Abstract memory store interface
    store.py           # Concrete SQLite + embeddings implementation
  tools/
    __init__.py        # Tool registry
    base.py            # Abstract tool interface
    filesystem.py      # File read/write/list tool
    websearch.py       # Web search tool
  ui/
    __init__.py
    cli.py             # Interactive terminal chat
    web.py             # FastAPI web server
    static/
      index.html       # Chat web UI
tests/
  test_memory.py
  test_agent.py
```

## Development

```bash
# Run tests
pytest

# Type check
mypy yoda

# Lint
ruff check yoda
```
