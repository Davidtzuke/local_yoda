## About This Project

Yoda is a local personal AI agent built in Python. It provides both a CLI terminal REPL and a web chat UI for interacting with an AI assistant that has persistent memory and tools (file system operations, web search). It runs entirely locally with no API keys required in default mode.

## Tech Stack

- **Python 3.12+** with async throughout
- **Rich** for CLI terminal rendering (syntax highlighting, tables, panels)
- **FastAPI + Uvicorn** for web chat UI with SSE streaming
- **Vanilla JS** single-page chat frontend (no framework)
- **JSON files** for persistent conversation memory (~/.yoda/memory/)
- **DuckDuckGo API** for web search (no key needed)
- **Pluggable LLM**: local rule-based, OpenAI API, or Ollama

## What This Branch Does

The `feature/integration` branch wires together the CLI interface (Rich REPL) and Web Chat UI (FastAPI+JS) with a unified `YodaAgent` core. It adds the agent orchestration layer (`yoda/agent/core.py`), persistent JSON memory (`yoda/agent/memory.py`), sandboxed filesystem and web search tools (`yoda/agent/tools/`), a CLI entry point with argparse (`yoda/main.py`), project packaging (`pyproject.toml`), and Docker deployment support.

## Key Files

- `yoda/main.py` — Entry point, CLI arg parsing, launches cli or web mode
- `yoda/agent/core.py` — YodaAgent class implementing both CLI and Web protocols
- `yoda/agent/memory.py` — ConversationMemory with JSON persistence
- `yoda/agent/tools/filesystem.py` — Sandboxed file read/write/list
- `yoda/agent/tools/websearch.py` — DuckDuckGo search tool
- `yoda/cli/main.py` — Async REPL with Rich rendering
- `yoda/web/app.py` — FastAPI app with SSE streaming
- `pyproject.toml` — Package config with optional deps
