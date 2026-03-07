# Yoda — Local Personal AI Agent

A Python-based personal AI assistant that runs locally with memory, tools (file system, web search), a CLI interface, and a web chat UI.

## Features

- **CLI REPL** — Rich-based terminal interface with streaming, syntax highlighting, slash commands
- **Web Chat UI** — FastAPI + vanilla JS single-page app with SSE streaming, dark/light theme
- **Persistent Memory** — JSON-backed conversation history stored in `~/.yoda/memory/`
- **Tools** — File system operations (sandboxed) and web search (DuckDuckGo)
- **Pluggable LLM** — Works without API keys (local mode), or connect OpenAI / Ollama

## Quick Start

```bash
# Install
pip install -e .

# Run CLI
yoda cli

# Run Web UI
yoda web
# Open http://127.0.0.1:8000
```

## LLM Backends

```bash
# Local mode (default) — rule-based, no API key needed
yoda cli

# OpenAI
export YODA_LLM_BACKEND=openai
export OPENAI_API_KEY=sk-...
yoda cli

# Ollama (local LLM)
export YODA_LLM_BACKEND=ollama
export OLLAMA_MODEL=llama3.2
yoda cli
```

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `YODA_LLM_BACKEND` | `local` | `local`, `openai`, or `ollama` |
| `OPENAI_API_KEY` | — | Required for OpenAI backend |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |

## CLI Commands

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/tools` | List available tools |
| `/history` | Show conversation history |
| `/clear` | Clear conversation history |
| `/reset` | Clear terminal screen |
| `exit` / `quit` | Exit Yoda |

## Docker

```bash
docker compose up
# Web UI at http://localhost:8000
```

## Project Structure

```
yoda/
  __init__.py
  main.py              # Entry point — CLI arg parsing, mode dispatch
  agent/
    __init__.py
    core.py            # YodaAgent — orchestrates LLM, memory, tools
    memory.py          # Persistent JSON-backed conversation memory
    tools/
      __init__.py      # Tool registry
      filesystem.py    # Sandboxed file read/write/list
      websearch.py     # DuckDuckGo web search
  cli/
    __init__.py
    main.py            # Async REPL loop
    protocol.py        # AgentProtocol ABC + data classes
    renderer.py        # Rich-based output rendering
    commands.py        # Slash command handling
  web/
    __init__.py
    app.py             # FastAPI app with SSE streaming
    static/
      index.html       # Chat UI (vanilla JS)
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Requirements

- Python 3.12+
- No API keys needed for local mode
