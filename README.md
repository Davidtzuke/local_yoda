# 🧠 Yoda — Personal AI Assistant

A locally-running Python AI agent with infinite memory, knowledge graph reasoning, token optimization, full tool access, and MCP server integration for Claude Code.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI / MCP Server                         │
│  ┌──────────────────┐    ┌──────────────────────────────────┐   │
│  │  Rich Terminal UI │    │  MCP Server (SSE + stdio)        │   │
│  │  Slash commands   │    │  Tools: remember, recall,        │   │
│  │  Streaming MD     │    │  graph_query, get_preferences    │   │
│  └────────┬─────────┘    └───────────────┬──────────────────┘   │
│           │                               │                      │
│           └───────────┬───────────────────┘                      │
│                       ▼                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Orchestrator                           │    │
│  │  Init order • Plugin wiring • Context injectors          │    │
│  │  Signal handling • Graceful shutdown                      │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                     Agent (ReAct Loop)                    │    │
│  │  Thought → Action → Observation • Streaming • Plugins    │    │
│  └──┬──────────┬──────────┬──────────┬─────────────────────┘    │
│     ▼          ▼          ▼          ▼                           │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────────┐                    │
│  │Memory│  │ Know │  │Token │  │  Tools   │                    │
│  │ RAG  │  │Graph │  │ Opt  │  │File/Shell│                    │
│  │Vector│  │NX+SQL│  │Cache │  │Web/MCP   │                    │
│  │BM25  │  │Reason│  │Cost  │  │Computer  │                    │
│  └──────┘  └──────┘  └──────┘  └──────────┘                    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              LLM Providers (Anthropic/OpenAI/Ollama)      │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install
pip install -e .

# Set API key
export YODA_PROVIDER_API_KEY=sk-...

# Run interactive CLI
yoda

# Run as MCP server (SSE)
yoda --mcp --port 8765

# Run as MCP server (stdio, for Claude Code)
yoda --mcp-stdio
```

## Docker

```bash
# Interactive CLI
docker compose up yoda

# MCP server
docker compose --profile mcp up yoda-mcp
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/remember <text>` | Store a memory |
| `/forget <id>` | Forget a memory |
| `/search <query>` | Search memories |
| `/graph <query>` | Query knowledge graph |
| `/status` | Show agent status |
| `/cost` | Show token usage & cost |
| `/claude` | Generate CLAUDE.md |
| `/reset` | Clear conversation |
| `/quit` | Exit |

## MCP Server for Claude Code

Add to your Claude Code MCP config (`~/.claude/mcp_servers.json`):

```json
{
  "yoda": {
    "command": "yoda",
    "args": ["--mcp-stdio"],
    "env": {
      "YODA_PROVIDER_API_KEY": "your-key"
    }
  }
}
```

### MCP Tools

- **remember** — Store information in Yoda's long-term memory
- **recall** — Search memory for relevant information
- **graph_query** — Query the knowledge graph
- **get_preferences** — Get learned user preferences

## Project Structure

```
yoda/
├── core/              # Agent, config, messages, providers, plugins
│   ├── agent.py       # ReAct loop with streaming
│   ├── config.py      # YAML config + env overrides
│   ├── messages.py    # Typed message protocol
│   ├── plugins.py     # Plugin framework
│   └── providers/     # LLM providers (Anthropic, OpenAI, Ollama)
├── memory/            # RAG + vector store + persistence
│   ├── vector_store.py # ChromaDB / FAISS backends
│   ├── embeddings.py  # Local + API embeddings
│   ├── chunking.py    # Multi-strategy chunking
│   ├── retrieval.py   # Hybrid retrieval (BM25 + dense + MMR)
│   ├── manager.py     # Memory lifecycle
│   └── plugin.py      # Agent integration
├── knowledge/         # Knowledge graph
│   ├── graph.py       # NetworkX + SQLite persistence
│   ├── extractor.py   # Entity/relationship extraction
│   ├── queries.py     # Natural language graph queries
│   ├── reasoning.py   # Multi-hop inference
│   ├── updater.py     # Auto-update from conversations
│   ├── visualization.py # D3.js, Mermaid, ASCII
│   └── plugin.py      # Agent integration
├── optimization/      # Token management
│   ├── tokens.py      # Per-model counting + budgets
│   ├── compressor.py  # Progressive context compression
│   ├── window.py      # Priority sliding window
│   ├── cache.py       # Semantic response cache
│   ├── cost.py        # Cost tracking + alerts
│   └── plugin.py      # Agent integration
├── tools/             # Tool access
│   ├── registry.py    # @tool decorator + registry
│   ├── executor.py    # Parallel execution + approval
│   ├── mcp_client.py  # MCP protocol client
│   ├── builtins/      # File ops, shell, web, calendar, notes
│   ├── computer/      # Screen, keyboard, mouse, apps
│   └── plugin.py      # Agent integration
├── cli/               # Terminal interface
│   ├── app.py         # Rich UI + slash commands
│   ├── orchestrator.py # Component wiring
│   └── claude_gen.py  # CLAUDE.md generator
├── mcp_server/        # MCP server
│   ├── server.py      # SSE + stdio transports
│   └── transport.py   # SSE utilities
└── cli.py             # Entry point
```

## Configuration

Create `yoda.yaml` or `~/.yoda/config.yaml`:

```yaml
provider:
  name: anthropic
  model: claude-sonnet-4-20250514
  api_key: ""  # or use YODA_PROVIDER_API_KEY env var

memory:
  backend: chromadb
  persist_dir: ~/.yoda/memory

knowledge_graph:
  backend: networkx
  persist_path: ~/.yoda/kg.json

tokens:
  max_context_tokens: 128000
  compression_enabled: true
  cost_tracking: true
```

## Tech Stack

- **Python 3.12+** with full async/await
- **Pydantic v2** for config and message models
- **Rich** for terminal UI
- **Click** for CLI framework
- **ChromaDB / FAISS** for vector storage
- **NetworkX + SQLite** for knowledge graph
- **tiktoken** for token counting
- **aiohttp** for MCP SSE server
- **Docker** for containerized deployment

## License

MIT
