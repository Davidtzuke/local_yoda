<p align="center">
  <pre align="center">
  ██╗   ██╗ ██████╗ ██████╗  █████╗
  ╚██╗ ██╔╝██╔═══██╗██╔══██╗██╔══██╗
   ╚████╔╝ ██║   ██║██║  ██║███████║
    ╚██╔╝  ██║   ██║██║  ██║██╔══██║
     ██║   ╚██████╔╝██████╔╝██║  ██║
     ╚═╝    ╚═════╝ ╚═════╝ ╚═╝  ╚═╝
  </pre>
  <strong>Personal AI Assistant — Infinite Memory</strong><br>
  <sub>Built with <a href="https://orcha.nl">ORCHA</a> · 6 agents · 2 workflow runs</sub>
</p>

<p align="center">
  <a href="https://orcha.nl"><img src="https://img.shields.io/badge/Built_with-ORCHA-blueviolet?style=for-the-badge" alt="Built with ORCHA"></a>
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License">
</p>

---

**Yoda** is a locally-running AI assistant that remembers everything you tell it, learns your preferences over time, reasons over a knowledge graph, and gives you full tool/computer access — all through a polished terminal interface.

> **This entire codebase was built by [ORCHA](https://orcha.nl)** — a multi-agent orchestration framework. 6 specialized AI agents collaborated across just 2 workflow runs to produce the full system. No human-written code.

---

## Quick Start

```bash
# Clone & install
git clone https://github.com/your-username/yoda.git
cd yoda
pip install -e .

# Set your API key (auto-saved to ~/.yoda/config.yaml)
export ANTHROPIC_API_KEY='sk-ant-...'

# Launch
yoda
```

That's it. Your API key is automatically persisted after the first run — you never need to export it again.

You can also set it inside Yoda:

```
🟢 You > /setup sk-ant-your-key-here
✓ API key saved to ~/.yoda/config.yaml
```

## What It Does

| Feature | Description |
|---------|-------------|
| **Infinite Memory** | Stores everything in ChromaDB vectors + BM25. Recalls relevant context automatically. |
| **Knowledge Graph** | Extracts entities and relationships from conversations. Multi-hop reasoning with NetworkX + SQLite. |
| **Token Optimization** | Context compression, sliding window, semantic caching, per-model cost tracking. |
| **Tool Access** | 42+ built-in tools: file ops, shell, web requests, browser automation, MCP client. |
| **MCP Server** | Deploy as an MCP server for Claude Code integration (SSE + stdio). |
| **Auto-Learning** | Extracts facts and preferences from every conversation. Gets smarter over time. |

## CLI Commands

```
🟢 You > /help
┌───────────┬─────────────────────────────────────┐
│ /help     │ Show available commands              │
│ /setup    │ Set API key: /setup <api-key>        │
│ /remember │ Store a memory: /remember <text>     │
│ /forget   │ Forget a memory: /forget <memory_id> │
│ /search   │ Search memories: /search <query>     │
│ /graph    │ Query knowledge graph: /graph <query>│
│ /status   │ Show agent status and stats          │
│ /cost     │ Show token usage & cost report       │
│ /claude   │ Generate CLAUDE.md from knowledge    │
│ /reset    │ Clear conversation history           │
│ /quit     │ Exit Yoda                            │
└───────────┴─────────────────────────────────────┘
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CLI / MCP Server                    │
│  Rich Terminal UI          MCP Server (SSE + stdio)  │
│  Slash commands            Tools: remember, recall,  │
│  Streaming Markdown        graph_query, preferences  │
├─────────────────────────────────────────────────────┤
│                    Orchestrator                       │
│  Init order · Plugin wiring · Context injectors      │
│  Signal handling · Graceful shutdown                  │
├─────────────────────────────────────────────────────┤
│                  Agent (ReAct Loop)                   │
│  Thought → Action → Observation · Streaming          │
├────────┬────────┬──────────┬────────────────────────┤
│ Memory │ Know.  │  Token   │       Tools            │
│  RAG   │ Graph  │   Opt    │   File/Shell/Web       │
│ Vector │ NX+SQL │  Cache   │   MCP/Computer         │
│  BM25  │ Reason │  Cost    │   42+ built-in         │
├────────┴────────┴──────────┴────────────────────────┤
│         LLM Providers (Anthropic/OpenAI/Ollama)      │
└─────────────────────────────────────────────────────┘
```

## Configuration

Yoda auto-creates `~/.yoda/config.yaml` when you set an API key. You can also edit it directly:

```yaml
provider:
  name: anthropic                    # or openai, local
  model: claude-sonnet-4-20250514
  api_key: sk-ant-...

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

Environment variables override config values:

```bash
export YODA_PROVIDER_MODEL=claude-opus-4-20250514
export YODA_PROVIDER_API_KEY=sk-ant-...
```

## MCP Server (Claude Code Integration)

Run Yoda as an MCP server so Claude Code can use its memory and knowledge graph:

```bash
# SSE transport
yoda --mcp --port 8765

# stdio transport (for Claude Code)
yoda --mcp-stdio
```

Add to `~/.claude/mcp_servers.json`:

```json
{
  "yoda": {
    "command": "yoda",
    "args": ["--mcp-stdio"]
  }
}
```

## Docker

```bash
docker compose up yoda          # Interactive CLI
docker compose --profile mcp up yoda-mcp  # MCP server
```

## Project Structure

```
yoda/
├── core/              # Agent, config, messages, providers, plugins
├── memory/            # RAG: ChromaDB/FAISS, BM25, hybrid retrieval
├── knowledge/         # Knowledge graph: NetworkX, entity extraction, reasoning
├── optimization/      # Token counting, compression, sliding window, cost tracking
├── tools/             # 42+ tools: file ops, shell, web, browser, MCP client
├── cli/               # Rich terminal UI, orchestrator, slash commands
└── mcp_server/        # MCP server (SSE + stdio transports)
```

## Tech Stack

- **Python 3.12+** — fully async
- **Pydantic v2** — config & message models
- **Rich** — terminal UI with streaming markdown
- **ChromaDB** — vector storage
- **NetworkX + SQLite** — knowledge graph
- **tiktoken** — token counting
- **aiohttp** — MCP SSE server

---

## Built with ORCHA

<p align="center">
  <a href="https://orcha.nl"><img src="https://img.shields.io/badge/Built_with-ORCHA-blueviolet?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0wIDE4Yy00LjQyIDAtOC0zLjU4LTgtOHMzLjU4LTggOC04IDggMy41OCA4IDgtMy41OCA4LTggOHoiLz48L3N2Zz4=" alt="ORCHA"></a>
</p>

This entire project was built by **[ORCHA](https://orcha.nl)** — a DAG-based multi-agent orchestration framework that coordinates specialized AI agents to build complete software systems.

**6 agents. 2 workflow runs. Zero human-written code.**

| Agent | Role |
|-------|------|
| Core Architect | ReAct agent loop, plugin system, LLM providers, config |
| RAG & Infinite Memory | Vector store, hybrid retrieval, chunking, embeddings |
| Knowledge Graph Builder | Entity extraction, multi-hop reasoning, visualization |
| Token Optimizer | Context compression, sliding window, semantic cache, cost tracking |
| Tool & Computer Access | File ops, shell, web, MCP client, browser automation |
| Integration Architect | CLI, MCP server, orchestrator, Docker, wiring |

Each agent worked on its assigned subsystem in dependency order, producing a fully integrated codebase that runs out of the box.

**[orcha.nl](https://orcha.nl)**

## License

MIT — see [LICENSE](LICENSE).
