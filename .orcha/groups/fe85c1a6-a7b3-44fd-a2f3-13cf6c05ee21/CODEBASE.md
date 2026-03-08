## About This Project
Yoda is a locally-running personal AI assistant with infinite memory (RAG + vector store), knowledge graph reasoning, token optimization, full tool/computer access, and MCP server integration for Claude Code. It uses a ReAct agent loop with plugin architecture.

## Tech Stack
Python 3.12+, asyncio throughout. Pydantic v2 for config/models. Rich + Click for CLI. ChromaDB/FAISS for vector storage, NetworkX + SQLite for knowledge graph. tiktoken for token counting, aiohttp for MCP SSE server. LLM providers: Anthropic, OpenAI, Ollama. Docker for deployment.

## What This Branch Does
Full build of the Yoda agent across 6 specialized areas: core agent framework (ReAct loop, providers, plugins), RAG memory system (hybrid retrieval, chunking, embeddings), knowledge graph (entity extraction, multi-hop reasoning, visualization), token optimization (compression, cost tracking, caching), tool access (file ops, shell, web, MCP client, computer control), and integration layer (Rich CLI, MCP server, orchestrator, Docker, tests).

## Key Files
- `yoda/core/agent.py` — ReAct agent loop with streaming and plugin hooks
- `yoda/cli/orchestrator.py` — Wires all subsystems with correct init/shutdown order
- `yoda/cli/app.py` — Rich terminal UI with slash commands
- `yoda/mcp_server/server.py` — MCP server (SSE + stdio) for Claude Code
- `yoda/memory/manager.py` — Memory lifecycle with hybrid retrieval
- `yoda/knowledge/graph.py` — NetworkX knowledge graph with SQLite persistence
- `yoda/optimization/plugin.py` — Token optimization integration
- `yoda/tools/plugin.py` — Tool access with MCP client
- `tests/test_integration.py` — 21 integration tests
