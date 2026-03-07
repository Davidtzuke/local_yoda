"""FastAPI web application for Yoda chat interface."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Protocol, runtime_checkable

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Agent protocol – Integration agent will supply a real implementation
# ---------------------------------------------------------------------------

@runtime_checkable
class AgentProtocol(Protocol):
    """Minimal contract the agent core must satisfy."""

    async def stream(self, message: str, conversation_id: str) -> AsyncIterator[str]:
        """Yield response tokens for *message* in the given conversation."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Stub agent (echoes input token-by-token) – replaced at wiring time
# ---------------------------------------------------------------------------

class StubAgent:
    """Placeholder agent that echoes the user message word-by-word."""

    async def stream(self, message: str, conversation_id: str) -> AsyncIterator[str]:
        words = f"Echo (conv {conversation_id[:8]}): {message}".split()
        for w in words:
            yield w + " "
            await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# In-memory conversation store
# ---------------------------------------------------------------------------

@dataclass
class Conversation:
    id: str
    title: str
    messages: list[dict] = field(default_factory=list)


class ConversationStore:
    """Simple in-memory store. Integration may swap for persistent storage."""

    def __init__(self) -> None:
        self._convos: dict[str, Conversation] = {}

    def create(self, title: str = "New chat") -> Conversation:
        cid = uuid.uuid4().hex[:12]
        c = Conversation(id=cid, title=title)
        self._convos[cid] = c
        return c

    def get(self, cid: str) -> Conversation | None:
        return self._convos.get(cid)

    def list_all(self) -> list[dict]:
        return [{"id": c.id, "title": c.title} for c in self._convos.values()]

    def add_message(self, cid: str, role: str, content: str) -> None:
        c = self._convos.get(cid)
        if c:
            c.messages.append({"role": role, "content": content})
            if role == "user" and c.title == "New chat":
                c.title = content[:40]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(agent: AgentProtocol | None = None) -> FastAPI:
    """Build and return the FastAPI application.

    Parameters
    ----------
    agent:
        An object satisfying ``AgentProtocol``. Falls back to ``StubAgent``
        when *None* (useful for development / testing).
    """

    _agent: AgentProtocol = agent or StubAgent()
    store = ConversationStore()
    static_dir = Path(__file__).parent / "static"

    app = FastAPI(title="Yoda Chat")
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # -- Serve the SPA -------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html = (static_dir / "index.html").read_text()
        return HTMLResponse(content=html)

    # -- Conversation CRUD ---------------------------------------------------

    @app.post("/conversations")
    async def create_conversation() -> dict:
        c = store.create()
        return {"id": c.id, "title": c.title}

    @app.get("/conversations")
    async def list_conversations() -> list[dict]:
        return store.list_all()

    @app.get("/conversations/{cid}")
    async def get_conversation(cid: str) -> dict:
        c = store.get(cid)
        if c is None:
            return {"error": "not found"}
        return {"id": c.id, "title": c.title, "messages": c.messages}

    # -- Chat (SSE streaming) ------------------------------------------------

    @app.post("/chat")
    async def chat(req: ChatRequest) -> StreamingResponse:
        cid = req.conversation_id
        if cid is None or store.get(cid) is None:
            c = store.create()
            cid = c.id

        store.add_message(cid, "user", req.message)

        async def event_stream() -> AsyncIterator[str]:
            # Send conversation id first so the client knows which conv this is
            yield f"event: conv_id\ndata: {cid}\n\n"
            chunks: list[str] = []
            async for token in _agent.stream(req.message, cid):
                chunks.append(token)
                yield f"data: {token}\n\n"
            full = "".join(chunks)
            store.add_message(cid, "assistant", full)
            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


# ---------------------------------------------------------------------------
# Convenience: run directly with ``python -m yoda.web.app``
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    application = create_app()
    uvicorn.run(application, host="127.0.0.1", port=8000)
