# Chat UI Agent Progress

## Checklist

- [x] Create `yoda/web/__init__.py`
- [x] Create `yoda/web/app.py` with FastAPI app: `POST /chat` (SSE streaming), `GET /` (serves SPA), conversation CRUD endpoints
- [x] Create `yoda/web/static/index.html` - minimal chat UI with message bubbles, input, sidebar
- [x] Use SSE for streaming responses from the agent
- [x] Define `AgentProtocol` (runtime_checkable Protocol class) so Integration can plug in the real agent
- [x] Note dependencies: `fastapi`, `uvicorn`

## Agent Updates

- Created `yoda/web/app.py` (171 lines) with:
  - `AgentProtocol` - runtime_checkable Protocol with `async stream(message, conversation_id) -> AsyncIterator[str]`
  - `StubAgent` - echo-based placeholder for dev/testing
  - `ConversationStore` - in-memory conversation management
  - `create_app(agent=None)` factory function for Integration to wire in the real agent
  - SSE streaming via `POST /chat`
  - Conversation CRUD: `POST /GET /conversations`, `GET /conversations/{cid}`
- Created `yoda/web/static/index.html` (180 lines) with:
  - Dark/light theme toggle (persisted in localStorage)
  - Conversation sidebar with new chat button
  - Message bubbles (user right-aligned, assistant left-aligned)
  - SSE streaming response rendering
  - Code block rendering with copy button
  - Inline code rendering
  - Auto-resizing textarea, Enter to send, Shift+Enter for newline
  - Responsive (sidebar hidden on mobile)
- Total: 351 lines HTML/CSS/JS, well under 500 limit
- Dependencies: `fastapi`, `uvicorn`

## Notes for Integration Agent

- Import `create_app` from `yoda.web.app` and pass your agent implementation
- Agent must implement `AgentProtocol`: `async def stream(self, message: str, conversation_id: str) -> AsyncIterator[str]`
- Run with: `uvicorn.run(create_app(agent), host="127.0.0.1", port=8000)`
- The `ConversationStore` is in-memory; replace if persistent storage is needed
