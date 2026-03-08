"""Async ReAct agent loop with streaming, tool execution, and dynamic context injection."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Callable

from yoda.core.config import YodaConfig, load_config
from yoda.core.messages import (
    AssistantMessage,
    Conversation,
    Message,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from yoda.core.plugins import PluginRegistry, ToolSchema
from yoda.core.providers.base import LLMProvider, ProviderResponse, StreamChunk, create_provider

logger = logging.getLogger(__name__)

# Max iterations to prevent infinite tool-call loops
DEFAULT_MAX_ITERATIONS = 20


# ---------------------------------------------------------------------------
# Context injector protocol
# ---------------------------------------------------------------------------

ContextInjector = Callable[[Conversation], dict[str, Any]]
"""A callable that takes the current conversation and returns extra context."""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """Async ReAct agent with tool use, streaming, and plugin support.

    The core loop:
        1. Build context (system prompt + injected context + conversation)
        2. Call LLM provider
        3. If assistant wants to use tools → execute them, feed results back (Observe)
        4. Repeat until assistant responds without tool calls or max iterations hit
    """

    def __init__(
        self,
        config: YodaConfig | None = None,
        provider: LLMProvider | None = None,
        plugins: PluginRegistry | None = None,
    ) -> None:
        self.config = config or load_config()
        self.provider = provider or create_provider(self.config.provider)
        self.plugins = plugins or PluginRegistry(self.config)
        self.conversation = Conversation(system_prompt=self.config.system_prompt)
        self._context_injectors: list[ContextInjector] = []
        self._max_iterations = DEFAULT_MAX_ITERATIONS

        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    # -- Setup / teardown --------------------------------------------------

    async def initialize(self) -> None:
        """Discover and load all plugins."""
        self.plugins.discover()
        await self.plugins.load_all()
        logger.info(
            "Agent initialized with %d plugins, %d tools",
            len(self.plugins.plugins),
            len(self.plugins.all_tools()),
        )

    async def shutdown(self) -> None:
        """Unload all plugins."""
        await self.plugins.unload_all()

    # -- Context injection -------------------------------------------------

    def add_context_injector(self, injector: ContextInjector) -> None:
        """Register a function that injects dynamic context before each LLM call."""
        self._context_injectors.append(injector)

    def _build_system_prompt(self) -> str:
        """Build the full system prompt with injected context."""
        parts = [self.config.system_prompt]

        # Gather injected context
        for injector in self._context_injectors:
            try:
                ctx = injector(self.conversation)
                if ctx:
                    for key, value in ctx.items():
                        parts.append(f"\n<{key}>\n{value}\n</{key}>")
            except Exception:
                logger.exception("Context injector failed")

        # Add available tools description
        all_tools = self.plugins.all_tools()
        if all_tools:
            tool_desc = "\n\nAvailable tools:\n"
            for _plugin_name, tool in all_tools:
                params = ", ".join(
                    f"{p.name}: {p.type}" for p in tool.parameters
                )
                tool_desc += f"- {tool.name}({params}): {tool.description}\n"
            parts.append(tool_desc)

        return "\n".join(parts)

    # -- Tool execution ----------------------------------------------------

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call via the plugin registry."""
        result = self.plugins.find_tool(tool_call.name)
        if result is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                error=f"Unknown tool: {tool_call.name}",
            )

        plugin, _schema = result
        try:
            output = await plugin.execute(tool_call.name, tool_call.arguments)
            return ToolResult(tool_call_id=tool_call.id, output=output)
        except Exception as e:
            logger.exception("Tool execution failed: %s", tool_call.name)
            return ToolResult(tool_call_id=tool_call.id, error=str(e))

    async def _execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls concurrently."""
        tasks = [self._execute_tool(tc) for tc in tool_calls]
        return await asyncio.gather(*tasks)

    # -- Plugin hooks ------------------------------------------------------

    async def _run_user_hooks(self, content: str) -> str:
        """Run on_user_message hooks from all plugins."""
        for plugin in self.plugins.plugins.values():
            try:
                modified = await plugin.on_user_message(content)
                if modified is not None:
                    content = modified
            except Exception:
                logger.exception("Plugin hook on_user_message failed: %s", plugin.name)
        return content

    async def _run_assistant_hooks(self, content: str) -> str:
        """Run on_assistant_response hooks from all plugins."""
        for plugin in self.plugins.plugins.values():
            try:
                modified = await plugin.on_assistant_response(content)
                if modified is not None:
                    content = modified
            except Exception:
                logger.exception("Plugin hook on_assistant_response failed: %s", plugin.name)
        return content

    # -- Messages to provider format ---------------------------------------

    def _prepare_messages(self) -> list[dict[str, Any]]:
        """Build the message list for the LLM, with sliding window if needed."""
        system_prompt = self._build_system_prompt()
        messages = self.conversation.messages

        # Apply sliding window
        window = self.config.tokens.sliding_window_size
        if window and len(messages) > window:
            messages = messages[-window:]

        out: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            entry: dict[str, Any] = {"role": msg.role.value, "content": msg.content}
            if isinstance(msg, AssistantMessage) and msg.has_tool_calls:
                entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
            if isinstance(msg, ToolResultMessage):
                entry["tool_results"] = [tr.model_dump() for tr in msg.tool_results]
            out.append(entry)
        return out

    def _get_tool_schemas(self) -> list[ToolSchema] | None:
        all_tools = self.plugins.all_tools()
        if not all_tools:
            return None
        return [schema for _, schema in all_tools]

    # -- Track usage -------------------------------------------------------

    def _track_usage(self, usage: dict[str, int]) -> None:
        self.total_input_tokens += usage.get("input_tokens", 0)
        self.total_output_tokens += usage.get("output_tokens", 0)

    # -- Core ReAct loop ---------------------------------------------------

    async def chat(self, user_input: str) -> AssistantMessage:
        """Process a user message through the full ReAct loop. Returns final assistant message."""
        # Run plugin hooks on user input
        user_input = await self._run_user_hooks(user_input)
        self.conversation.add_user(user_input)

        tools = self._get_tool_schemas()

        for iteration in range(self._max_iterations):
            messages = self._prepare_messages()
            response: ProviderResponse = await self.provider.complete(messages, tools=tools)
            self._track_usage(response.usage)

            assistant_msg = response.to_assistant_message()
            self.conversation.add(assistant_msg)

            # If no tool calls, we're done (the "Act" phase is complete or skipped)
            if not assistant_msg.has_tool_calls:
                assistant_msg.content = await self._run_assistant_hooks(assistant_msg.content)
                return assistant_msg

            # Execute tools (Observe phase)
            logger.info(
                "ReAct iteration %d: executing %d tool(s)",
                iteration + 1,
                len(assistant_msg.tool_calls),
            )
            results = await self._execute_tools(assistant_msg.tool_calls)
            tool_msg = ToolResultMessage(
                content="\n".join(
                    f"[{r.tool_call_id}] {'ERROR: ' + r.error if r.is_error else str(r.output)}"
                    for r in results
                ),
                tool_results=results,
            )
            self.conversation.add(tool_msg)
            # Loop back — the model will reason over tool results

        # Safety: if we hit max iterations, return last assistant message
        logger.warning("Hit max ReAct iterations (%d)", self._max_iterations)
        return assistant_msg  # type: ignore[possibly-undefined]

    async def chat_stream(self, user_input: str) -> AsyncIterator[StreamChunk]:
        """Process user message with streaming response.

        NOTE: Streaming only applies to the final response (no tool calls).
        For intermediate ReAct steps with tool calls, we use non-streaming complete().
        """
        user_input = await self._run_user_hooks(user_input)
        self.conversation.add_user(user_input)
        tools = self._get_tool_schemas()

        for iteration in range(self._max_iterations):
            messages = self._prepare_messages()

            # First try non-streaming to check for tool calls
            response = await self.provider.complete(messages, tools=tools)
            self._track_usage(response.usage)

            if not response.tool_calls:
                # No tool calls — now stream the final response
                # Remove last user/tool messages from conversation temporarily to re-stream
                # Actually, yield the already-received content as a single chunk for simplicity,
                # or re-request with streaming:
                self.conversation.add(response.to_assistant_message())
                final_content = await self._run_assistant_hooks(response.content)
                yield StreamChunk(delta=final_content, finish_reason="stop", usage=response.usage)
                return

            # Tool calls — add to conversation and execute
            assistant_msg = response.to_assistant_message()
            self.conversation.add(assistant_msg)

            results = await self._execute_tools(assistant_msg.tool_calls)
            tool_msg = ToolResultMessage(
                content="\n".join(
                    f"[{r.tool_call_id}] {'ERROR: ' + r.error if r.is_error else str(r.output)}"
                    for r in results
                ),
                tool_results=results,
            )
            self.conversation.add(tool_msg)
            # Continue ReAct loop

        logger.warning("Hit max iterations in streaming mode")

    # -- Convenience -------------------------------------------------------

    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation = Conversation(system_prompt=self.config.system_prompt)

    @property
    def usage_summary(self) -> dict[str, int]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "conversation_messages": len(self.conversation),
        }
