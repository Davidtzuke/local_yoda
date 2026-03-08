"""Plugin integration for token optimization — exposes tools and hooks into the agent loop."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from yoda.core.config import YodaConfig
from yoda.core.messages import Conversation
from yoda.core.plugins import Plugin, ToolParameter, ToolSchema
from yoda.optimization.cache import SemanticCache
from yoda.optimization.compressor import ContextCompressor
from yoda.optimization.cost import CostTracker
from yoda.optimization.prompt import ContextSection, PromptOptimizer
from yoda.optimization.tokens import TokenCounter
from yoda.optimization.window import Priority, SlidingWindow

logger = logging.getLogger(__name__)


class TokenOptimizerPlugin(Plugin):
    """Plugin that integrates all token optimization components into the agent.

    Provides tools:
    - token_count: Count tokens in text
    - token_budget: Show current token budget status
    - cost_report: Show usage cost report
    - cache_stats: Show cache statistics
    - set_budget_alert: Configure a spending alert

    Hooks:
    - on_context_build: Compress context and optimize prompts
    - on_assistant_response: Track costs, cache responses
    """

    name = "token_optimizer"
    version = "0.1.0"
    description = "Token counting, context compression, cost tracking, and caching"

    def __init__(self, config: YodaConfig) -> None:
        super().__init__(config)
        model = config.provider.model
        tc = config.tokens

        self.token_counter = TokenCounter(model)
        self.compressor = ContextCompressor(
            model=model,
            max_tokens=tc.max_context_tokens,
        )
        self.prompt_optimizer = PromptOptimizer(model=model)
        self.window = SlidingWindow(
            model=model,
            max_tokens=tc.max_context_tokens,
            max_messages=tc.sliding_window_size * 2,  # headroom
        )
        self.cache = SemanticCache(
            persist_path=f"{config.data_dir}/cache.db",
            model=model,
        )
        self.cost_tracker = CostTracker(
            persist_path=f"{config.data_dir}/cost_log.json",
            model=model,
        )
        self._last_query: str = ""
        self._request_start: float = 0.0

    # -- Lifecycle ---------------------------------------------------------

    async def on_load(self) -> None:
        await super().on_load()
        self.cache.initialize()
        self.cost_tracker.initialize()

        # Set up default budget alert
        if self.config.tokens.cost_tracking:
            self.cost_tracker.add_alert(
                "daily_default",
                threshold_usd=5.0,
                period="daily",
            )

        self.prompt_optimizer.set_base_prompt(self.config.system_prompt)
        logger.info("Token optimizer plugin loaded")

    async def on_unload(self) -> None:
        self.cost_tracker.save()
        self.cache.close()
        await super().on_unload()

    # -- Tools -------------------------------------------------------------

    def tools(self) -> list[ToolSchema]:
        return [
            ToolSchema(
                name="token_count",
                description="Count tokens in a given text for the current model",
                parameters=[
                    ToolParameter(name="text", type="string", description="Text to count tokens for", required=True),
                ],
                returns="integer",
            ),
            ToolSchema(
                name="token_budget",
                description="Show current token budget status (used, remaining, utilization)",
                parameters=[],
                returns="object",
            ),
            ToolSchema(
                name="cost_report",
                description="Show API usage cost report",
                parameters=[
                    ToolParameter(name="period", type="string", description="Report period: session, daily, total", required=False, default="session"),
                ],
                returns="object",
            ),
            ToolSchema(
                name="cache_stats",
                description="Show semantic cache statistics",
                parameters=[],
                returns="object",
            ),
            ToolSchema(
                name="set_budget_alert",
                description="Set a spending alert threshold",
                parameters=[
                    ToolParameter(name="name", type="string", description="Alert name", required=True),
                    ToolParameter(name="threshold_usd", type="number", description="Dollar threshold", required=True),
                    ToolParameter(name="period", type="string", description="Period: daily, weekly, monthly, total", required=False, default="daily"),
                ],
                returns="string",
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "token_count":
                text = arguments.get("text", "")
                count = self.token_counter.count(text)
                return {"tokens": count, "model": self.token_counter.profile.name}

            case "token_budget":
                return {
                    "budget_limit": self.token_counter.budget_limit,
                    "budget_used": self.token_counter.budget_used,
                    "budget_remaining": self.token_counter.budget_remaining,
                    "utilization": f"{self.token_counter.budget_utilization:.1%}",
                    "window": self.window.stats(),
                }

            case "cost_report":
                period = arguments.get("period", "session")
                if period == "session":
                    report = self.cost_tracker.session_report()
                elif period == "daily":
                    report = self.cost_tracker.daily_report()
                else:
                    report = self.cost_tracker.total_report()
                report["model_breakdown"] = self.cost_tracker.model_breakdown()
                return report

            case "cache_stats":
                return self.cache.stats()

            case "set_budget_alert":
                name = arguments["name"]
                threshold = float(arguments["threshold_usd"])
                period = arguments.get("period", "daily")
                self.cost_tracker.add_alert(name, threshold, period)
                return f"Alert '{name}' set: ${threshold:.2f} {period}"

            case _:
                return {"error": f"Unknown tool: {tool_name}"}

    # -- Agent hooks -------------------------------------------------------

    async def on_user_message(self, content: str) -> str | None:
        """Track the user query for relevance scoring and cache lookup."""
        self._last_query = content
        self._request_start = time.time()

        # Check cache
        cached = self.cache.get(content, self.config.provider.model)
        if cached:
            logger.info("Cache hit for query (saved %d tokens)", cached.tokens_saved)
            # Return None — we don't replace the message, but we could
            # use this for fast responses in the future

        return None

    async def on_assistant_response(self, content: str) -> str | None:
        """Track costs and cache response."""
        if self.config.tokens.cost_tracking and self._request_start > 0:
            latency = (time.time() - self._request_start) * 1000
            # The actual token tracking happens via the agent's _track_usage,
            # but we can record the response for caching
            self._request_start = 0.0

        # Cache the response
        if self._last_query and content:
            self.cache.put(
                self._last_query,
                content,
                self.config.provider.model,
            )

        return None

    async def on_context_build(self, context: dict[str, Any]) -> dict[str, Any]:
        """Inject optimization stats into context."""
        if self.config.tokens.cost_tracking:
            context["cost_info"] = (
                f"Session cost: ${self.cost_tracker.session_cost:.4f} | "
                f"Requests: {self.cost_tracker.request_count}"
            )
        return context

    # -- Context injector for the agent ------------------------------------

    def get_context_injector(self):
        """Return a context injector function for the agent."""
        def injector(conversation: Conversation) -> dict[str, Any]:
            ctx: dict[str, Any] = {}
            budget = self.token_counter
            if budget.budget_utilization > 0.7:
                ctx["token_warning"] = (
                    f"Context is {budget.budget_utilization:.0%} full. "
                    f"Remaining: {budget.budget_remaining:,} tokens."
                )
            return ctx
        return injector

    # -- Public API for agent integration ----------------------------------

    def compress_messages(
        self, messages: list[dict[str, Any]], target_tokens: int | None = None
    ) -> list[dict[str, Any]]:
        """Compress messages to fit within budget."""
        result = self.compressor.compress(messages, target_tokens)
        if result.tokens_saved > 0:
            logger.info(
                "Compressed context: %d → %d tokens (saved %d, ratio %.0f%%)",
                result.original_tokens,
                result.compressed_tokens,
                result.tokens_saved,
                result.ratio * 100,
            )
        return result.messages

    def track_usage(
        self, input_tokens: int, output_tokens: int, model: str | None = None
    ) -> None:
        """Record token usage for cost tracking."""
        self.token_counter.consume(input_tokens + output_tokens)
        if self.config.tokens.cost_tracking:
            self.cost_tracker.record(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
            )
