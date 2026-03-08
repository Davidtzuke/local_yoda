"""Prompt optimization: dynamic system prompts, relevance-gated injection, template caching."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from yoda.optimization.tokens import TokenCounter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates with caching
# ---------------------------------------------------------------------------

@dataclass
class PromptTemplate:
    """Reusable prompt template with variable substitution and token caching."""

    name: str
    template: str
    variables: dict[str, str] = field(default_factory=dict)
    _token_cache: dict[str, int] = field(default_factory=dict, repr=False)

    def render(self, **kwargs: str) -> str:
        """Render template with variables."""
        merged = {**self.variables, **kwargs}
        result = self.template
        for key, value in merged.items():
            result = result.replace(f"{{{key}}}", value)
        return result

    def token_count(self, counter: TokenCounter, **kwargs: str) -> int:
        """Get token count for rendered template (cached)."""
        rendered = self.render(**kwargs)
        cache_key = hashlib.md5(rendered.encode()).hexdigest()
        if cache_key not in self._token_cache:
            self._token_cache[cache_key] = counter.count(rendered)
        return self._token_cache[cache_key]


# ---------------------------------------------------------------------------
# Context section with relevance scoring
# ---------------------------------------------------------------------------

@dataclass
class ContextSection:
    """A section of context that can be injected into the system prompt."""

    name: str
    content: str
    priority: float = 0.5  # 0.0 = low, 1.0 = critical
    max_tokens: int | None = None
    relevance_fn: Callable[[str], float] | None = None  # scores relevance to user query

    def relevance_score(self, query: str) -> float:
        """Compute relevance of this section to the current query."""
        if self.relevance_fn:
            return self.relevance_fn(query)
        return self.priority

    def truncate(self, counter: TokenCounter, max_tokens: int) -> str:
        """Truncate content to fit within token budget."""
        tokens = counter.count(self.content)
        if tokens <= max_tokens:
            return self.content
        # Binary search for appropriate truncation point
        chars = len(self.content)
        ratio = max_tokens / tokens
        cut = int(chars * ratio * 0.9)  # slight safety margin
        return self.content[:cut] + "\n...[truncated]"


# ---------------------------------------------------------------------------
# Prompt optimizer
# ---------------------------------------------------------------------------

class PromptOptimizer:
    """Manages dynamic system prompt construction with budget-aware injection.

    Features:
    - Template caching (avoid re-tokenizing static prompts)
    - Relevance-gated context injection (only inject relevant sections)
    - Priority-based budget allocation
    - Dynamic system prompt based on conversation state
    """

    def __init__(self, model: str = "default", system_budget_ratio: float = 0.2) -> None:
        self.counter = TokenCounter(model)
        self.system_budget_ratio = system_budget_ratio  # fraction of context for system prompt
        self._templates: dict[str, PromptTemplate] = {}
        self._sections: list[ContextSection] = []
        self._base_prompt: str = ""
        self._template_cache: dict[str, tuple[str, int]] = {}  # hash -> (rendered, tokens)

    # -- Template management -----------------------------------------------

    def register_template(self, template: PromptTemplate) -> None:
        self._templates[template.name] = template

    def get_template(self, name: str) -> PromptTemplate | None:
        return self._templates.get(name)

    def set_base_prompt(self, prompt: str) -> None:
        """Set the base system prompt."""
        self._base_prompt = prompt

    # -- Context section management ----------------------------------------

    def add_section(self, section: ContextSection) -> None:
        """Add a context section for potential injection."""
        # Replace if same name exists
        self._sections = [s for s in self._sections if s.name != section.name]
        self._sections.append(section)

    def remove_section(self, name: str) -> None:
        self._sections = [s for s in self._sections if s.name != name]

    def clear_sections(self) -> None:
        self._sections.clear()

    # -- Dynamic prompt building -------------------------------------------

    def build_system_prompt(
        self,
        user_query: str = "",
        max_tokens: int | None = None,
        relevance_threshold: float = 0.2,
    ) -> str:
        """Build optimized system prompt with relevance-gated section injection.

        Args:
            user_query: Current user query for relevance scoring.
            max_tokens: Max tokens for system prompt. Defaults to budget ratio.
            relevance_threshold: Minimum relevance score to include a section.

        Returns:
            Optimized system prompt string.
        """
        budget = max_tokens or int(
            self.counter.profile.effective_context * self.system_budget_ratio
        )

        # Start with base prompt
        parts: list[str] = [self._base_prompt] if self._base_prompt else []
        used = self.counter.count(self._base_prompt) if self._base_prompt else 0

        # Score and sort sections by relevance
        scored: list[tuple[float, ContextSection]] = []
        for section in self._sections:
            score = section.relevance_score(user_query)
            if score >= relevance_threshold:
                scored.append((score, section))

        # Sort by combined priority + relevance (descending)
        scored.sort(key=lambda x: x[0] * x[1].priority, reverse=True)

        # Inject sections that fit in budget
        for score, section in scored:
            section_budget = min(
                section.max_tokens or (budget - used),
                budget - used,
            )
            if section_budget <= 0:
                break

            content = section.truncate(self.counter, section_budget)
            tokens = self.counter.count(content)
            if used + tokens <= budget:
                parts.append(f"\n<{section.name}>\n{content}\n</{section.name}>")
                used += tokens

        return "\n".join(parts)

    def optimize_messages(
        self,
        messages: list[dict[str, Any]],
        user_query: str = "",
    ) -> list[dict[str, Any]]:
        """Replace or optimize the system message in a message list.

        Returns new message list with optimized system prompt.
        """
        # Extract max tokens budget for system prompt
        total_budget = self.counter.profile.effective_context
        non_system_tokens = sum(
            self.counter.count_message(m)
            for m in messages
            if m.get("role") != "system"
        )
        system_budget = min(
            int(total_budget * self.system_budget_ratio),
            total_budget - non_system_tokens - 100,  # leave headroom
        )

        optimized_prompt = self.build_system_prompt(
            user_query=user_query,
            max_tokens=max(system_budget, 200),
        )

        # Replace system message
        result = []
        system_replaced = False
        for msg in messages:
            if msg.get("role") == "system" and not system_replaced:
                result.append({"role": "system", "content": optimized_prompt})
                system_replaced = True
            else:
                result.append(msg)

        if not system_replaced:
            result.insert(0, {"role": "system", "content": optimized_prompt})

        return result

    # -- Stats -------------------------------------------------------------

    @property
    def section_count(self) -> int:
        return len(self._sections)

    @property
    def template_count(self) -> int:
        return len(self._templates)

    def cache_stats(self) -> dict[str, int]:
        total_cached = sum(len(t._token_cache) for t in self._templates.values())
        return {
            "templates": len(self._templates),
            "sections": len(self._sections),
            "cached_renders": total_cached,
        }
