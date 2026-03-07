"""Web search tool using DuckDuckGo (no API key needed)."""

from __future__ import annotations

import asyncio
import json
import urllib.parse
import urllib.request

from yoda.agent.tools import ToolDef


async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo Instant Answer API."""

    def _search() -> str:
        url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
            {"q": query, "format": "json", "no_redirect": "1", "no_html": "1"}
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Yoda/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            return f"Search failed: {exc}"

        results: list[str] = []

        # Abstract (instant answer)
        if data.get("Abstract"):
            results.append(f"**{data.get('Heading', query)}**\n{data['Abstract']}")
            if data.get("AbstractURL"):
                results.append(f"Source: {data['AbstractURL']}")

        # Related topics
        for topic in (data.get("RelatedTopics") or [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                line = topic["Text"]
                if topic.get("FirstURL"):
                    line += f"\n  -> {topic['FirstURL']}"
                results.append(line)

        if not results:
            return f"No results found for: {query}"
        return "\n\n".join(results)

    return await asyncio.to_thread(_search)


def get_websearch_tools() -> list[ToolDef]:
    """Return web search tool definitions."""
    return [
        ToolDef(
            name="web_search",
            description="Search the web for information. Args: query (str), max_results (int, default=5)",
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Max results (default 5)",
                },
            },
            handler=web_search,
        ),
    ]
