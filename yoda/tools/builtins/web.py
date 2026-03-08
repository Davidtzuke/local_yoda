"""Web tools: HTTP requests, scraping, URL fetching."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urljoin, urlparse

from yoda.tools.registry import ToolPermission, tool


def register_web_tools() -> None:
    """No-op — importing this module registers tools via decorators."""


@tool(
    name="http_request",
    permission=ToolPermission.READ,
    category="web",
    tags=["http", "api", "request"],
    timeout=30.0,
    retries=2,
)
async def http_request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: str = "",
    timeout: float = 15.0,
) -> str:
    """Make an HTTP request and return the response.

    Args:
        url: URL to request.
        method: HTTP method (GET, POST, PUT, DELETE, PATCH).
        headers: Request headers.
        body: Request body (for POST/PUT/PATCH).
        timeout: Request timeout in seconds.
    """
    import httpx

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        kwargs: dict[str, Any] = {
            "method": method.upper(),
            "url": url,
            "headers": headers or {},
        }
        if body and method.upper() in ("POST", "PUT", "PATCH"):
            kwargs["content"] = body

        response = await client.request(**kwargs)

    # Format response
    parts = [
        f"Status: {response.status_code}",
        f"URL: {response.url}",
    ]

    # Content type handling
    content_type = response.headers.get("content-type", "")
    if "json" in content_type:
        try:
            data = response.json()
            parts.append(f"Body:\n{json.dumps(data, indent=2)[:10000]}")
        except Exception:
            parts.append(f"Body:\n{response.text[:10000]}")
    else:
        parts.append(f"Body:\n{response.text[:10000]}")

    return "\n".join(parts)


@tool(
    name="fetch_webpage",
    permission=ToolPermission.READ,
    category="web",
    tags=["web", "scrape", "fetch"],
    timeout=30.0,
    retries=2,
)
async def fetch_webpage(
    url: str,
    extract_text: bool = True,
    max_length: int = 10000,
) -> str:
    """Fetch a webpage and optionally extract readable text.

    Args:
        url: URL to fetch.
        extract_text: If True, extract readable text (strip HTML).
        max_length: Maximum content length to return.
    """
    import httpx

    async with httpx.AsyncClient(
        timeout=15.0,
        follow_redirects=True,
        headers={"User-Agent": "Yoda/1.0 (Personal AI Assistant)"},
    ) as client:
        response = await client.get(url)

    if not extract_text:
        content = response.text[:max_length]
        return f"Status: {response.status_code}\nURL: {response.url}\n\n{content}"

    # Simple HTML to text extraction
    text = _html_to_text(response.text)
    if len(text) > max_length:
        text = text[:max_length] + "\n... (truncated)"

    return f"URL: {response.url}\nTitle: {_extract_title(response.text)}\n\n{text}"


@tool(
    name="extract_links",
    permission=ToolPermission.READ,
    category="web",
    tags=["web", "links", "extract"],
    timeout=15.0,
)
async def extract_links(url: str, max_links: int = 50) -> str:
    """Extract all links from a webpage.

    Args:
        url: URL to fetch links from.
        max_links: Maximum number of links to return.
    """
    import httpx

    async with httpx.AsyncClient(
        timeout=15.0,
        follow_redirects=True,
        headers={"User-Agent": "Yoda/1.0"},
    ) as client:
        response = await client.get(url)

    # Extract href attributes
    links: list[str] = []
    for match in re.finditer(r'href=["\']([^"\']+)["\']', response.text):
        href = match.group(1)
        # Resolve relative URLs
        absolute = urljoin(str(response.url), href)
        if absolute.startswith(("http://", "https://")) and absolute not in links:
            links.append(absolute)
            if len(links) >= max_links:
                break

    if not links:
        return "No links found"
    return "\n".join(links)


@tool(
    name="download_file",
    permission=ToolPermission.WRITE,
    category="web",
    tags=["web", "download"],
    timeout=120.0,
    requires_approval=True,
)
async def download_file(url: str, save_path: str) -> str:
    """Download a file from a URL.

    Args:
        url: URL to download from.
        save_path: Local path to save the file.
    """
    import httpx
    from pathlib import Path

    dest = Path(save_path).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

    dest.write_bytes(response.content)
    size = len(response.content)
    return f"Downloaded {size} bytes to {dest}"


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _html_to_text(html: str) -> str:
    """Simple HTML to text conversion without external deps."""
    # Remove script/style blocks
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Convert common block elements to newlines
    text = re.sub(r"<(br|p|div|h[1-6]|li|tr)[^>]*>", "\n", text, flags=re.IGNORECASE)
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode common entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&quot;", '"')
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _extract_title(html: str) -> str:
    """Extract the page title from HTML."""
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else "(no title)"
