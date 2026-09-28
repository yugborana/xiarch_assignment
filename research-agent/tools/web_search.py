"""
tools/web_search.py — Tavily web search tool.

Uses Tavily's search API which is designed for AI agents:
- Returns relevant, high-quality results
- Includes extracted page content directly (reducing need for separate fetch)
- Free tier: 1000 searches/month
"""

import os
import time
import logging
from typing import Optional
from tavily import TavilyClient

logger = logging.getLogger(__name__)


def _get_client() -> TavilyClient:
    """Get a Tavily client, using the API key from environment."""
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY not set. Get a free key at https://tavily.com"
        )
    return TavilyClient(api_key=api_key)


def web_search(
    query: str,
    strategy: str = "general_web",
    max_results: int = 8,
    time_filter: Optional[str] = None,
) -> list[dict]:
    """Search the web using Tavily.

    Args:
        query: Search query string.
        strategy: One of "general_web", "news", "tech_docs", "company_research".
        max_results: Maximum number of results to return.
        time_filter: Not used by Tavily (kept for interface compatibility).

    Returns:
        List of dicts: [{title, url, snippet, source_type, raw_content}].
        Returns empty list on failure (never raises).
    """
    results = []

    try:
        client = _get_client()

        # Tavily search_depth: "basic" is fast, "advanced" extracts full content
        search_depth = "advanced" if strategy in ("tech_docs", "company_research") else "basic"

        # Use news topic for news strategy
        topic = "news" if strategy == "news" else "general"

        response = client.search(
            query=query,
            search_depth=search_depth,
            topic=topic,
            max_results=max_results,
            include_raw_content=True,
        )

        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
                "source_type": strategy,
                "raw_content": r.get("raw_content", ""),  # Full extracted text
            })

    except Exception as e:
        error_msg = str(e).lower()
        if any(kw in error_msg for kw in ("quota", "limit", "429", "exceeded", "rate")):
            logger.error(f"Tavily API quota/rate limit hit for query '{query}': {e}")
            print(f"[WARN] Tavily API limit reached. Check your plan at https://tavily.com", flush=True)
        else:
            logger.error(f"Tavily search failed for query '{query}': {e}")
        return []

    logger.info(f"Tavily search returned {len(results)} results for '{query}' (strategy={strategy})")
    return results


def search_multiple_queries(
    queries: list[str],
    strategy: str = "general_web",
    max_results_per_query: int = 5,
    time_filter: Optional[str] = None,
) -> list[dict]:
    """Run multiple search queries and merge results.

    Returns:
        Merged list of search results from all queries.
    """
    all_results = []
    for query in queries:
        results = web_search(query, strategy, max_results_per_query, time_filter)
        all_results.extend(results)

    # URL-level dedup (semantic dedup is in dedup_filter node)
    seen_urls = set()
    unique_results = []
    for r in all_results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            unique_results.append(r)

    return unique_results
