"""
nodes/dedup_filter.py — Dedup & Relevance Filter node.

Uses embedding similarity to:
1. Remove near-duplicate content across sources
2. Filter out content that's not relevant to the query
"""

import time
import logging
from tools.page_fetch import STATUS_OK
from tools.similarity import filter_by_relevance, deduplicate_items

logger = logging.getLogger(__name__)


def dedup_filter_node(state: dict) -> dict:
    """Deduplicate and filter fetched content by relevance."""
    fetched_pages = state.get("fetched_pages", {})
    parsed_query = state.get("parsed_query", {})
    goal = state.get("goal", "")
    search_results = state.get("search_results", [])
    start_time = time.time()

    tool_log_entry = {
        "tool": "embedding_similarity",
        "input": {"page_count": len(fetched_pages)},
        "output": None,
        "error": None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_ms": 0,
    }

    # Build items from successfully fetched pages
    items = []
    # Build a URL -> search result mapping for titles/snippets
    url_to_result = {r["url"]: r for r in search_results}

    for url, page_data in fetched_pages.items():
        if page_data["status"] == STATUS_OK and page_data.get("content"):
            search_info = url_to_result.get(url, {})
            items.append({
                "url": url,
                "title": page_data.get("title") or search_info.get("title", ""),
                "content": page_data["content"],
                "snippet": search_info.get("snippet", ""),
                "char_count": page_data["char_count"],
                "source_type": search_info.get("source_type", "general_web"),
            })

    print(f"[FILTER] Filtering {len(items)} pages for relevance and duplicates...", flush=True)

    if len(items) == 0:
        duration_ms = (time.time() - start_time) * 1000
        tool_log_entry["output"] = {"items_in": 0, "items_out": 0}
        tool_log_entry["duration_ms"] = duration_ms
        tool_log_entry["error"] = "No content to filter"

        return {
            "dedup_results": [],
            "tool_call_log": state.get("tool_call_log", []) + [tool_log_entry],
            "errors": state.get("errors", []) + ["No usable content after fetch -- all sources failed or were empty"],
        }

    # Step 1: Relevance filtering
    query_text = parsed_query.get("topic", goal)
    relevant_items = filter_by_relevance(
        query=query_text,
        items=items,
        text_key="content",
        threshold=0.25,  # Balanced threshold -- filters junk without losing borderline sources
    )

    # Step 2: Deduplication
    deduped_items = deduplicate_items(
        items=relevant_items,
        text_key="content",
        threshold=0.85,
    )

    duration_ms = (time.time() - start_time) * 1000
    tool_log_entry["output"] = {
        "items_in": len(items),
        "after_relevance_filter": len(relevant_items),
        "after_dedup": len(deduped_items),
    }
    tool_log_entry["duration_ms"] = duration_ms

    print(f"   {len(items)} -> {len(relevant_items)} (relevance) -> {len(deduped_items)} (dedup)", flush=True)

    return {
        "dedup_results": deduped_items,
        "tool_call_log": state.get("tool_call_log", []) + [tool_log_entry],
    }
