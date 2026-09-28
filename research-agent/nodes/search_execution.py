"""
nodes/search_execution.py — Parallel Search Execution node.

Runs multiple search queries and aggregates results.
Handles zero-result self-correction by broadening queries.
"""

import time
import logging
from tools.web_search import search_multiple_queries, web_search

logger = logging.getLogger(__name__)


def search_execution_node(state: dict) -> dict:
    """Execute search queries and collect results."""
    search_queries = state.get("search_queries", [])
    strategy = state.get("sources_strategy", "general_web")
    start_time = time.time()

    tool_log_entry = {
        "tool": "web_search",
        "input": {"queries": search_queries, "strategy": strategy},
        "output": None,
        "error": None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_ms": 0,
    }

    # Determine time filter based on strategy
    from nodes.source_selection import STRATEGIES
    strategy_config = STRATEGIES.get(strategy, STRATEGIES["general_web"])
    time_filter = strategy_config.get("time_filter")
    max_results = strategy_config.get("max_results_per_query", 5)

    print(f"[SEARCH] Searching with {len(search_queries)} queries...", flush=True)

    # Execute searches
    results = search_multiple_queries(
        queries=search_queries,
        strategy=strategy,
        max_results_per_query=max_results,
        time_filter=time_filter,
    )

    # Self-correction: zero results -> broaden and retry
    errors = list(state.get("errors", []))
    if len(results) == 0:
        logger.warning("Zero search results -- attempting broader search")
        errors.append("Initial search returned zero results, broadening query")
        print("[WARN] Zero results -- broadening search terms...", flush=True)

        # Strip qualifiers and try simpler query
        topic = state.get("parsed_query", {}).get("topic", state["goal"])
        broader_results = web_search(topic, strategy="general_web", max_results=8)

        if len(broader_results) == 0:
            # Try even simpler
            words = topic.split()
            if len(words) > 2:
                simple_query = " ".join(words[:3])
                broader_results = web_search(simple_query, strategy="general_web", max_results=8)

        results = broader_results
        if len(results) == 0:
            errors.append("Search returned zero results even after broadening query")

    duration_ms = (time.time() - start_time) * 1000
    tool_log_entry["output"] = {
        "result_count": len(results),
        "urls": [r["url"] for r in results[:5]],
    }
    tool_log_entry["duration_ms"] = duration_ms

    print(f"   Found {len(results)} unique results", flush=True)

    return {
        "search_results": results,
        "tool_call_log": state.get("tool_call_log", []) + [tool_log_entry],
        "errors": errors,
    }
