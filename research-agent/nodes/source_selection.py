"""
nodes/source_selection.py — Source Selection node.

Decides the search strategy based on query type.
This is the "autonomous source selection" bonus feature.
"""

import time
import logging

logger = logging.getLogger(__name__)

# Strategy configurations
STRATEGIES = {
    "news": {
        "description": "News-style search with recency weighting",
        "search_type": "news",
        "time_filter": "w",  # last week
        "max_results_per_query": 6,
        "query_modifiers": ["latest", "recent developments", "news"],
    },
    "tech_docs": {
        "description": "Technical documentation and research-focused search",
        "search_type": "general_web",
        "time_filter": None,
        "max_results_per_query": 6,
        "query_modifiers": ["technical overview", "research paper", "documentation"],
    },
    "company_research": {
        "description": "Company/competitive analysis search",
        "search_type": "general_web",
        "time_filter": "y",  # last year
        "max_results_per_query": 6,
        "query_modifiers": ["company analysis", "competitive landscape", "market position"],
    },
    "general_web": {
        "description": "General web search",
        "search_type": "general_web",
        "time_filter": None,
        "max_results_per_query": 6,
        "query_modifiers": [],
    },
}


def source_selection_node(state: dict) -> dict:
    """Select search strategy based on query type."""
    parsed_query = state.get("parsed_query", {})
    query_type = parsed_query.get("query_type", "general_web")
    search_queries = state.get("search_queries", [parsed_query.get("topic", state["goal"])])
    start_time = time.time()

    tool_log_entry = {
        "tool": "source_selection",
        "input": {"query_type": query_type, "original_queries": search_queries},
        "output": None,
        "error": None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_ms": 0,
    }

    strategy = STRATEGIES.get(query_type, STRATEGIES["general_web"])

    # Optionally expand queries with strategy-specific modifiers
    expanded_queries = list(search_queries)  # Keep originals
    modifiers = strategy["query_modifiers"]
    if modifiers and len(search_queries) > 0:
        base_topic = search_queries[0]
        # Add one modifier-enhanced query
        expanded_queries.append(f"{base_topic} {modifiers[0]}")

    duration_ms = (time.time() - start_time) * 1000
    tool_log_entry["output"] = {
        "strategy": query_type,
        "description": strategy["description"],
        "expanded_queries": expanded_queries,
    }
    tool_log_entry["duration_ms"] = duration_ms

    print(f"[SOURCE] Strategy: {strategy['description']} (type={query_type})", flush=True)
    print(f"   Queries: {expanded_queries}", flush=True)

    return {
        "sources_strategy": query_type,
        "search_queries": expanded_queries,
        "tool_call_log": state.get("tool_call_log", []) + [tool_log_entry],
    }
