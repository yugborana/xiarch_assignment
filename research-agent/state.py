"""
state.py — Shared state schema for the Research Agent graph.

This TypedDict is the single source of truth passed between every node.
LangGraph uses it for type checking and state management.
"""

from typing import TypedDict, Optional


class ResearchAgentState(TypedDict):
    # --- Input ---
    goal: str                              # Raw user query

    # --- Query Understanding ---
    parsed_query: dict                     # {topic, constraints, output_format, query_type}

    # --- Planning ---
    plan: list[str]                        # Visible planning trace (step descriptions)

    # --- Source Selection ---
    sources_strategy: str                  # "general_web" | "news" | "tech_docs" | "company_research"
    search_queries: list[str]              # Expanded queries for parallel search

    # --- Search ---
    search_results: list[dict]             # [{title, url, snippet, source_type}]

    # --- Fetch & Extract ---
    fetched_pages: dict                    # url -> {status, content, error, char_count}

    # --- Dedup & Relevance ---
    dedup_results: list[dict]              # Filtered, deduped chunks with source refs

    # --- Synthesis ---
    structured_summary: Optional[dict]     # The final structured report

    # --- Control Flow ---
    retry_count: int                       # Number of replan attempts so far
    max_retries: int                       # Maximum replan attempts (default 2)

    # --- Observability ---
    tool_call_log: list[dict]              # Every tool invocation: {tool, input, output, error, timestamp}
    errors: list[str]                      # Accumulated error messages

    # --- Memory ---
    memory_hit: Optional[dict]             # Cached result if query was previously researched
    memory_action: str                     # "use_cache" | "refresh" | "new_search"

    # --- Export ---
    export_path: Optional[str]            # Path to the exported report file
