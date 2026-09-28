"""
graph.py — LangGraph state machine for the Autonomous Research Agent.

Wires all nodes into a directed graph with conditional edges for:
- Memory cache hits (skip search if already researched)
- Coverage-based replanning (rerun search if too few sources survive)
- Max retry limits
"""

import logging
from langgraph.graph import StateGraph, END

from state import ResearchAgentState
from nodes.query_understanding import query_understanding_node
from nodes.planning import planning_node
from nodes.source_selection import source_selection_node
from nodes.search_execution import search_execution_node
from nodes.fetch_extract import fetch_extract_node
from nodes.dedup_filter import dedup_filter_node
from nodes.synthesis import synthesis_node
from nodes.export_node import export_node
from tools.memory import check_memory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory check node
# ---------------------------------------------------------------------------

def memory_check_node(state: dict) -> dict:
    """Check if we've researched this topic before."""
    import time

    goal = state["goal"]
    start_time = time.time()

    # Respect --no-cache flag
    if state.get("memory_action") == "skip_cache":
        print("[MEMORY] Cache bypassed (--no-cache)", flush=True)
        return {
            "memory_hit": None,
            "memory_action": "new_search",
            "tool_call_log": state.get("tool_call_log", []) + [{
                "tool": "memory_check",
                "input": {"goal": goal},
                "output": {"hit": False, "skipped": True},
                "error": None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_ms": 0,
            }],
        }

    tool_log_entry = {
        "tool": "memory_check",
        "input": {"goal": goal},
        "output": None,
        "error": None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_ms": 0,
    }

    try:
        hit = check_memory(goal)
        duration_ms = (time.time() - start_time) * 1000
        tool_log_entry["duration_ms"] = duration_ms

        if hit:
            print(f"[MEMORY] Hit! Similar past query: '{hit['query']}' (similarity: {hit['similarity']:.2f})")
            tool_log_entry["output"] = {"hit": True, "past_query": hit["query"], "similarity": hit["similarity"]}
            return {
                "memory_hit": hit,
                "memory_action": "use_cache",
                "tool_call_log": state.get("tool_call_log", []) + [tool_log_entry],
            }
        else:
            print("[MEMORY] No hit -- proceeding with fresh search")
            tool_log_entry["output"] = {"hit": False}
            return {
                "memory_hit": None,
                "memory_action": "new_search",
                "tool_call_log": state.get("tool_call_log", []) + [tool_log_entry],
            }

    except Exception as e:
        logger.error(f"Memory check failed: {e}")
        duration_ms = (time.time() - start_time) * 1000
        tool_log_entry["error"] = str(e)
        tool_log_entry["duration_ms"] = duration_ms
        return {
            "memory_hit": None,
            "memory_action": "new_search",
            "tool_call_log": state.get("tool_call_log", []) + [tool_log_entry],
            "errors": state.get("errors", []) + [f"Memory check failed: {e}"],
        }


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def should_use_cache(state: dict) -> str:
    """Decide whether to use cached result or do a fresh search."""
    if state.get("memory_action") == "use_cache" and state.get("memory_hit"):
        hit = state["memory_hit"]
        if hit.get("summary"):
            return "use_cache"
    return "fresh_search"


def should_replan(state: dict) -> str:
    """Check if coverage is too thin and we should replan."""
    summary = state.get("structured_summary") or {}
    coverage = summary.get("coverage_assessment") or {}
    coverage_quality = coverage.get("coverage_quality", "unknown")

    dedup_results = state.get("dedup_results", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    # Replan conditions:
    # 1. Coverage is "thin"
    # 2. Very few usable sources (< 2)
    # 3. Haven't exceeded retry limit
    if retry_count < max_retries:
        if coverage_quality == "thin" or len(dedup_results) < 2:
            print(f"[REPLAN] Coverage too thin (quality={coverage_quality}, sources={len(dedup_results)}) -- replanning (attempt {retry_count + 1}/{max_retries})")
            return "replan"

    return "export"


# ---------------------------------------------------------------------------
# Cache export node (for memory hits)
# ---------------------------------------------------------------------------

def cache_export_node(state: dict) -> dict:
    """Use cached summary and export it."""
    import time

    hit = state.get("memory_hit", {})
    cached_summary = hit.get("summary", {})

    print(f"[CACHE] Using cached result from {hit.get('timestamp', 'unknown')}")

    # Still export a fresh copy
    from tools.export import export_markdown

    export_path = None
    try:
        export_path = export_markdown(
            structured_summary=cached_summary,
            goal=state["goal"],
            plan=[f"Used cached result from previous search: '{hit.get('query', '')}'"],
            tool_call_log=state.get("tool_call_log", []),
            errors=state.get("errors", []),
        )
        print(f"[EXPORT] Cached report re-exported to: {export_path}")
    except Exception as e:
        logger.error(f"Cache export failed: {e}")

    return {
        "structured_summary": cached_summary,
        "export_path": export_path,
        "plan": [f"Used cached result from previous search: '{hit.get('query', '')}'"],
    }


# ---------------------------------------------------------------------------
# Replan increment node
# ---------------------------------------------------------------------------

def replan_node(state: dict) -> dict:
    """Increment retry counter and clear stale data for replanning."""
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "search_results": [],
        "fetched_pages": {},
        "dedup_results": [],
        "structured_summary": None,
        "errors": [],  # Clear stale errors from previous loop
    }


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and compile the LangGraph research agent."""

    graph = StateGraph(ResearchAgentState)

    # --- Add nodes ---
    graph.add_node("query_understanding", query_understanding_node)
    graph.add_node("memory_check", memory_check_node)
    graph.add_node("cache_export", cache_export_node)
    graph.add_node("planning", planning_node)
    graph.add_node("source_selection", source_selection_node)
    graph.add_node("search_execution", search_execution_node)
    graph.add_node("fetch_extract", fetch_extract_node)
    graph.add_node("dedup_filter", dedup_filter_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("export", export_node)
    graph.add_node("replan", replan_node)

    # --- Set entry point ---
    graph.set_entry_point("query_understanding")

    # --- Add edges ---
    # Linear flow: query_understanding -> memory_check
    graph.add_edge("query_understanding", "memory_check")

    # Conditional: memory_check -> cache_export OR planning
    graph.add_conditional_edges(
        "memory_check",
        should_use_cache,
        {
            "use_cache": "cache_export",
            "fresh_search": "planning",
        },
    )

    # cache_export -> END
    graph.add_edge("cache_export", END)

    # Linear flow: planning -> source_selection -> search_execution -> fetch_extract -> dedup_filter -> synthesis
    graph.add_edge("planning", "source_selection")
    graph.add_edge("source_selection", "search_execution")
    graph.add_edge("search_execution", "fetch_extract")
    graph.add_edge("fetch_extract", "dedup_filter")
    graph.add_edge("dedup_filter", "synthesis")

    # Conditional: synthesis -> export OR replan
    graph.add_conditional_edges(
        "synthesis",
        should_replan,
        {
            "export": "export",
            "replan": "replan",
        },
    )

    # Replan loops back to planning
    graph.add_edge("replan", "planning")

    # Export -> END
    graph.add_edge("export", END)

    return graph.compile()


def run_research_agent(goal: str, skip_cache: bool = False) -> dict:
    """Run the full research agent pipeline.

    Args:
        goal: The user's research query.
        skip_cache: If True, bypass memory cache and force fresh search.

    Returns:
        Final state dict with all results.
    """
    graph = build_graph()

    initial_state = {
        "goal": goal,
        "parsed_query": {},
        "plan": [],
        "sources_strategy": "general_web",
        "search_queries": [],
        "search_results": [],
        "fetched_pages": {},
        "dedup_results": [],
        "structured_summary": None,
        "retry_count": 0,
        "max_retries": 2,
        "tool_call_log": [],
        "errors": [],
        "memory_hit": None,
        "memory_action": "skip_cache" if skip_cache else "new_search",
        "export_path": None,
    }

    print("\n" + "=" * 60, flush=True)
    print("  AUTONOMOUS RESEARCH AGENT", flush=True)
    print(f"  Goal: {goal}", flush=True)
    if skip_cache:
        print("  (Cache bypassed -- forcing fresh search)", flush=True)
    print("=" * 60 + "\n", flush=True)

    final_state = graph.invoke(initial_state)

    # Print summary
    print("\n" + "=" * 60)
    print("RESEARCH COMPLETE")
    print("=" * 60)

    if final_state.get("export_path"):
        print(f"[REPORT] {final_state['export_path']}")

    errors = final_state.get("errors", [])
    if errors:
        print(f"[WARN] {len(errors)} issue(s) encountered:")
        for err in errors[:5]:
            print(f"   • {err[:100]}")

    tool_calls = final_state.get("tool_call_log", [])
    print(f"[INFO] Total tool calls: {len(tool_calls)}")
    print("=" * 60)

    return final_state
