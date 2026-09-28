"""
nodes/export_node.py — Export node + Memory Write.

Exports the structured summary as Markdown and stores the search in memory.
"""

import time
import logging
from tools.export import export_markdown
from tools.memory import store_search

logger = logging.getLogger(__name__)


def export_node(state: dict) -> dict:
    """Export the report and store in memory."""
    structured_summary = state.get("structured_summary")
    goal = state.get("goal", "")
    plan = state.get("plan", [])
    tool_call_log = state.get("tool_call_log", [])
    errors = state.get("errors", [])
    strategy = state.get("sources_strategy", "general_web")
    dedup_results = state.get("dedup_results", [])
    start_time = time.time()

    export_log_entry = {
        "tool": "export_markdown",
        "input": {"goal": goal},
        "output": None,
        "error": None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_ms": 0,
    }

    memory_log_entry = {
        "tool": "memory_write",
        "input": {"goal": goal},
        "output": None,
        "error": None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_ms": 0,
    }

    export_path = None

    # --- Export ---
    if structured_summary:
        try:
            export_path = export_markdown(
                structured_summary=structured_summary,
                goal=goal,
                plan=plan,
                tool_call_log=tool_call_log,
                errors=errors,
            )
            export_log_entry["output"] = {"path": export_path}
            print(f"[EXPORT] Report exported to: {export_path}", flush=True)

        except Exception as e:
            export_log_entry["error"] = str(e)
            errors.append(f"Export failed: {e}")
            logger.error(f"Export failed: {e}")
    else:
        export_log_entry["error"] = "No structured summary to export"

    export_duration = (time.time() - start_time) * 1000
    export_log_entry["duration_ms"] = export_duration

    # --- Memory Write ---
    memory_start = time.time()
    try:
        store_search(
            query=goal,
            sources_count=len(dedup_results),
            summary_path=export_path,
            summary_dict=structured_summary,
            strategy=strategy,
        )
        memory_log_entry["output"] = {"stored": True}
        print("[MEMORY] Search stored in memory", flush=True)

    except Exception as e:
        memory_log_entry["error"] = str(e)
        errors.append(f"Memory write failed: {e}")
        logger.error(f"Memory write failed: {e}")

    memory_log_entry["duration_ms"] = (time.time() - memory_start) * 1000

    return {
        "export_path": export_path,
        "tool_call_log": tool_call_log + [export_log_entry, memory_log_entry],
        "errors": errors,
    }
