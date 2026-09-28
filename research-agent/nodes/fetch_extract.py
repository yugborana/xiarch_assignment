"""
nodes/fetch_extract.py — Fetch & Extract node.

Fetches pages concurrently and extracts main content.
Handles timeouts, 404s, empty content, and logs all failures.
"""

import time
import logging
from tools.page_fetch import fetch_pages_parallel, STATUS_OK, STATUS_EMPTY_CONTENT

logger = logging.getLogger(__name__)


def fetch_extract_node(state: dict) -> dict:
    """Fetch and extract content from all search result URLs."""
    search_results = state.get("search_results", [])
    start_time = time.time()

    urls = [r["url"] for r in search_results if r.get("url")]

    # Collect any pre-extracted content from Tavily search results
    raw_contents = {}
    for r in search_results:
        if r.get("url") and r.get("raw_content"):
            raw_contents[r["url"]] = r["raw_content"]

    tool_log_entry = {
        "tool": "page_fetch",
        "input": {"url_count": len(urls), "urls": urls[:5]},
        "output": None,
        "error": None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_ms": 0,
    }

    pre_fetched = len(raw_contents)
    print(f"[FETCH] Processing {len(urls)} pages ({pre_fetched} pre-extracted by Tavily)...", flush=True)

    # Fetch all pages in parallel (pre-extracted ones skip network calls)
    fetched_pages = fetch_pages_parallel(urls, raw_contents=raw_contents, max_workers=5, timeout=15)

    # Classify results
    ok_count = sum(1 for v in fetched_pages.values() if v["status"] == STATUS_OK)
    failed_count = len(fetched_pages) - ok_count
    errors = list(state.get("errors", []))

    # Log individual failures
    for url, result in fetched_pages.items():
        if result["status"] != STATUS_OK:
            error_msg = f"Fetch failed for {url}: {result['status']} -- {result.get('error', 'unknown')}"
            errors.append(error_msg)
            logger.warning(error_msg)
            print(f"   [FAIL] {result['status']}: {url[:60]}...", flush=True)

    # Log successes
    for url, result in fetched_pages.items():
        if result["status"] == STATUS_OK:
            print(f"   [OK] {result['char_count']:,} chars: {url[:60]}...", flush=True)

    duration_ms = (time.time() - start_time) * 1000
    tool_log_entry["output"] = {
        "ok_count": ok_count,
        "failed_count": failed_count,
        "total": len(fetched_pages),
    }
    tool_log_entry["duration_ms"] = duration_ms

    if failed_count > 0:
        tool_log_entry["error"] = f"{failed_count}/{len(fetched_pages)} fetches failed"

    print(f"   Summary: {ok_count} OK, {failed_count} failed out of {len(fetched_pages)}", flush=True)

    return {
        "fetched_pages": fetched_pages,
        "tool_call_log": state.get("tool_call_log", []) + [tool_log_entry],
        "errors": errors,
    }
