"""
tools/page_fetch.py — Page fetch and content extraction tool.

Uses Jina Reader API (free) as primary extractor — handles PDFs, JS pages,
and complex layouts. Falls back to trafilatura for offline/simple pages.
"""

import logging
import requests
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Status codes for fetch results
STATUS_OK = "ok"
STATUS_TIMEOUT = "timeout"
STATUS_HTTP_ERROR = "http_error"
STATUS_EMPTY_CONTENT = "empty_content"
STATUS_FETCH_ERROR = "fetch_error"

# Minimum useful content length (chars)
MIN_CONTENT_LENGTH = 100

# Jina Reader API (free, no key required for basic use)
JINA_READER_URL = "https://r.jina.ai/"


def fetch_page(url: str, timeout: int = 15) -> dict:
    """Fetch a single URL and extract its main content.

    Strategy:
    1. If raw_content was already provided by Tavily, use it directly.
    2. Otherwise, try Jina Reader API (handles PDFs, JS, complex pages).
    3. Fall back to trafilatura if Jina fails.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        dict with keys: url, status, content, error, char_count, title
    """
    result = {
        "url": url,
        "status": STATUS_FETCH_ERROR,
        "content": "",
        "error": None,
        "char_count": 0,
        "title": "",
    }

    # Try Jina Reader API first
    try:
        jina_url = f"{JINA_READER_URL}{url}"
        headers = {
            "Accept": "text/plain",
            "X-Return-Format": "text",
        }
        resp = requests.get(jina_url, headers=headers, timeout=timeout)

        if resp.status_code == 200 and len(resp.text.strip()) >= MIN_CONTENT_LENGTH:
            result["status"] = STATUS_OK
            result["content"] = resp.text.strip()
            result["char_count"] = len(result["content"])
            # Try to extract title from first line
            lines = result["content"].split("\n")
            if lines:
                result["title"] = lines[0].strip("# ").strip()[:200]
            logger.info(f"Jina Reader fetched {url}: {result['char_count']} chars")
            return result
        elif resp.status_code == 429:
            logger.warning(f"Jina Reader rate limited for {url}, falling back to trafilatura")
        else:
            logger.warning(f"Jina Reader returned insufficient content for {url} ({len(resp.text)} chars, status {resp.status_code})")

    except requests.Timeout:
        logger.warning(f"Jina Reader timed out for {url}")
    except Exception as e:
        logger.warning(f"Jina Reader failed for {url}: {e}")

    # Fallback: trafilatura
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            result["status"] = STATUS_HTTP_ERROR
            result["error"] = "Failed to download page (timeout, 404, or SSL error)"
            return result

        content = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )

        if content is None or len(content.strip()) < MIN_CONTENT_LENGTH:
            result["status"] = STATUS_EMPTY_CONTENT
            result["content"] = content or ""
            result["char_count"] = len(content) if content else 0
            result["error"] = f"Extracted content too short ({result['char_count']} chars, minimum {MIN_CONTENT_LENGTH})"
            return result

        result["status"] = STATUS_OK
        result["content"] = content.strip()
        result["char_count"] = len(result["content"])
        logger.info(f"Trafilatura fetched {url}: {result['char_count']} chars")

    except Exception as e:
        result["status"] = STATUS_FETCH_ERROR
        result["error"] = str(e)
        logger.error(f"All fetch methods failed for {url}: {e}")

    return result


def fetch_page_with_raw(url: str, raw_content: str = "", timeout: int = 15) -> dict:
    """Fetch a page, using pre-extracted raw_content from Tavily if available.

    Args:
        url: The URL to fetch.
        raw_content: Pre-extracted content from Tavily's search results.
        timeout: Request timeout in seconds.

    Returns:
        dict with keys: url, status, content, error, char_count, title
    """
    # If Tavily already gave us good content, use it directly
    if raw_content and len(raw_content.strip()) >= MIN_CONTENT_LENGTH:
        return {
            "url": url,
            "status": STATUS_OK,
            "content": raw_content.strip(),
            "error": None,
            "char_count": len(raw_content.strip()),
            "title": "",
        }

    # Otherwise fetch fresh
    return fetch_page(url, timeout)


def fetch_pages_parallel(
    urls: list[str],
    raw_contents: Optional[dict] = None,
    max_workers: int = 5,
    timeout: int = 15,
) -> dict:
    """Fetch multiple pages concurrently.

    Args:
        urls: List of URLs to fetch.
        raw_contents: Optional dict of url -> raw_content from Tavily.
        max_workers: Maximum concurrent fetch threads.
        timeout: Per-request timeout in seconds.

    Returns:
        dict mapping url -> fetch result dict.
    """
    if raw_contents is None:
        raw_contents = {}

    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(
                fetch_page_with_raw,
                url,
                raw_contents.get(url, ""),
                timeout,
            ): url
            for url in urls
        }

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result(timeout=timeout + 5)
                results[url] = result
            except Exception as e:
                results[url] = {
                    "url": url,
                    "status": STATUS_FETCH_ERROR,
                    "content": "",
                    "error": f"Thread error: {e}",
                    "char_count": 0,
                    "title": "",
                }

    return results
