"""
tools/llm.py — Shared LLM call helper with retry logic.

Wraps Groq API calls with exponential backoff for rate limits (429).
Used by query_understanding, planning, and synthesis nodes.
"""

import time
import logging
from groq import Groq

logger = logging.getLogger(__name__)

# Default model for all LLM calls
DEFAULT_MODEL = "openai/gpt-oss-20b"


def groq_chat(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1000,
    max_retries: int = 3,
) -> str:
    """Make a Groq chat completion call with retry logic.

    Args:
        messages: Chat messages list.
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.
        max_retries: Number of retry attempts for rate limits.

    Returns:
        The raw response content string.

    Raises:
        Exception: If all retries are exhausted or a non-retryable error occurs.
    """
    client = Groq()

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            error_str = str(e)
            is_rate_limit = any(kw in error_str for kw in ("429", "rate_limit", "rate limit", "too many"))

            if is_rate_limit and attempt < max_retries:
                wait_time = 2 ** attempt  # 2, 4, 8 seconds
                logger.warning(
                    f"Groq rate limited (attempt {attempt}/{max_retries}), "
                    f"retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
                continue
            else:
                raise
