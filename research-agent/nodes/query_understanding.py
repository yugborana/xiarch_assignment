"""
nodes/query_understanding.py — Goal Intake / Query Understanding node.

Uses the LLM to parse a raw user query into structured components:
topic, constraints, output format, and query type.
"""

import json
import time
import logging
from tools.llm import groq_chat

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research query analyzer. Given a user's research goal, extract the following structured information:

1. topic: The main subject to research (string)
2. constraints: Any constraints like time period, geography, specific aspects (list of strings, can be empty)
3. output_format: What kind of output the user expects - "summary", "comparison", "brief", "analysis" (string)
4. query_type: Classify the query as one of:
   - "news" — recent events, developments, trends
   - "tech_docs" — technical/scientific topics, how things work
   - "company_research" — company analysis, competitive landscape
   - "general_web" — general knowledge, broad topics

Respond ONLY with valid JSON, no other text. Example:
{
    "topic": "KV-cache compression in LLMs",
    "constraints": ["recent approaches", "2024"],
    "output_format": "summary",
    "query_type": "tech_docs"
}"""


def query_understanding_node(state: dict) -> dict:
    """Parse the user's goal into structured query components."""
    goal = state["goal"]
    start_time = time.time()

    tool_log_entry = {
        "tool": "query_understanding",
        "input": {"goal": goal},
        "output": None,
        "error": None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_ms": 0,
    }

    try:
        raw_response = groq_chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Research goal: {goal}"},
            ],
            temperature=0.1,
            max_tokens=500,
        )

        # Parse JSON from response (handle potential markdown code blocks)
        json_str = raw_response
        if "```" in json_str:
            # Extract JSON from code block
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            json_str = json_str.strip()

        # Repair common JSON issues from LLMs
        import re
        json_str = json_str.replace("\u2018", "'").replace("\u2019", "'")  # smart single quotes
        json_str = json_str.replace("\u201c", '"').replace("\u201d", '"')  # smart double quotes
        json_str = json_str.replace("\u2013", "-").replace("\u2014", "-")  # en/em dashes
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # trailing commas

        parsed_query = json.loads(json_str)

        # Validate required fields
        required = ["topic", "constraints", "output_format", "query_type"]
        for field in required:
            if field not in parsed_query:
                parsed_query[field] = "" if field != "constraints" else []

        # Ensure query_type is valid
        valid_types = {"news", "tech_docs", "company_research", "general_web"}
        if parsed_query["query_type"] not in valid_types:
            parsed_query["query_type"] = "general_web"

        duration_ms = (time.time() - start_time) * 1000
        tool_log_entry["output"] = parsed_query
        tool_log_entry["duration_ms"] = duration_ms

        logger.info(f"Query understood: topic='{parsed_query['topic']}', type='{parsed_query['query_type']}'")

        return {
            "parsed_query": parsed_query,
            "tool_call_log": state.get("tool_call_log", []) + [tool_log_entry],
        }

    except json.JSONDecodeError as e:
        # Self-correction: if JSON parsing fails, use a simple fallback
        logger.warning(f"JSON parse failed, using fallback: {e}")

        fallback = {
            "topic": goal,
            "constraints": [],
            "output_format": "summary",
            "query_type": "general_web",
        }

        duration_ms = (time.time() - start_time) * 1000
        tool_log_entry["output"] = fallback
        tool_log_entry["error"] = f"JSON parse failed, used fallback: {e}"
        tool_log_entry["duration_ms"] = duration_ms

        return {
            "parsed_query": fallback,
            "tool_call_log": state.get("tool_call_log", []) + [tool_log_entry],
            "errors": state.get("errors", []) + [f"Query parsing fallback used: {e}"],
        }

    except Exception as e:
        logger.error(f"Query understanding failed: {e}")

        fallback = {
            "topic": goal,
            "constraints": [],
            "output_format": "summary",
            "query_type": "general_web",
        }

        duration_ms = (time.time() - start_time) * 1000
        tool_log_entry["output"] = fallback
        tool_log_entry["error"] = str(e)
        tool_log_entry["duration_ms"] = duration_ms

        return {
            "parsed_query": fallback,
            "tool_call_log": state.get("tool_call_log", []) + [tool_log_entry],
            "errors": state.get("errors", []) + [f"Query understanding error: {e}"],
        }
