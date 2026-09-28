"""
nodes/planning.py — Planning node.

Produces a visible planning trace: the steps the agent will take
and which tools it will use.
"""

import json
import time
import logging
from tools.llm import groq_chat

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research planning agent. Given a parsed research query, produce a step-by-step plan for how to research this topic.

Your plan should include:
1. What search queries to run (be specific)
2. What types of sources to prioritize
3. How many sources to aim for
4. What aspects to focus on during synthesis
5. Any special considerations

IMPORTANT rules for search_queries:
- Use simple, natural-language phrases (e.g. "KV cache compression techniques for LLMs")
- Do NOT use quoted exact-match syntax (no wrapping phrases in double quotes)
- Do NOT use special characters like smart quotes or non-breaking hyphens
- Keep queries concise (5-10 words each)
- Generate 2-4 diverse queries that approach the topic from different angles

Respond ONLY with valid JSON in this format:
{
    "steps": [
        "Step 1: Search for ...",
        "Step 2: Fetch and extract content from top results",
        "Step 3: Filter for relevance and remove duplicates",
        "Step 4: Synthesize findings into structured report",
        "Step 5: Export report as Markdown"
    ],
    "search_queries": [
        "primary search query in plain language",
        "alternative complementary query"
    ],
    "target_source_count": 5,
    "focus_areas": ["area1", "area2"]
}"""


def planning_node(state: dict) -> dict:
    """Generate a visible planning trace before execution begins."""
    parsed_query = state.get("parsed_query", {})
    goal = state.get("goal", "")
    retry_count = state.get("retry_count", 0)
    start_time = time.time()

    tool_log_entry = {
        "tool": "planning",
        "input": {"parsed_query": parsed_query, "retry_count": retry_count},
        "output": None,
        "error": None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_ms": 0,
    }

    # If this is a replan (retry), adjust the prompt
    replan_context = ""
    if retry_count > 0:
        prev_errors = state.get("errors", [])
        replan_context = (
            f"\n\nIMPORTANT: This is replan attempt #{retry_count}. "
            f"Previous attempt had issues: {'; '.join(prev_errors[-3:])}. "
            f"Please reformulate the search queries to be broader or use different terms."
        )

    # Attempt planning with retry for malformed JSON
    max_attempts = 2
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Research goal: {goal}\n"
            f"Parsed query: {json.dumps(parsed_query)}"
            f"{replan_context}"
        )},
    ]

    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON. "
                        "Please respond with ONLY valid JSON. No markdown, no explanation. "
                        "Use only plain ASCII quotes, no smart quotes."
                    ),
                })

            raw = groq_chat(
                messages=messages,
                temperature=0.3,
                max_tokens=800,
            )

            # Extract JSON from markdown code blocks if present
            json_str = raw
            if "```" in json_str:
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                json_str = json_str.strip()

            # Repair common JSON issues from LLMs
            json_str = json_str.replace("\u2018", "'").replace("\u2019", "'")  # smart single quotes
            json_str = json_str.replace("\u201c", '"').replace("\u201d", '"')  # smart double quotes
            json_str = json_str.replace("\u2013", "-").replace("\u2014", "-")  # en/em dashes
            import re
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # trailing commas

            plan_data = json.loads(json_str)

            steps = plan_data.get("steps", [f"Research '{goal}'"])
            search_queries = plan_data.get("search_queries", [parsed_query.get("topic", goal)])

            duration_ms = (time.time() - start_time) * 1000
            tool_log_entry["output"] = plan_data
            tool_log_entry["duration_ms"] = duration_ms

            # Print plan for visibility
            print("\n" + "=" * 60, flush=True)
            print("RESEARCH PLAN", flush=True)
            print("=" * 60, flush=True)
            for i, step in enumerate(steps, 1):
                print(f"  {i}. {step}", flush=True)
            print(f"  Search queries: {search_queries}", flush=True)
            print("=" * 60 + "\n", flush=True)

            return {
                "plan": steps,
                "search_queries": search_queries,
                "tool_call_log": state.get("tool_call_log", []) + [tool_log_entry],
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Planning attempt {attempt}: JSON parse failed: {e}")
            if attempt < max_attempts:
                continue  # Retry with corrective prompt

        except Exception as e:
            logger.error(f"Planning failed on attempt {attempt}: {e}")
            break # Break out of loop to trigger fallback

    # If we get here, it means all attempts failed or we broke out of the loop
    logger.error("Planning failed after all attempts. Using fallback.")

    # Fallback plan
    topic = parsed_query.get("topic", goal)
    fallback_steps = [
        f"Search for '{topic}' using web search",
        "Fetch and extract content from top results",
        "Filter for relevance and remove duplicates",
        "Synthesize findings into structured report",
        "Export report as Markdown",
    ]
    fallback_queries = [topic]

    duration_ms = (time.time() - start_time) * 1000
    tool_log_entry["output"] = {"steps": fallback_steps, "search_queries": fallback_queries}
    tool_log_entry["error"] = "Planning failed after all attempts"
    tool_log_entry["duration_ms"] = duration_ms

    print("\n" + "=" * 60, flush=True)
    print("RESEARCH PLAN (fallback)", flush=True)
    print("=" * 60, flush=True)
    for i, step in enumerate(fallback_steps, 1):
        print(f"  {i}. {step}", flush=True)
    print("=" * 60 + "\n", flush=True)

    return {
        "plan": fallback_steps,
        "search_queries": fallback_queries,
        "tool_call_log": state.get("tool_call_log", []) + [tool_log_entry],
        "errors": state.get("errors", []) + ["Planning fallback used due to parse failure"],
    }
