"""
nodes/synthesis.py — Synthesis node.

Uses the LLM to produce a structured research report from
the deduplicated, relevant content chunks. Includes grounding/coverage
check and self-correction for malformed output.
"""

import json
import time
import logging
from tools.llm import groq_chat

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research synthesis agent. Given a collection of source materials about a topic, produce a comprehensive, well-structured research report.

Your output MUST be valid JSON with this exact structure:
{
    "executive_summary": "A 2-3 paragraph summary of the key findings",
    "key_points": [
        "Key point 1",
        "Key point 2",
        ...
    ],
    "important_findings": [
        "Finding 1 with specific details and data",
        "Finding 2 with specific details and data",
        ...
    ],
    "actionable_insights": [
        "Actionable insight 1",
        "Actionable insight 2",
        ...
    ],
    "detailed_analysis": "A longer multi-paragraph analysis section",
    "references": [
        {"title": "Source Title", "url": "https://..."},
        ...
    ],
    "coverage_assessment": {
        "sources_used": 5,
        "coverage_quality": "good",
        "gaps": ["Any identified gaps in coverage"]
    }
}

Requirements:
- Be factual and cite specific information from the sources
- Include at least 3 key points and 3 findings
- Include specific data, numbers, or quotes where available
- Identify any gaps or limitations in the available information
- Rate coverage_quality as "good", "moderate", or "thin"

IMPORTANT: Respond ONLY with valid JSON. No other text before or after."""


def synthesis_node(state: dict) -> dict:
    """Synthesize deduplicated content into a structured report."""
    dedup_results = state.get("dedup_results", [])
    parsed_query = state.get("parsed_query", {})
    goal = state.get("goal", "")
    errors = list(state.get("errors", []))
    start_time = time.time()

    tool_log_entry = {
        "tool": "synthesis",
        "input": {"source_count": len(dedup_results), "goal": goal},
        "output": None,
        "error": None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_ms": 0,
    }

    # Build source material for the LLM
    source_text_parts = []
    references = []
    for i, item in enumerate(dedup_results, 1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        content = item.get("content", "")
        # Truncate very long content but keep intro + conclusion
        if len(content) > 8000:
            content = content[:5000] + "\n\n... [middle truncated] ...\n\n" + content[-3000:]

        source_text_parts.append(
            f"--- SOURCE {i}: {title} ---\n"
            f"URL: {url}\n"
            f"Relevance Score: {item.get('relevance_score', 'N/A')}\n\n"
            f"{content}\n"
        )
        references.append({"title": title, "url": url})

    source_text = "\n".join(source_text_parts)

    if not source_text.strip():
        # No content to synthesize
        empty_summary = {
            "executive_summary": f"Unable to find sufficient information about '{goal}'. The search returned results but content extraction failed for most sources.",
            "key_points": ["Insufficient data available for comprehensive analysis"],
            "important_findings": ["No significant findings -- sources were unavailable or contained insufficient content"],
            "actionable_insights": ["Consider refining the search query or trying alternative sources"],
            "detailed_analysis": "The research agent was unable to gather enough source material to produce a detailed analysis.",
            "references": [],
            "coverage_assessment": {"sources_used": 0, "coverage_quality": "thin", "gaps": ["All sources"]},
        }

        duration_ms = (time.time() - start_time) * 1000
        tool_log_entry["output"] = {"coverage": "thin", "sources": 0}
        tool_log_entry["error"] = "No source content available"
        tool_log_entry["duration_ms"] = duration_ms

        return {
            "structured_summary": empty_summary,
            "tool_call_log": state.get("tool_call_log", []) + [tool_log_entry],
            "errors": errors + ["Synthesis produced thin report -- no source content available"],
        }

    print(f"[SYNTHESIS] Synthesizing report from {len(dedup_results)} sources...", flush=True)

    # Attempt synthesis with retry for malformed JSON
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Research goal: {goal}\n"
                    f"Topic: {parsed_query.get('topic', goal)}\n"
                    f"Constraints: {parsed_query.get('constraints', [])}\n\n"
                    f"SOURCE MATERIALS ({len(dedup_results)} sources):\n\n"
                    f"{source_text}"
                )},
            ]

            # On retry, add a correction message
            if attempt > 1:
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON. "
                        "Please respond with ONLY valid JSON matching the required schema. "
                        "No markdown code blocks, no explanation text -- just the JSON object."
                    ),
                })

            raw = groq_chat(
                messages=messages,
                temperature=0.2,
                max_tokens=4000,
            )

            # Handle markdown code blocks
            json_str = raw
            if "```" in json_str:
                parts = json_str.split("```")
                for part in parts:
                    stripped = part.strip()
                    if stripped.startswith("json"):
                        stripped = stripped[4:].strip()
                    if stripped.startswith("{"):
                        json_str = stripped
                        break

            # Repair common JSON issues from LLMs
            import re
            json_str = json_str.replace("\u2018", "'").replace("\u2019", "'")  # smart single quotes
            json_str = json_str.replace("\u201c", '"').replace("\u201d", '"')  # smart double quotes
            json_str = json_str.replace("\u2013", "-").replace("\u2014", "-")  # en/em dashes
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # trailing commas

            structured_summary = json.loads(json_str)

            # Ensure references from our sources are included
            if not structured_summary.get("references"):
                structured_summary["references"] = references

            # Coverage check
            coverage = structured_summary.get("coverage_assessment", {})
            coverage_quality = coverage.get("coverage_quality", "unknown")

            duration_ms = (time.time() - start_time) * 1000
            tool_log_entry["output"] = {
                "coverage": coverage_quality,
                "sources": len(dedup_results),
                "key_points": len(structured_summary.get("key_points", [])),
            }
            tool_log_entry["duration_ms"] = duration_ms

            print(f"   [OK] Report generated (coverage: {coverage_quality})", flush=True)

            return {
                "structured_summary": structured_summary,
                "tool_call_log": state.get("tool_call_log", []) + [tool_log_entry],
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Synthesis attempt {attempt}: JSON parse failed: {e}")
            if attempt == max_attempts:
                # Final fallback: plain text report
                errors.append(f"LLM returned malformed JSON after {max_attempts} attempts, using fallback")
                fallback = {
                    "executive_summary": raw[:2000] if raw else "Synthesis failed",
                    "key_points": ["See executive summary for details"],
                    "important_findings": ["Report generated from raw LLM output due to JSON formatting issues"],
                    "actionable_insights": [],
                    "detailed_analysis": raw if raw else "",
                    "references": references,
                    "coverage_assessment": {
                        "sources_used": len(dedup_results),
                        "coverage_quality": "moderate",
                        "gaps": ["Report formatting was degraded due to LLM output issues"],
                    },
                }

                duration_ms = (time.time() - start_time) * 1000
                tool_log_entry["output"] = {"coverage": "moderate", "sources": len(dedup_results)}
                tool_log_entry["error"] = f"JSON parse failed after {max_attempts} attempts, used fallback"
                tool_log_entry["duration_ms"] = duration_ms

                return {
                    "structured_summary": fallback,
                    "tool_call_log": state.get("tool_call_log", []) + [tool_log_entry],
                    "errors": errors,
                }

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            errors.append(f"Synthesis error: {e}")

            duration_ms = (time.time() - start_time) * 1000
            tool_log_entry["error"] = str(e)
            tool_log_entry["duration_ms"] = duration_ms

            # Minimal fallback
            fallback = {
                "executive_summary": f"Synthesis failed: {e}",
                "key_points": [],
                "important_findings": [],
                "actionable_insights": [],
                "detailed_analysis": "",
                "references": references,
                "coverage_assessment": {
                    "sources_used": len(dedup_results),
                    "coverage_quality": "thin",
                    "gaps": [f"Synthesis error: {e}"],
                },
            }

            return {
                "structured_summary": fallback,
                "tool_call_log": state.get("tool_call_log", []) + [tool_log_entry],
                "errors": errors,
            }

    # Should not reach here
    return state
