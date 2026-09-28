"""
tools/export.py — Export tool for generating Markdown reports.

Converts the structured_summary dict into a well-formatted Markdown file.
"""

import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def export_markdown(
    structured_summary: dict,
    goal: str,
    plan: list[str],
    tool_call_log: list[dict],
    errors: list[str],
    output_dir: str = "output",
) -> str:
    """Export the research report as a Markdown file.

    Args:
        structured_summary: The synthesis output dict.
        goal: Original user query.
        plan: Planning trace.
        tool_call_log: Full tool call log.
        errors: Accumulated errors.
        output_dir: Directory to write the report to.

    Returns:
        Path to the exported Markdown file.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a safe filename from the goal
    safe_goal = "".join(c if c.isalnum() or c in " -_" else "_" for c in goal[:50]).strip()
    safe_goal = safe_goal.replace(" ", "_")
    filename = f"report_{safe_goal}_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)

    lines = []

    # --- Header ---
    lines.append(f"# Research Report: {goal}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # --- Executive Summary ---
    summary = structured_summary.get("executive_summary", "")
    if summary:
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(summary)
        lines.append("")
        lines.append("---")
        lines.append("")

    # --- Key Points ---
    key_points = structured_summary.get("key_points", [])
    if key_points:
        lines.append("## Key Points")
        lines.append("")
        for i, point in enumerate(key_points, 1):
            lines.append(f"{i}. {point}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # --- Important Findings ---
    findings = structured_summary.get("important_findings", [])
    if findings:
        lines.append("## Important Findings")
        lines.append("")
        for finding in findings:
            lines.append(f"- {finding}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # --- Actionable Insights ---
    insights = structured_summary.get("actionable_insights", [])
    if insights:
        lines.append("## Actionable Insights")
        lines.append("")
        for insight in insights:
            lines.append(f"- {insight}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # --- Detailed Analysis ---
    analysis = structured_summary.get("detailed_analysis", "")
    if analysis:
        lines.append("## Detailed Analysis")
        lines.append("")
        lines.append(analysis)
        lines.append("")
        lines.append("---")
        lines.append("")

    # --- References / Sources ---
    references = structured_summary.get("references", [])
    if references:
        lines.append("## References & Sources")
        lines.append("")
        for i, ref in enumerate(references, 1):
            if isinstance(ref, dict):
                title = ref.get("title", "Untitled")
                url = ref.get("url", "")
                lines.append(f"{i}. [{title}]({url})")
            else:
                lines.append(f"{i}. {ref}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # --- Limitations / Failed Sources ---
    if errors:
        lines.append("## Limitations & Failed Sources")
        lines.append("")
        for error in errors:
            lines.append(f"- [!] {error}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # --- Planning Trace ---
    lines.append("## Agent Planning Trace")
    lines.append("")
    for i, step in enumerate(plan, 1):
        lines.append(f"{i}. {step}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # --- Tool Call Log ---
    lines.append("## Tool Call Log")
    lines.append("")
    lines.append("| # | Tool | Status | Duration |")
    lines.append("|---|------|--------|----------|")
    for i, call in enumerate(tool_call_log, 1):
        tool = call.get("tool", "unknown")
        status = "OK" if not call.get("error") else "FAIL"
        duration = call.get("duration_ms", "—")
        if isinstance(duration, (int, float)):
            duration = f"{duration:.0f}ms"
        lines.append(f"| {i} | {tool} | {status} | {duration} |")
    lines.append("")

    # Write file
    content = "\n".join(lines)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Report exported to {filepath}")
    return filepath
