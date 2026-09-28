"""
cli.py — Command-line interface for the Autonomous Research Agent.

Usage:
    python cli.py "your research query here"
    python cli.py --interactive
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# Windows console uses cp1252 by default, which can't handle some unicode
# chars that the LLM may return (e.g. non-breaking hyphens). Force UTF-8.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load .env for API keys
load_dotenv()


def setup_logging(verbose: bool = False):
    """Configure logging to stdout (avoids red stderr lines in PowerShell)."""
    level = logging.DEBUG if verbose else logging.WARNING
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.root.handlers = [handler]
    logging.root.setLevel(level)
    # Quiet down noisy libraries
    for lib in ("httpx", "httpcore", "urllib3", "trafilatura", "chromadb", "primp"):
        logging.getLogger(lib).setLevel(logging.ERROR)


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Research Agent — search, extract, analyze, report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "Recent developments in AI safety"
  python cli.py "Competitive landscape of OpenAI" --verbose
  python cli.py --interactive
        """,
    )

    parser.add_argument(
        "goal",
        nargs="?",
        help="Research goal/query (or use --interactive mode)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode (prompt for query)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip memory cache, force fresh research",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Check for required API keys
    if not os.environ.get("GROQ_API_KEY"):
        print("[ERROR] GROQ_API_KEY environment variable not set.")
        print("   Get a free key at: https://console.groq.com")
        print("   Or add it to a .env file in this directory.")
        sys.exit(1)

    if not os.environ.get("TAVILY_API_KEY"):
        print("[ERROR] TAVILY_API_KEY environment variable not set.")
        print("   Get a free key at: https://tavily.com (1000 searches/month)")
        print("   Or add it to a .env file in this directory.")
        sys.exit(1)

    # Get the goal
    if args.interactive or not args.goal:
        print("\nAutonomous Research Agent")
        print("=" * 40)
        goal = input("\nEnter your research query: ").strip()
        if not goal:
            print("No query provided. Exiting.")
            sys.exit(0)
    else:
        goal = args.goal.strip()
        if not goal:
            print("[ERROR] Empty research query provided.")
            sys.exit(1)

    # Run the agent
    from graph import run_research_agent

    try:
        final_state = run_research_agent(goal, skip_cache=args.no_cache)

        # Print the structured summary to console
        summary = final_state.get("structured_summary")
        if summary:
            print("\n" + "=" * 60)
            print("STRUCTURED SUMMARY")
            print("=" * 60)

            exec_summary = summary.get("executive_summary", "")
            if exec_summary:
                print(f"\n{exec_summary}\n")

            key_points = summary.get("key_points", [])
            if key_points:
                print("Key Points:")
                for i, point in enumerate(key_points, 1):
                    print(f"  {i}. {point}")

            insights = summary.get("actionable_insights", [])
            if insights:
                print("\nActionable Insights:")
                for insight in insights:
                    print(f"  • {insight}")

            print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nResearch interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logging.getLogger(__name__).error(f"Agent failed: {e}", exc_info=True)
        print(f"\n[ERROR] Agent failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
