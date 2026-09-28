# Autonomous Research Agent

An autonomous AI agent that accepts a research query, searches the web, extracts and deduplicates content, and produces a structured, actionable research report — all with visible planning, tool-use logging, and self-correction on failures.

## Features

- **Visible Planning Trace** — The agent produces a legible plan before executing any actions
- **Multi-tool Orchestration** — Web search (Tavily), page fetch/extraction (Jina Reader + trafilatura), embedding similarity (ONNX), export, and memory
- **Self-correction** — Handles fetch failures, zero search results, empty content, malformed LLM output, and API rate limits gracefully
- **Autonomous Source Selection** — Picks search strategy (news/tech/company/general) based on query type
- **Parallel Fetching** — Concurrent page downloads using thread pool
- **Dedup & Relevance** — Embedding-based near-duplicate detection and relevance filtering
- **Memory/Cache** — SQLite-backed memory of past searches with semantic similarity matching (auto-pruned to last 50 entries)
- **Structured Output** — Markdown report with key points, findings, insights, references, and limitations
- **Rate Limit Resilience** — Automatic retry with exponential backoff for Groq API rate limits

## Architecture
<img width="1010" height="1146" alt="image" src="https://github.com/user-attachments/assets/dc3593e6-288d-4c87-acce-39533d0ed548" />

## Tech Stack

| Component | Choice |
|---|---|
| Orchestration | LangGraph (explicit nodes/edges) |
| LLM | Groq (openai/gpt-oss-20b) with retry + backoff |
| Web Search | Tavily (free tier, 1000/month) |
| Content Extraction | Jina Reader API (primary) + trafilatura (fallback) |
| Embeddings | ChromaDB ONNX runtime (all-MiniLM-L6-v2) |
| Memory | SQLite (auto-pruned, max 50 entries) |
| Export | Markdown |

## Setup

### 1. Clone and install dependencies

```bash
cd research-agent
pip install -r requirements.txt
```

### 2. Set your API keys

```bash
# Copy the template
cp .env.example .env

# Edit .env and add your keys:
GROQ_API_KEY=your_key_here       # Free at https://console.groq.com
TAVILY_API_KEY=your_key_here     # Free at https://tavily.com (1000 searches/month)
```

## Usage

### Single query

```bash
python cli.py "Recent developments in AI safety"
```

### Interactive mode

```bash
python cli.py --interactive
```

### Verbose logging

```bash
python cli.py "KV-cache compression in LLMs" --verbose
```

### Skip memory cache (force fresh search)

```bash
python cli.py "KV-cache compression in LLMs" --no-cache
```

### Example queries

```bash
# News-style (exercises recency-weighted search)
python cli.py "Latest developments in quantum computing this week"

# Technical (exercises tech docs strategy)
python cli.py "Approaches to KV-cache compression in LLMs"

# Company research (exercises company analysis strategy + failure handling)
python cli.py "Competitive landscape of Anthropic"
```

## Running Tests

```bash
pytest tests/ -v --tb=short
```

## Project Structure

```
research-agent/
├── requirements.txt
├── .env.example
├── graph.py              # LangGraph state machine (orchestration)
├── state.py              # Shared state schema (TypedDict)
├── cli.py                # CLI entry point
├── tools/
│   ├── llm.py            # Groq LLM wrapper with retry + backoff
│   ├── web_search.py     # Tavily search wrapper
│   ├── page_fetch.py     # Jina Reader API + trafilatura fallback
│   ├── similarity.py     # Embedding similarity (dedup + relevance)
│   ├── embeddings.py     # ChromaDB ONNX embedding functions
│   ├── export.py         # Markdown report writer
│   └── memory.py         # SQLite memory store (auto-pruned)
├── nodes/
│   ├── query_understanding.py
│   ├── planning.py
│   ├── source_selection.py
│   ├── search_execution.py
│   ├── fetch_extract.py
│   ├── dedup_filter.py
│   ├── synthesis.py
│   └── export_node.py
├── memory/
│   └── past_searches.db  # Created at runtime
├── output/               # Generated reports
```

## Self-Correction Mechanisms

1. **Zero search results** → Progressively broadens query (strip qualifiers, use fewer terms)
2. **Fetch failures (timeout/404/SSL)** → Logs error, skips source, continues with remaining
3. **Empty content (<100 chars)** → Treated as failed fetch, same handling
4. **Jina Reader rate limit (429)** → Falls back to trafilatura extraction
5. **Groq API rate limit (429)** → Automatic retry with exponential backoff (up to 3 attempts)
6. **Tavily quota exhaustion** → Clear error message surfaced to user
7. **Too many failures (>50% sources)** → Triggers replan with reformulated queries
8. **Malformed LLM JSON** → Re-prompts once with schema restated, falls back to plain text
9. **Coverage too thin** → Replans up to 2 times with broadened search terms (stale errors cleared each loop)

## Design Decisions

- **ChromaDB ONNX vs sentence-transformers**: Uses ChromaDB's built-in ONNX runtime for the same all-MiniLM-L6-v2 model, avoiding the ~2GB PyTorch dependency
- **LangGraph over plain state machine**: Explicit graph structure makes the control flow visible and the conditional edges (replan, cache) easy to reason about
- **Tavily over DuckDuckGo**: Tavily provides pre-extracted content alongside search results, reducing network calls and improving reliability. DuckDuckGo was replaced due to rate limiting issues.
- **Jina Reader as primary extractor**: Handles PDFs, JS-heavy sites, and paywalled content better than trafilatura alone. Falls back to trafilatura if Jina is rate-limited.
- **Centralized LLM helper**: All Groq API calls go through `tools/llm.py` which provides automatic retry with exponential backoff for rate limits
- **Balanced relevance threshold (0.25)**: Filters out clearly irrelevant content while keeping borderline-useful sources; the LLM handles final synthesis-time filtering
- **Smart content truncation**: Long sources keep first 5000 + last 3000 chars (preserving introductions and conclusions) instead of just the beginning
