# Design Write-Up — Autonomous Research Agent

## Design Decisions

### Orchestration: LangGraph
I chose LangGraph over a plain Python state machine because the graph structure directly maps to the architecture spec's state diagram. Each node is a self-contained function, and the conditional edges (memory cache, coverage replan) are expressed declaratively rather than buried in if/else chains. This makes the "legible plan" requirement trivially visible — the graph *is* the plan's skeleton.

### Embeddings: ChromaDB ONNX vs sentence-transformers
The architecture spec originally called for `sentence-transformers` with `all-MiniLM-L6-v2`. I replaced this with ChromaDB's built-in ONNX runtime, which ships the same model but runs on ONNX Runtime instead of PyTorch. The trade-off: identical embedding quality with ~2GB less disk/install overhead. The `tools/embeddings.py` module provides `embed_texts()`, `embed_query()`, and `get_chroma_embedding_function()` as a clean API boundary.

### Search: Tavily with pre-extracted content
Initially DuckDuckGo was used (zero-signup), but it proved unreliable due to aggressive rate limiting that caused frequent search failures. I switched to Tavily, which provides two key advantages: (1) high-quality search results with pre-extracted `raw_content` that can be passed directly to the pipeline, reducing redundant network calls, and (2) a generous free tier of 1,000 searches/month. The agent surfaces clear error messages if the Tavily quota is exhausted.

### Content Extraction: Jina Reader API + trafilatura fallback
Jina Reader (`r.jina.ai`) is used as the primary content extractor because it handles PDFs, JavaScript-heavy sites, and complex layouts that trafilatura struggles with. When Jina is rate-limited (HTTP 429) or returns insufficient content, the agent falls back to trafilatura. This two-tier approach maximizes extraction success rates.

### LLM Calls: Centralized retry with backoff
All LLM interactions go through a shared `tools/llm.py` helper (`groq_chat()`) that wraps the Groq SDK with automatic retry and exponential backoff for rate limits (429 errors). This avoids duplicating retry logic across the three LLM-calling nodes (query understanding, planning, synthesis) and makes the model name configurable in one place.

### Self-Correction Strategy
Rather than a single retry mechanism, I built correction at multiple levels:
1. **Query parsing**: JSON parse failure → fallback to raw goal text
2. **Search**: Zero results → progressively broaden query (strip qualifiers, fewer terms)
3. **Fetch**: Per-URL status classification (ok/timeout/404/empty) → skip failed, continue with rest. Jina 429 → trafilatura fallback.
4. **Synthesis**: Malformed JSON → re-prompt once with schema restated → plain text fallback
5. **Coverage**: If synthesis reports "thin" coverage or <2 usable sources → replan with reformulated queries (up to 2 retries). Stale errors from previous loops are cleared to keep the final report clean.
6. **API rate limits**: Groq 429 → exponential backoff retry (2s, 4s, 8s). Tavily quota → clear user-facing error message.

This layered approach means the agent never crashes on recoverable errors, and always produces *some* output — even if degraded.

### Memory: SQLite with auto-pruning
Past searches are stored with their embeddings for semantic cache hits. To prevent unbounded database growth, entries are automatically pruned to the most recent 50 after each write. A `--no-cache` CLI flag allows users to force fresh searches when needed.

### Relevance Filtering
The relevance threshold is set to 0.25 (cosine similarity against the query embedding). This is a balanced value that filters out clearly irrelevant content (e.g., unrelated school websites that sometimes appear in search results) while keeping borderline-useful sources. The deduplication threshold is 0.85 for near-duplicate detection using a union-find algorithm.

### Content Truncation
Long source content is truncated to 8,000 characters using a "beginning + end" strategy: first 5,000 chars + last 3,000 chars. This preserves both the introduction/context and the conclusions/key findings that often appear at the end of articles, rather than naively cutting at a fixed point.

## Limitations

1. **Tavily free tier cap**: 1,000 searches/month. Heavy use requires a paid plan or switching to a different search backend.
2. **Context window constraints**: Very long pages are truncated to 8,000 chars before synthesis. A chunking + map-reduce approach would handle longer documents better.
3. **No iterative refinement**: The agent doesn't ask the user clarifying questions. A human-in-the-loop checkpoint after planning would improve result quality.
4. **Embedding model size**: all-MiniLM-L6-v2 is a small model. Larger models (e.g., BGE-large) would give better relevance scoring, at the cost of speed.
5. **Single LLM model**: All LLM calls use the same model (openai/gpt-oss-20b via Groq). Using a smaller model for planning and a larger one for synthesis could optimize cost/quality.
6. **Jina Reader free tier**: Jina Reader has usage limits. Under heavy concurrent load, some requests may be rate-limited and fall back to trafilatura.

## What I'd Change With More Time

1. **Async throughout**: Replace `ThreadPoolExecutor` with proper `asyncio` for fetching, and make the graph nodes async-compatible.
2. **Chunk-level dedup**: Currently dedup operates on full page content. Chunking pages into paragraphs and deduplicating at the chunk level would be more precise.
3. **Citation grounding**: Cross-reference each claim in the synthesis against specific source passages to ensure factual grounding.
4. **Evaluation harness**: Build a small eval suite with known-answer queries to measure retrieval recall and synthesis accuracy.
5. **Streaming output**: Stream the synthesis as it's generated rather than waiting for the full report.
6. **Multi-modal sources**: Add PDF parsing, academic paper search (Semantic Scholar API), and code repository search as additional tools.
