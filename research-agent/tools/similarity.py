"""
tools/similarity.py — Embedding similarity tool for dedup and relevance filtering.

Uses the shared embeddings.py module (ChromaDB ONNX all-MiniLM-L6-v2)
instead of sentence-transformers + PyTorch.
"""

import sys
import os
import logging
import numpy as np

from tools.embeddings import embed_texts, embed_query

logger = logging.getLogger(__name__)


def compute_relevance_scores(query: str, texts: list[str]) -> list[float]:
    """Compute cosine similarity between a query and a list of texts.

    Args:
        query: The reference query string.
        texts: List of text strings to score against the query.

    Returns:
        List of cosine similarity scores (0.0 to 1.0), same length as texts.
    """
    if not texts:
        return []

    try:
        query_embedding = embed_query(query)
        text_embeddings = embed_texts(texts)

        # Cosine similarity (embeddings are already L2-normalized)
        scores = np.dot(text_embeddings, query_embedding).tolist()
        return scores

    except Exception as e:
        logger.error(f"Relevance scoring failed: {e}")
        # Return neutral scores so nothing gets filtered out
        return [0.5] * len(texts)


def find_duplicates(
    texts: list[str],
    threshold: float = 0.85,
) -> list[set[int]]:
    """Find groups of near-duplicate texts using embedding similarity.

    Args:
        texts: List of text strings to check for duplicates.
        threshold: Cosine similarity threshold above which texts are considered duplicates.

    Returns:
        List of sets, where each set contains indices of duplicate texts.
        Only groups with 2+ members are returned.
    """
    if len(texts) < 2:
        return []

    try:
        embeddings = embed_texts(texts)

        # Pairwise cosine similarity matrix
        sim_matrix = np.dot(embeddings, embeddings.T)

        # Find duplicate groups via union-find
        n = len(texts)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] >= threshold:
                    union(i, j)

        # Collect groups
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = set()
            groups[root].add(i)

        # Return only groups with duplicates
        return [g for g in groups.values() if len(g) > 1]

    except Exception as e:
        logger.error(f"Duplicate detection failed: {e}")
        return []


def filter_by_relevance(
    query: str,
    items: list[dict],
    text_key: str = "content",
    threshold: float = 0.25,
) -> list[dict]:
    """Filter items by relevance to query, keeping only those above threshold.

    Each item gets a 'relevance_score' field added.

    Args:
        query: The reference query.
        items: List of dicts, each containing text in text_key field.
        text_key: Key in each dict that contains the text to score.
        threshold: Minimum relevance score to keep.

    Returns:
        Filtered and scored list of items, sorted by relevance (descending).
    """
    if not items:
        return []

    texts = [item.get(text_key, "") for item in items]
    scores = compute_relevance_scores(query, texts)

    # Add scores and filter
    scored_items = []
    for item, score in zip(items, scores):
        item_copy = dict(item)
        item_copy["relevance_score"] = round(score, 4)
        if score >= threshold:
            scored_items.append(item_copy)

    # Sort by relevance descending
    scored_items.sort(key=lambda x: x["relevance_score"], reverse=True)

    logger.info(
        f"Relevance filter: {len(scored_items)}/{len(items)} items kept "
        f"(threshold={threshold})"
    )
    return scored_items


def deduplicate_items(
    items: list[dict],
    text_key: str = "content",
    threshold: float = 0.85,
) -> list[dict]:
    """Remove near-duplicate items, keeping the first (or highest-scored) in each group.

    Args:
        items: List of dicts with text content.
        text_key: Key containing text to compare.
        threshold: Similarity threshold for dedup.

    Returns:
        Deduplicated list of items.
    """
    if len(items) < 2:
        return items

    texts = [item.get(text_key, "") for item in items]
    dup_groups = find_duplicates(texts, threshold)

    # Mark indices to remove (keep first in each group, or highest scored)
    indices_to_remove = set()
    for group in dup_groups:
        sorted_group = sorted(group)
        # If items have relevance_score, keep the highest; otherwise keep first
        if "relevance_score" in items[sorted_group[0]]:
            best_idx = max(group, key=lambda i: items[i].get("relevance_score", 0))
        else:
            best_idx = sorted_group[0]
        for idx in group:
            if idx != best_idx:
                indices_to_remove.add(idx)

    deduped = [item for i, item in enumerate(items) if i not in indices_to_remove]

    logger.info(
        f"Dedup: {len(items)} -> {len(deduped)} items "
        f"({len(indices_to_remove)} duplicates removed)"
    )
    return deduped
