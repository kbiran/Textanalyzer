"""
txtsearch_navigate.py

Provides a search function that satisfies the following user stories:
- US3.1 Generate Embeddings
- US3.2 Natural-Language Search
- US3.3 Ranked Search Results
- US3.4 Highlight Matching Sections

The user's requested function name contains characters invalid in Python identifiers: "txtsearch&Navigaye'".

Comments are included throughout the code.
"""

from typing import List, Union, Dict, Any
import os
import re
import math

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Helper utilities ---

def _split_into_passages(text: str, sentences_per_passage: int = 3) -> List[str]:
    """Split a document into short passages (groups of sentences).

    Uses a simple sentence boundary regex to avoid heavy deps. Each passage
    contains up to `sentences_per_passage` sentences (sliding window not used
    for simplicity).
    """
    # naive sentence split: keep punctuation as sentence boundary
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    passages = []
    for i in range(0, len(sentences), sentences_per_passage):
        passage = " ".join(sentences[i:i + sentences_per_passage]).strip()
        if passage:
            passages.append(passage)
    return passages


def _simple_highlight(passage: str, query: str) -> str:
    """Return passage with query terms highlighted using **bold** markers.

    This is a straightforward, case-insensitive word match highlighting.
    Short stop-words (length < 3) are skipped to reduce noise.
    """
    tokens = re.findall(r"\w+", query.lower())
    tokens = [t for t in tokens if len(t) >= 3]
    if not tokens:
        return passage

    # escape tokens for regex
    pattern = r"(" + r"|".join(re.escape(t) for t in set(tokens)) + r")"

    def repl(m):
        # preserve original casing while adding highlight
        return "**" + m.group(0) + "**"

    highlighted = re.sub(pattern, repl, passage, flags=re.IGNORECASE)
    return highlighted


# --- Primary implementation ---

def txtsearch_navigate(documents: Union[str, List[str], Dict[str, str]],
                       query: str,
                       model_name: str = "all-MiniLM-L6-v2",
                       top_k: int = 5,
                       sentences_per_passage: int = 3) -> List[Dict[str, Any]]:
    """Search documents semantically and return ranked, highlighted snippets.

    Parameters:
    - documents: either a single string (text or file path), a list of text strings,
      or a dict mapping document ids/names to text.
    - query: user natural-language query string.
    - model_name: sentence-transformers model used for embeddings.
    - top_k: number of top passages to return.
    - sentences_per_passage: number of sentences grouped into one passage.

    Returns: list of results sorted by relevance; each result is a dict:
      {"doc_id": str, "passage": str, "score": float, "highlight": str}

    Comments are included in the source to explain each step.
    """
    # Normalize documents into a list of (doc_id, text)
    doc_items = []

    if isinstance(documents, str):
        # treat as either a file path (if exists) or as a single document string
        if os.path.exists(documents):
            with open(documents, "r", encoding="utf-8") as f:
                text = f.read()
            doc_items.append((os.path.basename(documents), text))
        else:
            doc_items.append(("doc_0", documents))
    elif isinstance(documents, dict):
        for k, v in documents.items():
            doc_items.append((str(k), str(v)))
    elif isinstance(documents, list):
        for i, v in enumerate(documents):
            doc_items.append((f"doc_{i}", str(v)))
    else:
        raise ValueError("Unsupported documents type; provide str/list/dict")

    # Build passages across all documents for fine-grained ranking + highlighting
    passages = []  # list of (doc_id, passage_text)
    for doc_id, text in doc_items:
        for passage in _split_into_passages(text, sentences_per_passage):
            passages.append((doc_id, passage))

    if not passages:
        return []

    # Load embedding model (user can change `model_name`)
    model = SentenceTransformer(model_name)

    # Create embeddings for passages (US3.1 Generate Embeddings)
    passage_texts = [p[1] for p in passages]
    passage_embeddings = model.encode(passage_texts, convert_to_numpy=True, show_progress_bar=False)

    # Normalize embeddings for cosine similarity speed
    eps = 1e-8
    norms = np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
    passage_embeddings = passage_embeddings / (norms + eps)

    # Embed query
    query_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
    query_emb = query_emb / (np.linalg.norm(query_emb) + eps)

    # Compute cosine similarity scores between query and passages (US3.2 Natural-Language Search)
    scores = (passage_embeddings @ query_emb).astype(float)

    # Rank results (US3.3 Ranked Search Results)
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        doc_id, passage = passages[int(idx)]
        score = float(scores[int(idx)])
        highlighted = _simple_highlight(passage, query)  # US3.4 Highlight Matching Sections
        results.append({
            "doc_id": doc_id,
            "passage": passage,
            "score": score,
            "highlight": highlighted,
        })

    return results


# Expose the function under the user-requested name (contains special characters).
# This lets callers access the function by using: globals()["txtsearch&Navigaye'"](...)
globals()["txtsearch&Navigaye'"] = txtsearch_navigate


# Simple demo when run as a script
if __name__ == "__main__":
    sample_docs = {
        "terms": "These are the terms of service. The user agrees to pay fees. In case of dispute, arbitration applies.",
        "privacy": "We collect data for service improvement. We do not sell personal data. Users may opt-out.",
        "guide": "To search documents with this tool, pass documents and a natural language query. The tool returns highlighted passages."
    }

    query = "how do I opt out of data collection"
    print("Running demo search...\n")
    res = txtsearch_navigate(sample_docs, query, top_k=4)
    for r in res:
        print(f"Doc: {r['doc_id']} | Score: {r['score']:.4f}")
        print(r['highlight'])
        print("---")
