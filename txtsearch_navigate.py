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

from typing import List, Union, Dict, Any, Tuple
import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer

# --- Module Constants ---
SENTENCE_BOUNDARY_PATTERN = r'(?<=[.!?])\s+'
MIN_TOKEN_LENGTH = 3
EMBEDDING_EPSILON = 1e-8
DEFAULT_SENTENCES_PER_PASSAGE = 3
DEFAULT_TOP_K = 5
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Helper utilities ---

def _split_into_passages(text: str, sentences_per_passage: int = DEFAULT_SENTENCES_PER_PASSAGE) -> List[str]:
    """Split a document into short passages (groups of sentences).

    Uses a simple sentence boundary regex to avoid heavy deps. Each passage
    contains up to `sentences_per_passage` sentences (sliding window not used
    for simplicity).
    """
    # Split on sentence boundaries
    sentences = re.split(SENTENCE_BOUNDARY_PATTERN, text.strip())
    passages = []
    for i in range(0, len(sentences), sentences_per_passage):
        passage = " ".join(sentences[i:i + sentences_per_passage]).strip()
        if passage:
            passages.append(passage)
    return passages


def _simple_highlight(passage: str, query: str) -> str:
    """Return passage with query terms highlighted using **bold** markers.

    This is a straightforward, case-insensitive word match highlighting.
    Short stop-words (length < MIN_TOKEN_LENGTH) are skipped to reduce noise.
    """
    tokens = re.findall(r"\w+", query.lower())
    tokens = [t for t in tokens if len(t) >= MIN_TOKEN_LENGTH]
    if not tokens:
        return passage

    # escape tokens for regex; use boundaries to avoid substring-only matches
    pattern = r"\b(?:" + r"|".join(re.escape(t) for t in set(tokens)) + r")\b"

    def repl(m):
        # preserve original casing while adding highlight
        return "**" + m.group(0) + "**"

    highlighted = re.sub(pattern, repl, passage, flags=re.IGNORECASE)
    return highlighted


# --- Input validation helpers ---

def _validate_parameters(query: str, sentences_per_passage: int, top_k: int) -> None:
    """Validate function parameters and raise ValueError if invalid.
    
    Parameters:
    - query: must be a non-empty string
    - sentences_per_passage: must be a positive integer
    - top_k: must be a positive integer
    
    Raises: ValueError if any parameter is invalid
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(sentences_per_passage, int) or sentences_per_passage <= 0:
        raise ValueError("sentences_per_passage must be a positive integer")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


def _normalize_documents(documents: Union[str, List[str], Dict[str, str]]) -> List[Tuple[str, str]]:
    """Convert various document formats into a standardized list of (doc_id, text) tuples.
    
    Parameters:
    - documents: can be:
        * str: either a file path (if exists) or a single document string
        * list: list of document strings, will be assigned auto-generated IDs
        * dict: mapping of document IDs/names to text
    
    Returns: list of (doc_id, text) tuples for all non-empty documents
    
    Raises: ValueError if documents are of unsupported type or file cannot be read
    """
    doc_items = []

    if isinstance(documents, str):
        # Treat as either a file path (if exists) or as a single document string
        if os.path.exists(documents):
            try:
                with open(documents, "r", encoding="utf-8") as f:
                    text = f.read()
            except (OSError, UnicodeDecodeError) as exc:
                raise ValueError(f"Failed to read file '{documents}': {exc}") from exc
            if text.strip():
                doc_items.append((os.path.basename(documents), text))
        else:
            if documents.strip():
                doc_items.append(("doc_0", documents))
    elif isinstance(documents, dict):
        for k, v in documents.items():
            if v is None:
                continue
            text = str(v)
            if text.strip():
                doc_items.append((str(k), text))
    elif isinstance(documents, list):
        for i, v in enumerate(documents):
            if v is None:
                continue
            text = str(v)
            if text.strip():
                doc_items.append((f"doc_{i}", text))
    else:
        raise ValueError("Unsupported documents type; provide str/list/dict")

    return doc_items


# --- Primary implementation ---

def txtsearch_navigate(documents: Union[str, List[str], Dict[str, str]],
                       query: str,
                       model_name: str = DEFAULT_MODEL_NAME,
                       top_k: int = DEFAULT_TOP_K,
                       sentences_per_passage: int = DEFAULT_SENTENCES_PER_PASSAGE) -> List[Dict[str, Any]]:
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
    """
    # Validate input parameters
    _validate_parameters(query, sentences_per_passage, top_k)

    # Normalize documents into a list of (doc_id, text)
    doc_items = _normalize_documents(documents)
    if not doc_items:
        return []

    # Build passages across all documents for fine-grained ranking + highlighting
    passages = []  # list of (doc_id, passage_text)
    for doc_id, text in doc_items:
        for passage in _split_into_passages(text, sentences_per_passage):
            passages.append((doc_id, passage))

    if not passages:
        return []

    # Load embedding model
    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:
        raise ValueError(f"Failed to load model '{model_name}': {exc}") from exc

    # Create embeddings for passages (US3.1 Generate Embeddings)
    passage_texts = [p[1] for p in passages]
    passage_embeddings = model.encode(passage_texts, convert_to_numpy=True, show_progress_bar=False)
    
    # Normalize embeddings for cosine similarity speed
    norms = np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
    passage_embeddings = passage_embeddings / (norms + EMBEDDING_EPSILON)

    # Embed query and normalize
    query_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
    query_emb = query_emb / (np.linalg.norm(query_emb) + EMBEDDING_EPSILON)

    # Compute cosine similarity scores between query and passages (US3.2 Natural-Language Search)
    scores = (passage_embeddings @ query_emb).astype(float)

    # Rank results (US3.3 Ranked Search Results)
    top_idx = np.argsort(scores)[::-1][:top_k]

    # Build results with highlighting (US3.4 Highlight Matching Sections)
    results = []
    for idx in top_idx:
        doc_id, passage = passages[int(idx)]
        score = float(scores[int(idx)])
        highlighted = _simple_highlight(passage, query)
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
