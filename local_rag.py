from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Tuple


DATA_DIR = os.path.join(os.getcwd(), "data")
INDEX_PATH = os.path.join(DATA_DIR, "local_index.json")


def _ensure_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


_word_re = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _word_re.findall(text)]


def _split_chunks(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + max_chars, n)
        chunk = text[i:j]
        chunks.append(chunk)
        if j >= n:
            break
        i = j - overlap
        if i < 0:
            i = 0
    return chunks


def _load() -> Dict:
    _ensure_dir()
    if not os.path.exists(INDEX_PATH):
        return {"files": [], "chunks": [], "df": {}, "N": 0}
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"files": [], "chunks": [], "df": {}, "N": 0}


def _save(state: Dict):
    _ensure_dir()
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f)


def add_file(file_name: str, content_bytes: bytes) -> str:
    """Add a file's text content into the local index. Returns a pseudo file_id."""
    # Best-effort UTF-8 decode; ignore errors
    try:
        text = content_bytes.decode("utf-8", errors="ignore")
    except Exception:
        text = ""
    if not text.strip():
        return "local:empty"

    chunks = _split_chunks(text)
    state = _load()

    # Simple pseudo id
    file_id = f"local:{len(state['files'])+1}"
    state["files"].append({"id": file_id, "name": file_name, "num_chunks": len(chunks)})

    # Update df counts and store chunks
    for ch in chunks:
        toks = sorted(set(_tokenize(ch)))
        for t in toks:
            state["df"][t] = int(state["df"].get(t, 0)) + 1
        state["chunks"].append({"file_id": file_id, "text": ch})

    state["N"] = len(state["chunks"])
    _save(state)
    return file_id


def _idf(state: Dict, term: str) -> float:
    # log( (N + 1) / (df + 1) ) + 1
    import math

    N = max(1, int(state.get("N", 1)))
    df = int(state.get("df", {}).get(term, 0))
    return math.log((N + 1) / (df + 1)) + 1.0


def search(query: str, k: int = 5) -> List[Tuple[str, float]]:
    """Return top-k chunk texts with scores."""
    state = _load()
    if state.get("N", 0) == 0:
        return []
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    # Compute naive TF and TF-IDF for query
    from collections import Counter
    import math

    q_tf = Counter(q_tokens)
    q_weights: Dict[str, float] = {}
    for term, tf in q_tf.items():
        q_weights[term] = (1.0 + math.log(tf)) * _idf(state, term)
    q_norm = math.sqrt(sum(w * w for w in q_weights.values())) or 1.0

    def score_chunk(text: str) -> float:
        toks = _tokenize(text)
        tf = Counter(toks)
        weights: Dict[str, float] = {}
        for term, f in tf.items():
            weights[term] = (1.0 + math.log(f)) * _idf(state, term)
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        # cosine similarity on overlapping terms only
        dot = 0.0
        for t, q_w in q_weights.items():
            w = weights.get(t)
            if w:
                dot += q_w * w
        return dot / (q_norm * norm)

    scored: List[Tuple[str, float]] = []
    for ch in state["chunks"]:
        s = score_chunk(ch["text"])  # type: ignore
        if s > 0:
            scored.append((ch["text"], s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]

