from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "faiss is required. Please install faiss-cpu in your environment."
    ) from e


# Simple FAISS-backed embedding store persisted under ./data
DATA_DIR = os.path.join(os.getcwd(), "data")
INDEX_BIN = os.path.join(DATA_DIR, "faiss.index")
META_JSON = os.path.join(DATA_DIR, "faiss_meta.json")


_CLIENT = None
_EMBED_MODEL = None
_INDEX: Optional[faiss.Index] = None  # type: ignore
_DIM: Optional[int] = None
_META: Dict = {}


def _ensure_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


def init(client, embedding_model_name: str = "text-embedding-3-small"):
    """Bind OpenAI client and load or create FAISS index."""
    global _CLIENT, _EMBED_MODEL, _INDEX, _DIM, _META
    _CLIENT = client
    _EMBED_MODEL = embedding_model_name
    _ensure_dir()
    # Load metadata
    if os.path.exists(META_JSON):
        try:
            with open(META_JSON, "r", encoding="utf-8") as f:
                _META = json.load(f)
        except Exception:
            _META = {}
    else:
        _META = {}

    if os.path.exists(INDEX_BIN):
        _INDEX = faiss.read_index(INDEX_BIN)
        _DIM = int(_META.get("dim", 1536)) if _META else None
        # If dim missing, infer from index (best-effort)
        if not _DIM:
            try:
                _DIM = _INDEX.d  # type: ignore[attr-defined]
            except Exception:
                _DIM = 1536
    else:
        # Lazily create index on first add when we know dim
        _INDEX = None
        _DIM = None
        if not _META:
            _META = {"chunks": [], "next_id": 0, "dim": None}


def _save():
    global _INDEX, _META
    _ensure_dir()
    if _INDEX is not None:
        faiss.write_index(_INDEX, INDEX_BIN)
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(_META, f)


def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + max_chars, n)
        out.append(text[i:j])
        if j >= n:
            break
        i = max(0, j - overlap)
    return out


def _embed_texts(texts: List[str]) -> np.ndarray:
    from openai import OpenAI  # type: ignore

    if _CLIENT is None or _EMBED_MODEL is None:
        raise RuntimeError("embedding_store not initialized. Call init(client, embedding_model_name) first.")
    assert isinstance(_CLIENT, OpenAI)  # for type hints only

    # OpenAI embeddings API supports batch inputs
    resp = _CLIENT.embeddings.create(model=_EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    arr = np.array(vecs, dtype=np.float32)
    # Normalize to use inner product as cosine similarity
    faiss.normalize_L2(arr)
    return arr


def _ensure_index(dim: int):
    global _INDEX, _DIM
    if _INDEX is None:
        _INDEX = faiss.IndexFlatIP(dim)
        _DIM = dim
        _META["dim"] = dim


def add_file(file_name: str, content_bytes: bytes) -> str:
    """Add file content to FAISS index. Returns a pseudo file_id."""
    global _META, _INDEX, _DIM
    # Decode text
    try:
        text = content_bytes.decode("utf-8", errors="ignore")
    except Exception:
        text = ""
    if not text.strip():
        return "local:empty"

    chunks = _chunk_text(text)
    if not chunks:
        return "local:empty"

    # Embed chunks
    embs = _embed_texts(chunks)
    dim = int(embs.shape[1])
    _ensure_index(dim)
    if _DIM != dim:
        # In case an existing index has different dim, this is a hard error
        raise RuntimeError(f"Embedding dimension mismatch: index dim={_DIM}, new dim={dim}")

    start_id = int(_META.get("next_id", 0))
    ids = list(range(start_id, start_id + len(chunks)))

    # Add to index
    _INDEX.add(embs)

    # Persist chunk metadata in order
    for i, ch in enumerate(chunks):
        _META.setdefault("chunks", []).append(
            {
                "id": ids[i],
                "file_name": file_name,
                "file_id": _meta_file_id(file_name),
                "text": ch,
            }
        )
    _META["next_id"] = ids[-1] + 1 if ids else start_id
    _save()
    return _META["chunks"][-1]["file_id"] if _META.get("chunks") else "local:empty"


def _meta_file_id(file_name: str) -> str:
    # Stable pseudo-id based on order: count of files with same name so far
    cnt = 1
    if "chunks" in _META:
        for ch in _META["chunks"]:
            if ch.get("file_name") == file_name:
                cnt = max(cnt, int(ch.get("id", 0)) + 1)
    return f"local:{file_name}:{cnt}"


def search(query: str, k: int = 5) -> List[Tuple[str, float]]:
    if not query.strip():
        return []
    if _INDEX is None or (_META.get("chunks") is None or len(_META.get("chunks")) == 0):
        return []
    q = _embed_texts([query])
    D, I = _INDEX.search(q, min(k, len(_META["chunks"])) )
    idxs = I[0]
    scores = D[0]
    results: List[Tuple[str, float]] = []
    for pos, score in zip(idxs, scores):
        if pos < 0:
            continue
        try:
            ch = _META["chunks"][int(pos)]
            results.append((ch.get("text", ""), float(score)))
        except Exception:
            continue
    return results

