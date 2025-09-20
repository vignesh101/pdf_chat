from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "faiss is required. Please install faiss-cpu in your environment."
    ) from e


# Separate FAISS-backed embedding store for Web chat, persisted under ./data/web
DATA_DIR = os.path.join(os.getcwd(), "data", "web")
INDEX_BIN = os.path.join(DATA_DIR, "faiss.index")
META_JSON = os.path.join(DATA_DIR, "faiss_meta.json")


_CLIENT = None
_EMBED_MODEL = None
_INDEX: Optional[faiss.Index] = None  # type: ignore
_DIM: Optional[int] = None
_META: Dict = {}
_INIT_DONE: bool = False


def _ensure_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


def init(client, embedding_model_name: str = "text-embedding-3-small"):
    """Bind OpenAI client and load or create FAISS index for web chat."""
    global _CLIENT, _EMBED_MODEL, _INDEX, _DIM, _META, _INIT_DONE
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

    # Ensure schema keys exist
    if not isinstance(_META, dict):
        _META = {}
    _META.setdefault("chunks", [])
    _META.setdefault("next_id", 0)
    _META.setdefault("dim", None)
    _META.setdefault("files", [])  # list of {file_id(url), name(title or url), created_at}

    if os.path.exists(INDEX_BIN):
        _INDEX = faiss.read_index(INDEX_BIN)
        _DIM = int(_META.get("dim", 1536)) if _META else None
        if not _DIM:
            try:
                _DIM = _INDEX.d  # type: ignore[attr-defined]
            except Exception:
                _DIM = 1536
    else:
        _INDEX = None
        _DIM = None
        if not _META:
            _META = {"chunks": [], "next_id": 0, "dim": None, "files": []}
    _INIT_DONE = True


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
        raise RuntimeError("web_embedding_store not initialized. Call init(client, embedding_model_name) first.")
    assert isinstance(_CLIENT, OpenAI)  # for type hints only

    resp = _CLIENT.embeddings.create(model=_EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    arr = np.array(vecs, dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr


def _ensure_index(dim: int):
    global _INDEX, _DIM
    if _INDEX is None:
        _INDEX = faiss.IndexFlatIP(dim)
        _DIM = dim
    _META["dim"] = dim


def add_page(url: str, title: Optional[str], text: str) -> str:
    """Add a fetched web page's text into the web FAISS store.

    Uses URL as file_id and title (or URL) as name for metadata; returns the URL.
    """
    global _META, _INDEX, _DIM
    if not (text or "").strip():
        return url or "web:empty"

    chunks = _chunk_text(text)
    if not chunks:
        return url or "web:empty"

    # Embed chunks
    embs = _embed_texts(chunks)
    dim = int(embs.shape[1])
    _ensure_index(dim)
    if _DIM != dim:
        raise RuntimeError(f"Embedding dimension mismatch: index dim={_DIM}, new dim={dim}")

    start_id = int(_META.get("next_id", 0))
    ids = list(range(start_id, start_id + len(chunks)))

    # Add to index
    _INDEX.add(embs)

    # Register file metadata entry (deduplicate by URL)
    try:
        from time import time
        created_at = int(time())
    except Exception:
        created_at = 0

    try:
        _META.setdefault("files", [])
        exists = any((f.get("file_id") == url) for f in _META["files"])
        if not exists:
            _META["files"].append({"file_id": url, "name": (title or url or ""), "created_at": created_at})
    except Exception:
        pass

    # Persist chunk metadata in order
    for i, ch in enumerate(chunks):
        _META.setdefault("chunks", []).append(
            {
                "id": ids[i],
                "file_name": (title or url or ""),
                "file_id": url,
                "text": ch,
            }
        )
    _META["next_id"] = ids[-1] + 1 if ids else start_id
    _save()
    return url


def search_with_meta(query: str, k: int = 5) -> List[Dict[str, Any]]:
    if not query.strip():
        return []
    if _INDEX is None or (_META.get("chunks") is None or len(_META.get("chunks")) == 0):
        return []
    q = _embed_texts([query])
    D, I = _INDEX.search(q, min(k, len(_META["chunks"])) )
    idxs = I[0]
    scores = D[0]
    out: List[Dict[str, Any]] = []
    for pos, score in zip(idxs, scores):
        if pos < 0:
            continue
        try:
            ch = _META["chunks"][int(pos)]
            out.append({
                "chunk_id": ch.get("id"),
                "file_name": ch.get("file_name"),
                "file_id": ch.get("file_id"),
                "text": ch.get("text", ""),
                "score": float(score),
            })
        except Exception:
            continue
    return out


def get_status() -> Dict[str, Any]:
    try:
        chunks = _META.get("chunks") if isinstance(_META, dict) else []
        count = len(chunks) if isinstance(chunks, list) else 0
    except Exception:
        count = 0
    dim = _DIM if _DIM is not None else (_META.get("dim") if isinstance(_META, dict) else None)
    index_exists = os.path.exists(INDEX_BIN)
    meta_exists = os.path.exists(META_JSON)
    index_ready = _INDEX is not None and count >= 0
    try:
        seen = set()
        for ch in (chunks or []):
            fid = ch.get("file_id") or (ch.get("file_name") or "")
            if fid:
                seen.add(fid)
        files_count = len(seen)
    except Exception:
        files_count = 0
    return {
        "init": bool(_INIT_DONE),
        "index_exists": bool(index_exists),
        "meta_exists": bool(meta_exists),
        "index_ready": bool(index_ready),
        "chunks_count": int(count),
        "files_count": int(files_count),
        "dim": int(dim) if isinstance(dim, (int,)) else None,
    }


def clear_all() -> Dict[str, Any]:
    """Delete the web FAISS index and metadata, and reset state."""
    global _INDEX, _DIM, _META
    try:
        if os.path.exists(INDEX_BIN):
            os.remove(INDEX_BIN)
    except Exception:
        pass
    try:
        if os.path.exists(META_JSON):
            os.remove(META_JSON)
    except Exception:
        pass
    _INDEX = None
    _DIM = None
    _META = {"chunks": [], "next_id": 0, "dim": None, "files": []}
    return get_status()
