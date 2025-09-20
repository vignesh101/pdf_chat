from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple, Any
import threading

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
_INIT_DONE: bool = False
_REBUILD: Dict[str, Any] = {"in_progress": False, "total": 0, "current": 0, "error": None, "cancelled": False}
_REBUILD_CANCEL = threading.Event()


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

    # Ensure schema keys exist
    if not isinstance(_META, dict):
        _META = {}
    _META.setdefault("chunks", [])
    _META.setdefault("next_id", 0)
    _META.setdefault("dim", None)
    _META.setdefault("files", [])  # list of {file_id, name, created_at}

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


def _next_file_id(file_name: str) -> str:
    """Generate a stable new file_id based on number of files with same name."""
    try:
        same = [f for f in _META.get("files", []) if f.get("name") == file_name]
        suffix = len(same) + 1
    except Exception:
        suffix = 1
    return f"local:{file_name}:{suffix}"


def clear_index() -> Dict[str, Any]:
    """Remove the FAISS index file and reset in-memory index state.

    Metadata (chunks) is retained so the index can be rebuilt later.
    """
    global _INDEX, _DIM
    try:
        if os.path.exists(INDEX_BIN):
            os.remove(INDEX_BIN)
    except Exception:
        pass
    _INDEX = None
    _DIM = None
    try:
        if isinstance(_META, dict):
            _META["dim"] = None
            _save()
    except Exception:
        pass
    return get_status()


def rebuild_index(batch_size: int = 256) -> Dict[str, Any]:
    """Rebuild FAISS index from stored chunks in metadata.

    Requires that the client has been initialized. Embeds stored chunk texts
    in order to preserve alignment of FAISS ids with metadata order.
    """
    global _INDEX, _DIM
    if _CLIENT is None:
        raise RuntimeError("OpenAI client not configured; cannot rebuild index.")
    chunks = _META.get("chunks") if isinstance(_META, dict) else None
    if not chunks:
        # Nothing to build
        return get_status()

    texts: List[str] = [str(ch.get("text", "")) for ch in chunks]
    # Remove any empty texts (but must keep alignment). To preserve alignment,
    # we will keep empty strings; embeddings API can accept empty? Better to
    # replace empty with a single space.
    texts = [t if t.strip() else " " for t in texts]

    # Embed first batch to determine dim
    first = texts[: min(batch_size, len(texts))]
    arr = _embed_texts(first)
    dim = int(arr.shape[1])
    new_index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(arr)
    new_index.add(arr)

    # Remaining batches
    i = len(first)
    while i < len(texts):
        j = min(i + batch_size, len(texts))
        part = _embed_texts(texts[i:j])
        faiss.normalize_L2(part)
        new_index.add(part)
        i = j

    _INDEX = new_index
    _DIM = dim
    try:
        _META["dim"] = dim
        _save()
    except Exception:
        pass
    return get_status()


def rebuild_index_async(batch_size: int = 256) -> Dict[str, Any]:
    """Start an asynchronous rebuild of the FAISS index.

    Returns current status; if already in progress, does not start another.
    """
    global _REBUILD, _INDEX, _DIM
    if _CLIENT is None:
        raise RuntimeError("OpenAI client not configured; cannot rebuild index.")
    chunks = _META.get("chunks") if isinstance(_META, dict) else None
    total = len(chunks or [])
    if _REBUILD.get("in_progress"):
        return get_status()

    _REBUILD = {"in_progress": True, "total": total, "current": 0, "error": None, "cancelled": False}
    _REBUILD_CANCEL.clear()

    def _worker():
        global _REBUILD, _INDEX, _DIM
        try:
            if not chunks:
                _INDEX = None
                _DIM = None
                _REBUILD.update({"in_progress": False})
                return
            texts: List[str] = [str(ch.get("text", "")) for ch in chunks]
            texts = [t if t.strip() else " " for t in texts]

            first = texts[: min(batch_size, len(texts))]
            arr = _embed_texts(first)
            dim = int(arr.shape[1])
            new_index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(arr)
            new_index.add(arr)
            _REBUILD["current"] = len(first)

            i = len(first)
            while i < len(texts):
                j = min(i + batch_size, len(texts))
                part = _embed_texts(texts[i:j])
                faiss.normalize_L2(part)
                new_index.add(part)
                i = j
                _REBUILD["current"] = i
                if _REBUILD_CANCEL.is_set():
                    # cancellation requested: do not commit the new index
                    _REBUILD.update({"in_progress": False, "cancelled": True})
                    return

            _INDEX = new_index
            _DIM = dim
            try:
                _META["dim"] = dim
                _save()
            except Exception:
                pass
            _REBUILD.update({"in_progress": False})
        except Exception as e:
            _REBUILD.update({"error": str(e), "in_progress": False})

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    return get_status()


def add_text(file_name: str, text: str) -> str:
    """Add raw text to FAISS index. Returns a pseudo file_id."""
    global _META, _INDEX, _DIM
    if not (text or "").strip():
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

    # Register file metadata entry
    try:
        from time import time
        created_at = int(time())
    except Exception:
        created_at = 0
    file_id = _next_file_id(file_name)
    try:
        _META.setdefault("files", [])
        _META["files"].append({"file_id": file_id, "name": file_name, "created_at": created_at})
    except Exception:
        pass

    # Persist chunk metadata in order
    for i, ch in enumerate(chunks):
        _META.setdefault("chunks", []).append(
            {
                "id": ids[i],
                "file_name": file_name,
                "file_id": file_id,
                "text": ch,
            }
        )
    _META["next_id"] = ids[-1] + 1 if ids else start_id
    _save()
    return file_id if _META.get("chunks") else "local:empty"


def add_file(file_name: str, content_bytes: bytes) -> str:
    """Backwards-compatible: decode bytes and index. Prefer add_text() when possible."""
    try:
        text = content_bytes.decode("utf-8", errors="ignore")
    except Exception:
        text = ""
    return add_text(file_name, text)


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


def search_with_meta(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Return top-k chunk dicts: {text, score, file_name, file_id, chunk_id}.

    Does not modify the store. Uses the same FAISS search as `search` but
    includes metadata for UI display (e.g., sources panel).
    """
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
    """Return a lightweight status of the local FAISS store."""
    try:
        chunks = _META.get("chunks") if isinstance(_META, dict) else []
        count = len(chunks) if isinstance(chunks, list) else 0
    except Exception:
        count = 0
    dim = _DIM if _DIM is not None else (_META.get("dim") if isinstance(_META, dict) else None)
    index_exists = os.path.exists(INDEX_BIN)
    meta_exists = os.path.exists(META_JSON)
    index_ready = _INDEX is not None and count >= 0
    # Count distinct files by file_id if available, else file_name
    files_count = 0
    try:
        seen = set()
        for ch in (chunks or []):
            fid = ch.get("file_id") or (ch.get("file_name") or "")
            if fid:
                seen.add(fid)
        files_count = len(seen)
    except Exception:
        files_count = 0
    # Rebuild status
    try:
        rb = dict(_REBUILD)
        total = int(rb.get("total") or 0)
        current = int(rb.get("current") or 0)
        progress = (current / total * 100.0) if total > 0 else (100.0 if index_ready else 0.0)
        rb_out = {
            "in_progress": bool(rb.get("in_progress")),
            "total": total,
            "current": current,
            "progress": float(progress),
            "error": rb.get("error"),
            "cancelled": bool(rb.get("cancelled")),
        }
    except Exception:
        rb_out = {"in_progress": False, "total": 0, "current": 0, "progress": 0.0, "error": None, "cancelled": False}

    # Build recent files (up to 10) from files meta
    recent_files: List[Dict[str, Any]] = []
    try:
        files_meta = _META.get("files", []) if isinstance(_META, dict) else []
        counts: Dict[str, int] = {}
        for ch in (chunks or []):
            fid = ch.get("file_id") or ""
            if fid:
                counts[fid] = counts.get(fid, 0) + 1
        ordered = sorted(files_meta, key=lambda f: int(f.get("created_at", 0)), reverse=True)
        for f in ordered[:10]:
            fid = f.get("file_id")
            recent_files.append({
                "file_id": fid,
                "name": f.get("name"),
                "chunks": counts.get(fid, 0),
                "created_at": f.get("created_at"),
            })
    except Exception:
        recent_files = []

    return {
        "init": bool(_INIT_DONE),
        "index_exists": bool(index_exists),
        "meta_exists": bool(meta_exists),
        "index_ready": bool(index_ready),
        "chunks_count": int(count),
        "files_count": int(files_count),
        "dim": int(dim) if isinstance(dim, (int,)) else None,
        "rebuild": rb_out,
        "recent_files": recent_files,
    }


def clear_all() -> Dict[str, Any]:
    """Delete both index and metadata files and reset in-memory state."""
    global _INDEX, _DIM, _META, _REBUILD
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
    _REBUILD = {"in_progress": False, "total": 0, "current": 0, "error": None, "cancelled": False}
    return get_status()


def cancel_rebuild() -> Dict[str, Any]:
    """Signal the async rebuild worker to cancel as soon as possible."""
    _REBUILD_CANCEL.set()
    return get_status()


def get_files() -> List[Dict[str, Any]]:
    """Return all uploaded files with chunk counts and timestamps, newest first."""
    try:
        files_meta = _META.get("files", [])
        counts: Dict[str, int] = {}
        for ch in _META.get("chunks", []):
            fid = ch.get("file_id")
            counts[fid] = counts.get(fid, 0) + 1
        out: List[Dict[str, Any]] = []
        for f in files_meta:
            out.append({
                "file_id": f.get("file_id"),
                "name": f.get("name"),
                "created_at": f.get("created_at"),
                "chunks": counts.get(f.get("file_id"), 0),
            })
        out.sort(key=lambda x: int(x.get("created_at") or 0), reverse=True)
        return out
    except Exception:
        return []


def remove_file(file_id: str) -> Dict[str, Any]:
    """Remove all chunks for a file and update index accordingly."""
    global _INDEX, _DIM
    if not file_id:
        return get_status()
    try:
        _META["chunks"] = [ch for ch in _META.get("chunks", []) if ch.get("file_id") != file_id]
        _META["files"] = [f for f in _META.get("files", []) if f.get("file_id") != file_id]
        _save()
    except Exception:
        pass

    remaining = len(_META.get("chunks", []))
    if remaining == 0:
        try:
            if os.path.exists(INDEX_BIN):
                os.remove(INDEX_BIN)
        except Exception:
            pass
        _INDEX = None
        _DIM = None
        _META["dim"] = None
        _save()
        return get_status()

    # If client is available, rebuild; otherwise clear index to avoid inconsistency
    if _CLIENT is None:
        try:
            if os.path.exists(INDEX_BIN):
                os.remove(INDEX_BIN)
        except Exception:
            pass
        _INDEX = None
        _DIM = None
        _META["dim"] = None
        _save()
        return get_status()
    else:
        return rebuild_index()
