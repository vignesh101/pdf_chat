from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from openai import OpenAI


STATE_DIR = os.path.join(os.getcwd(), "data")
STATE_PATH = os.path.join(STATE_DIR, "state.json")


def _ensure_state_dir():
    if not os.path.exists(STATE_DIR):
        os.makedirs(STATE_DIR, exist_ok=True)


def _load_state() -> Dict:
    _ensure_state_dir()
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state: Dict):
    _ensure_state_dir()
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def ensure_vector_store(client: OpenAI, name: str = "Document Chat Vector Store") -> str:
    state = _load_state()
    vs_id = state.get("vector_store_id")
    if vs_id:
        return vs_id
    vs = client.vector_stores.create(name=name)
    state["vector_store_id"] = vs.id
    _save_state(state)
    return vs.id


def ensure_assistant(client: OpenAI, model_name: str, instructions: Optional[str] = None) -> str:
    state = _load_state()
    assistant_id = state.get("assistant_id")
    vector_store_id = ensure_vector_store(client)

    if assistant_id:
        # Make sure the assistant is connected to our vector store
        try:
            client.assistants.update(
                assistant_id,
                tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
                tools=[{"type": "file_search"}],
            )
        except Exception:
            pass
        return assistant_id

    assistant = client.assistants.create(
        name="Document Chat Assistant",
        instructions=(
            instructions
            or """You are a helpful assistant. Use the provided files to answer questions accurately. If information is not in the files, say you are unsure and suggest how to find it."""
        ),
        model=model_name,
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
    )
    state["assistant_id"] = assistant.id
    _save_state(state)
    return assistant.id


def attach_files_to_vector_store(client: OpenAI, file_ids: List[str]):
    vector_store_id = ensure_vector_store(client)
    if not file_ids:
        return
    # Batch attach files to the vector store
    client.vector_stores.file_batches.create(
        vector_store_id=vector_store_id,
        file_ids=file_ids,
    )


def upload_file(client: OpenAI, file_path: str) -> str:
    with open(file_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="assistants")
    return uploaded.id


def ensure_thread_state(session_store: Dict) -> str:
    thread_id = session_store.get("thread_id")
    if thread_id:
        return thread_id
    # Thread is created lazily by OpenAI on first message/run; create explicitly
    thread = client_from_session(session_store).threads.create()
    session_store["thread_id"] = thread.id
    return thread.id


def client_from_session(session_store: Dict) -> OpenAI:
    # The session store should have the client injected from the Flask app context
    client = session_store.get("_client")
    if client is None:
        raise RuntimeError("OpenAI client not bound to session.")
    return client


def run_chat_turn(client: OpenAI, assistant_id: str, thread_id: str, user_message: str, timeout_s: float = 120.0) -> List[Dict]:
    # Add user message to thread
    client.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message,
    )
    # Create a run
    run = client.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )

    # Poll until completed or requires_action/error
    start = time.time()
    while True:
        run = client.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if run.status in ("completed", "cancelled", "failed", "expired"):
            break
        if run.status == "requires_action":
            # For this app, we do not implement tool invocation; just break
            break
        if time.time() - start > timeout_s:
            break
        time.sleep(1.0)

    # Fetch all messages, newest first
    msgs = client.threads.messages.list(thread_id=thread_id, order="desc", limit=20)

    formatted: List[Dict] = []
    for m in reversed(list(msgs)):
        parts: List[str] = []
        for c in m.content:
            if c.type == "text":
                parts.append(c.text.value)
            elif c.type == "image_file":
                parts.append(f"[image:{c.image_file.file_id}]")
            elif c.type == "file_path":
                parts.append(f"[file:{c.file_path.file_id}]")
        formatted.append({
            "id": m.id,
            "role": m.role,
            "text": "\n\n".join(parts) if parts else "",
            "created_at": getattr(m, "created_at", None),
        })
    return formatted

