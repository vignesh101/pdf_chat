from __future__ import annotations

import os
from flask import Flask, render_template, request, jsonify, session
from typing import Any, Dict, List

from config_loader import load_config
from openai_client import build_openai_client, build_httpx_client
import embedding_store


def create_app() -> Flask:
    cfg = load_config()
    app = Flask(__name__)
    app.config['SECRET_KEY'] = cfg.secret_key or os.environ.get('FLASK_SECRET', 'dev-secret-change-me')

    # Prefer server-side sessions if Flask-Session is available.
    # Falls back gracefully to client-side secure cookies.
    app.config['SESSION_BACKEND'] = 'client'
    try:
        # Lazy import so the app still runs without the extra dependency
        from flask_session import Session as FlaskSession  # type: ignore
        sessions_dir = os.path.join(os.getcwd(), 'data', 'flask_sessions')
        os.makedirs(sessions_dir, exist_ok=True)
        # Allow overriding signer via env if desired
        use_signer_env = os.environ.get('SESSION_USE_SIGNER', '').strip().lower() in ('1', 'true', 'yes', 'on')
        app.config.update(
            SESSION_TYPE='filesystem',
            SESSION_FILE_DIR=sessions_dir,
            SESSION_PERMANENT=False,
            # NOTE: Some Flask-Session versions return a bytes value for the
            # signed session id when used with newer Werkzeug, which causes
            # a TypeError in set_cookie. Server-side sessions don’t require
            # signing since the cookie only stores a random id, so disable it
            # for maximum compatibility across environments.
            SESSION_USE_SIGNER=bool(use_signer_env),
            SESSION_COOKIE_NAME='dc_session',
        )
        FlaskSession(app)
        app.config['SESSION_BACKEND'] = 'server'
    except Exception:
        # Keep using signed cookie sessions
        pass

    # Session size management knobs (smaller limits for cookie-backed sessions)
    if app.config['SESSION_BACKEND'] == 'server':
        app.config.setdefault('MAX_HISTORY_MESSAGES', 50)
        app.config.setdefault('MAX_HISTORY_TEXT_LEN', 16000)  # effectively unlimited for typical usage
        app.config.setdefault('MAX_SOURCE_PREVIEW', 1000)
    else:
        # Conservative caps to stay well under ~4KB cookie limits
        app.config.setdefault('MAX_HISTORY_MESSAGES', 12)
        app.config.setdefault('MAX_HISTORY_TEXT_LEN', 1500)
        app.config.setdefault('MAX_SOURCE_PREVIEW', 300)

    def _compact_sources(sources: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
        if not sources:
            return []
        max_prev = int(app.config.get('MAX_SOURCE_PREVIEW', 300))
        slim: List[Dict[str, Any]] = []
        for s in sources:
            if not isinstance(s, dict):
                continue
            # Preserve common meta, trim the text preview
            text = s.get('text') or ''
            slim.append({
                'file_id': s.get('file_id'),
                'file_name': s.get('file_name'),
                'score': s.get('score'),
                'text': text[:max_prev],
            })
        return slim

    def _store_history(hist: List[Dict[str, Any]]) -> None:
        """Clamp history length and text sizes before saving to session.
        This prevents over-large client-side cookies while remaining no-op-ish on server sessions.
        """
        max_n = int(app.config.get('MAX_HISTORY_MESSAGES', 50))
        max_txt = int(app.config.get('MAX_HISTORY_TEXT_LEN', 16000))

        def clamp_msg(m: Dict[str, Any]) -> Dict[str, Any]:
            text = (m.get('text') or '')
            slim: Dict[str, Any] = {
                'id': m.get('id'),
                'role': m.get('role'),
                'text': text[:max_txt],
            }
            if m.get('role') != 'user':
                srcs = m.get('sources')
                if isinstance(srcs, list):
                    slim['sources'] = _compact_sources(srcs)
            return slim

        session['chat_history'] = [clamp_msg(m) for m in (hist[-max_n:])]

    # Build OpenAI + HTTP clients and hold them in app context (may be None if not configured)
    client = build_openai_client(cfg)
    http_client = build_httpx_client(cfg)
    app.config['OPENAI_CLIENT'] = client
    app.config['HTTP_CLIENT'] = http_client
    app.config['MODEL_NAME'] = cfg.model_name
    # Initialize local FAISS embedding store only if client is available
    try:
        if client is not None:
            embedding_store.init(client, embedding_model_name=cfg.embedding_model_name)
    except Exception:
        # Do not block app startup; search/upload will surface errors
        pass

    @app.route('/')
    def index():
        # Single mode: Chat with local FAISS RAG
        session.setdefault('chat_history', [])
        session.setdefault('web_chat_history', [])
        thread_id = None
        return render_template('index.html',
                               model_name=cfg.model_name,
                               proxy_url=cfg.proxy_url,
                               base_url=cfg.openai_base_url,
                               disable_ssl=cfg.disable_ssl,
                               client_ready=bool(app.config.get('OPENAI_CLIENT')),
                               thread_id=thread_id)

    @app.post('/upload')
    def upload():
        from file_ingest import extract_text

        if 'file' not in request.files:
            return jsonify({'ok': False, 'error': 'No file part'}), 400
        files = request.files.getlist('file')
        saved_ids = []
        os.makedirs(os.path.join(os.getcwd(), 'uploads'), exist_ok=True)
        # Ensure client is configured before attempting embeddings
        if app.config.get('OPENAI_CLIENT') is None:
            return jsonify({'ok': False, 'error': 'OpenAI client not configured. Set OPENAI_API_KEY or update config.toml.'}), 400
        try:
            for f in files:
                if not f.filename:
                    continue
                # Extract text (handles PDFs locally if pypdf/PyPDF2 is installed)
                raw = f.read()
                text = extract_text(f.filename, raw)
                file_id = embedding_store.add_text(f.filename, text)
                saved_ids.append(file_id)
            return jsonify({'ok': True, 'file_ids': saved_ids})
        except Exception as e:
            # Surface helpful message for missing PDF libs
            msg = str(e)
            if 'PDF support requires' in msg:
                return jsonify({'ok': False, 'error': msg}), 400
            return jsonify({'ok': False, 'error': msg}), 500

    @app.post('/chat')
    def chat():
        data = request.get_json(silent=True) or {}
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'ok': False, 'error': 'Empty message'}), 400
        # Retrieve context locally (embeddings + FAISS) and call Chat Completions
        k_ctx = 5
        try:
            ctx_chunks_meta = embedding_store.search_with_meta(message, k=k_ctx)
        except Exception as e:
            # If vector search fails (e.g., embeddings unavailable), continue without context
            ctx_chunks_meta = []
        context_text = "\n\n".join([f"[Snippet {i+1}]\n{it['text']}" for i, it in enumerate(ctx_chunks_meta)])
        system_prompt = (
            "You are a helpful assistant. Use the provided snippets to answer. "
            "If the answer isn't in the snippets, say you aren't sure.\n\n"
            + (f"Snippets:\n{context_text}" if context_text else "No snippets available.")
        )
        # Ensure client is configured
        if app.config.get('OPENAI_CLIENT') is None:
            return jsonify({'ok': False, 'error': 'OpenAI client not configured. Set OPENAI_API_KEY or update config.toml.'}), 500
        try:
            resp = client.chat.completions.create(
                model=cfg.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
            )
            answer = resp.choices[0].message.content if resp and resp.choices else ""
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

        # Update local history (compacted for session)
        hist = session.get('chat_history', [])
        hist.append({'id': f'user-{len(hist)+1}', 'role': 'user', 'text': message})
        hist.append({
            'id': f'assistant-{len(hist)+1}',
            'role': 'assistant',
            'text': answer,
            'sources': _compact_sources(ctx_chunks_meta),
        })
        _store_history(hist)
        return jsonify({'ok': True, 'messages': session.get('chat_history', [])})

    @app.get('/messages')
    def get_messages():
        # Return local history
        return jsonify({'ok': True, 'messages': session.get('chat_history', [])})

    # --- Web Chat: retrieve data from internet and chat ---
    def _html_to_text(html: str) -> str:
        import re
        from html import unescape
        # Remove script/style
        html = re.sub(r'<script[\s\S]*?</script>', ' ', html, flags=re.I)
        html = re.sub(r'<style[\s\S]*?</style>', ' ', html, flags=re.I)
        # Replace breaks/paragraphs/headings with newlines
        html = re.sub(r'</?(?:br|p|div|h[1-6]|li|tr|td|th|ul|ol|table)[^>]*>', '\n', html, flags=re.I)
        # Strip all other tags
        html = re.sub(r'<[^>]+>', ' ', html)
        # Unescape entities and normalize whitespace
        text = unescape(html)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _fetch_urls_text(urls: List[str], limit_bytes: int = 200_000) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        http = app.config.get('HTTP_CLIENT')
        if http is None:
            return out
        for u in urls:
            if not isinstance(u, str):
                continue
            url = u.strip()
            if not url:
                continue
            try:
                r = http.get(url, follow_redirects=True)
                ctype = (r.headers.get('content-type') or '').lower()
                if 'text' not in ctype and 'html' not in ctype:
                    continue
                content = r.text
                text = _html_to_text(content)[:limit_bytes]
                title = None
                try:
                    import re as _re
                    m = _re.search(r'<title[^>]*>(.*?)</title>', r.text, flags=_re.I|_re.S)
                    if m:
                        title = (m.group(1) or '').strip()
                except Exception:
                    pass
                out.append({'url': url, 'title': title, 'text': text})
            except Exception:
                # Skip failures but continue others
                pass
        return out

    def _extract_urls_from_text(s: str) -> List[str]:
        import re
        urls = re.findall(r'(https?://[^\s]+)', s)
        # Basic cleanup for trailing punctuation
        cleaned = []
        for u in urls:
            cleaned.append(u.rstrip(').,;\'\"]'))
        return cleaned

    @app.post('/webchat')
    def webchat():
        data = request.get_json(silent=True) or {}
        message = (data.get('message') or '').strip()
        extra_urls = data.get('urls') or []
        if not message and not extra_urls:
            return jsonify({'ok': False, 'error': 'Empty message'}), 400
        urls = list(dict.fromkeys((_extract_urls_from_text(message) + [u for u in extra_urls if isinstance(u, str)])))[:5]

        # Fetch web context
        web_snippets = _fetch_urls_text(urls)
        context_text = "\n\n".join([
            f"[Web {i+1}] {it.get('title') or it['url']}\n{it['text']}"
            for i, it in enumerate(web_snippets)
        ])
        system_prompt = (
            "You are a helpful assistant with access to web page excerpts. "
            "Use the provided web content to answer accurately. If the answer isn't present, say you aren't sure.\n\n"
            + (f"Web Content:\n{context_text}" if context_text else "No web content available.")
        )
        client = app.config.get('OPENAI_CLIENT')
        if client is None:
            return jsonify({'ok': False, 'error': 'OpenAI client not configured. Set OPENAI_API_KEY or update config.toml.'}), 500
        try:
            resp = client.chat.completions.create(
                model=cfg.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
            )
            answer = resp.choices[0].message.content if resp and resp.choices else ""
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

        # Update web history
        hist = session.get('web_chat_history', [])
        hist.append({'id': f'user-{len(hist)+1}', 'role': 'user', 'text': message})
        hist.append({
            'id': f'assistant-{len(hist)+1}',
            'role': 'assistant',
            'text': answer,
            'sources': [{'file_id': it.get('url'), 'file_name': it.get('title') or it.get('url'), 'score': None, 'text': it.get('text', '')[:app.config.get('MAX_SOURCE_PREVIEW', 300)]} for it in web_snippets],
        })
        session['web_chat_history'] = hist
        return jsonify({'ok': True, 'messages': session.get('web_chat_history', [])})

    @app.post('/webchat_stream')
    def webchat_stream():
        from flask import Response, stream_with_context
        data = request.get_json(silent=True) or {}
        message = (data.get('message') or '').strip()
        extra_urls = data.get('urls') or []
        if not message and not extra_urls:
            return jsonify({'ok': False, 'error': 'Empty message'}), 400
        urls = list(dict.fromkeys((_extract_urls_from_text(message) + [u for u in extra_urls if isinstance(u, str)])))[:5]

        web_snippets = _fetch_urls_text(urls)
        context_text = "\n\n".join([
            f"[Web {i+1}] {it.get('title') or it['url']}\n{it['text']}"
            for i, it in enumerate(web_snippets)
        ])
        system_prompt = (
            "You are a helpful assistant with access to web page excerpts. "
            "Use the provided web content to answer accurately. If the answer isn't present, say you aren't sure.\n\n"
            + (f"Web Content:\n{context_text}" if context_text else "No web content available.")
        )
        client = app.config.get('OPENAI_CLIENT')
        if client is None:
            return jsonify({'ok': False, 'error': 'OpenAI client not configured. Set OPENAI_API_KEY or update config.toml.'}), 500

        try:
            session['pending_web_assistant'] = {
                'sources': [{'file_id': it.get('url'), 'file_name': it.get('title') or it.get('url'), 'score': None, 'text': it.get('text', '')[:app.config.get('MAX_SOURCE_PREVIEW', 300)]} for it in web_snippets],
            }
        except Exception:
            pass

        def generate():
            try:
                stream = client.chat.completions.create(
                    model=cfg.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message},
                    ],
                    stream=True,
                )
                for chunk in stream:
                    try:
                        delta = chunk.choices[0].delta
                        content = getattr(delta, 'content', None) if delta is not None else None
                    except Exception:
                        content = None
                    if content:
                        yield content
            except Exception as e:
                yield f"\n[ERROR] {str(e)}"

        return Response(stream_with_context(generate()), mimetype='text/plain; charset=utf-8')

    @app.post('/webchat/commit')
    def webchat_commit():
        data = request.get_json(silent=True) or {}
        question = (data.get('question') or '').strip()
        answer = (data.get('answer') or '').strip()
        if not question or not answer:
            return jsonify({'ok': False, 'error': 'question and answer required'}), 400
        try:
            hist = session.get('web_chat_history', [])
            pending = session.get('pending_web_assistant') or {}
            sources = pending.get('sources') or []
            hist.append({'id': f'user-{len(hist)+1}', 'role': 'user', 'text': question})
            hist.append({'id': f'assistant-{len(hist)+1}', 'role': 'assistant', 'text': answer, 'sources': sources})
            session['web_chat_history'] = hist
            session.pop('pending_web_assistant', None)
            return jsonify({'ok': True, 'messages': session.get('web_chat_history', [])})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    @app.get('/web/messages')
    def web_get_messages():
        return jsonify({'ok': True, 'messages': session.get('web_chat_history', [])})

    @app.post('/web/clear')
    def web_clear_messages():
        session['web_chat_history'] = []
        return jsonify({'ok': True, 'messages': []})

    @app.post('/chat_stream')
    def chat_stream():
        from flask import Response, stream_with_context
        data = request.get_json(silent=True) or {}
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'ok': False, 'error': 'Empty message'}), 400

        # Build context from RAG if available
        k_ctx = 5
        try:
            ctx_chunks_meta = embedding_store.search_with_meta(message, k=k_ctx)
        except Exception:
            ctx_chunks_meta = []
        context_text = "\n\n".join([f"[Snippet {i+1}]\n{it['text']}" for i, it in enumerate(ctx_chunks_meta)])
        system_prompt = (
            "You are a helpful assistant. Use the provided snippets to answer. "
            "If the answer isn't in the snippets, say you aren't sure.\n\n"
            + (f"Snippets:\n{context_text}" if context_text else "No snippets available.")
        )

        client = app.config.get('OPENAI_CLIENT')
        if client is None:
            return jsonify({'ok': False, 'error': 'OpenAI client not configured. Set OPENAI_API_KEY or update config.toml.'}), 500

        # Stash sources for later commit. Do not mutate chat_history here because
        # streamed responses finalize headers before the generator completes.
        try:
            session['pending_assistant'] = {
                'sources': _compact_sources(ctx_chunks_meta),
            }
        except Exception:
            pass

        def generate():
            acc = []
            try:
                stream = client.chat.completions.create(
                    model=cfg.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message},
                    ],
                    stream=True,
                )
                for chunk in stream:
                    try:
                        delta = chunk.choices[0].delta
                        content = getattr(delta, 'content', None) if delta is not None else None
                    except Exception:
                        content = None
                    if content:
                        acc.append(content)
                        yield content
            except Exception as e:
                yield f"\n[ERROR] {str(e)}"

        return Response(stream_with_context(generate()), mimetype='text/plain; charset=utf-8')

    @app.post('/chat/commit')
    def chat_commit():
        """Commit the assistant's streamed reply to session history.
        Expects JSON: { question: str, answer: str }
        """
        data = request.get_json(silent=True) or {}
        question = (data.get('question') or '').strip()
        answer = (data.get('answer') or '').strip()
        if not question or not answer:
            return jsonify({'ok': False, 'error': 'question and answer required'}), 400
        try:
            hist = session.get('chat_history', [])
            pending = session.get('pending_assistant') or {}
            sources = pending.get('sources') or []
            # Append user then assistant
            hist.append({
                'id': f'user-{len(hist)+1}',
                'role': 'user',
                'text': question,
            })
            hist.append({
                'id': f'assistant-{len(hist)+1}',
                'role': 'assistant',
                'text': answer,
                'sources': sources,
            })
            _store_history(hist)
            # clear pending
            session.pop('pending_assistant', None)
            return jsonify({'ok': True, 'messages': session.get('chat_history', [])})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    @app.post('/clear')
    def clear_messages():
        # Clear local chat history for this session
        session['chat_history'] = []
        return jsonify({'ok': True, 'messages': []})

    @app.get('/status')
    def status():
        client_ready = bool(app.config.get('OPENAI_CLIENT'))
        try:
            store = embedding_store.get_status()
        except Exception:
            store = {"init": False, "index_exists": False, "meta_exists": False, "index_ready": False, "chunks_count": 0, "dim": None}
        return jsonify({
            'ok': True,
            'client_ready': client_ready,
            'model_name': cfg.model_name,
            'index': store,
        })

    @app.get('/health')
    def health():
        client_ready = bool(app.config.get('OPENAI_CLIENT'))
        try:
            store = embedding_store.get_status()
        except Exception:
            store = {"index_ready": False}
        ready = client_ready and bool(store.get('index_ready'))
        code = 200 if client_ready else 200  # liveness ok; readiness reflected in payload
        return jsonify({'ok': True, 'client_ready': client_ready, 'ready': ready}), code

    @app.post('/index/clear')
    def index_clear():
        try:
            st = embedding_store.clear_index()
            return jsonify({'ok': True, 'index': st})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    @app.post('/index/rebuild')
    def index_rebuild():
        if app.config.get('OPENAI_CLIENT') is None:
            return jsonify({'ok': False, 'error': 'OpenAI client not configured. Set OPENAI_API_KEY or update config.toml.'}), 400
        try:
            st = embedding_store.rebuild_index()
            return jsonify({'ok': True, 'index': st})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    @app.post('/index/rebuild_async')
    def index_rebuild_async():
        if app.config.get('OPENAI_CLIENT') is None:
            return jsonify({'ok': False, 'error': 'OpenAI client not configured. Set OPENAI_API_KEY or update config.toml.'}), 400
        try:
            st = embedding_store.rebuild_index_async()
            return jsonify({'ok': True, 'index': st})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    @app.post('/index/clear_all')
    def index_clear_all():
        try:
            st = embedding_store.clear_all()
            return jsonify({'ok': True, 'index': st})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    @app.post('/index/rebuild_cancel')
    def index_rebuild_cancel():
        try:
            st = embedding_store.cancel_rebuild()
            return jsonify({'ok': True, 'index': st})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    @app.get('/files')
    def list_files():
        try:
            files = embedding_store.get_files()
            return jsonify({'ok': True, 'files': files})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    @app.post('/files/delete')
    def delete_file():
        data = request.get_json(silent=True) or {}
        file_id = data.get('file_id')
        if not file_id:
            return jsonify({'ok': False, 'error': 'file_id required'}), 400
        try:
            st = embedding_store.remove_file(file_id)
            return jsonify({'ok': True, 'index': st})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
