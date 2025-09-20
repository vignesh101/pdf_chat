from __future__ import annotations

import os
from flask import Flask, render_template, request, jsonify, session

from config_loader import load_config
from openai_client import build_openai_client
import embedding_store


def create_app() -> Flask:
    cfg = load_config()
    app = Flask(__name__)
    app.config['SECRET_KEY'] = cfg.secret_key or os.environ.get('FLASK_SECRET', 'dev-secret-change-me')

    # Build OpenAI client and hold it in app context (may be None if not configured)
    client = build_openai_client(cfg)
    app.config['OPENAI_CLIENT'] = client
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

        # Update local history
        hist = session.get('chat_history', [])
        hist.append({'id': f'user-{len(hist)+1}', 'role': 'user', 'text': message})
        hist.append({
            'id': f'assistant-{len(hist)+1}',
            'role': 'assistant',
            'text': answer,
            'sources': ctx_chunks_meta,
        })
        # keep last 50
        session['chat_history'] = hist[-50:]
        return jsonify({'ok': True, 'messages': session['chat_history']})

    @app.get('/messages')
    def get_messages():
        # Return local history
        return jsonify({'ok': True, 'messages': session.get('chat_history', [])})

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
                'sources': ctx_chunks_meta,
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
            session['chat_history'] = hist[-50:]
            # clear pending
            session.pop('pending_assistant', None)
            return jsonify({'ok': True, 'messages': session['chat_history']})
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
