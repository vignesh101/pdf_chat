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

    # Build OpenAI client and hold it in app context
    client = build_openai_client(cfg)
    app.config['OPENAI_CLIENT'] = client
    app.config['MODEL_NAME'] = cfg.model_name
    # Initialize local FAISS embedding store
    embedding_store.init(client, embedding_model_name=cfg.embedding_model_name)

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
                               thread_id=thread_id)

    @app.post('/upload')
    def upload():
        if 'file' not in request.files:
            return jsonify({'ok': False, 'error': 'No file part'}), 400
        files = request.files.getlist('file')
        saved_ids = []
        tmp_dir = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            for f in files:
                if not f.filename:
                    continue
                # Index file content locally (embeddings + FAISS) without persisting the file
                content = f.read()
                file_id = embedding_store.add_file(f.filename, content)
                saved_ids.append(file_id)
            return jsonify({'ok': True, 'file_ids': saved_ids})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    @app.post('/chat')
    def chat():
        data = request.get_json(silent=True) or {}
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'ok': False, 'error': 'Empty message'}), 400
        # Retrieve context locally (embeddings + FAISS) and call Chat Completions
        k_ctx = 5
        ctx_chunks = embedding_store.search(message, k=k_ctx)
        context_text = "\n\n".join([f"[Snippet {i+1}]\n{t}" for i, (t, _) in enumerate(ctx_chunks)])
        system_prompt = (
            "You are a helpful assistant. Use the provided snippets to answer. "
            "If the answer isn't in the snippets, say you aren't sure.\n\n"
            + (f"Snippets:\n{context_text}" if context_text else "No snippets available.")
        )
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
        hist.append({'id': f'assistant-{len(hist)+1}', 'role': 'assistant', 'text': answer})
        # keep last 50
        session['chat_history'] = hist[-50:]
        return jsonify({'ok': True, 'messages': session['chat_history']})

    @app.get('/messages')
    def get_messages():
        # Return local history
        return jsonify({'ok': True, 'messages': session.get('chat_history', [])})

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
