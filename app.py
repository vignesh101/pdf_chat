from __future__ import annotations

import os
from flask import Flask, render_template, request, jsonify, session

from config_loader import load_config
from openai_client import build_openai_client
from assistants import (
    ensure_assistant,
    ensure_vector_store,
    upload_file,
    attach_files_to_vector_store,
    run_chat_turn,
)


def create_app() -> Flask:
    cfg = load_config()
    app = Flask(__name__)
    app.config['SECRET_KEY'] = cfg.secret_key or os.environ.get('FLASK_SECRET', 'dev-secret-change-me')

    # Build OpenAI client and hold it in app context
    client = build_openai_client(cfg)
    app.config['OPENAI_CLIENT'] = client
    app.config['MODEL_NAME'] = cfg.model_name

    # Ensure vector store and assistant exist at startup
    ensure_vector_store(client, embedding_model_name=cfg.embedding_model_name)
    assistant_id = ensure_assistant(client, cfg.model_name, embedding_model_name=cfg.embedding_model_name)
    app.config['ASSISTANT_ID'] = assistant_id

    @app.route('/')
    def index():
        # Lazily create per-session thread
        thread_id = session.get('thread_id')
        if not thread_id:
            thread = client.threads.create()
            thread_id = thread.id
            session['thread_id'] = thread_id
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
                local_path = os.path.join(tmp_dir, f.filename)
                f.save(local_path)
                file_id = upload_file(client, local_path)
                saved_ids.append(file_id)
                try:
                    os.remove(local_path)
                except OSError:
                    pass
            if saved_ids:
                attach_files_to_vector_store(client, saved_ids)
            return jsonify({'ok': True, 'file_ids': saved_ids})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    @app.post('/chat')
    def chat():
        data = request.get_json(silent=True) or {}
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'ok': False, 'error': 'Empty message'}), 400
        thread_id = session.get('thread_id')
        if not thread_id:
            thread = client.threads.create()
            thread_id = thread.id
            session['thread_id'] = thread_id

        assistant_id = app.config['ASSISTANT_ID']
        try:
            messages = run_chat_turn(client, assistant_id, thread_id, message)
            return jsonify({'ok': True, 'messages': messages})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    @app.get('/messages')
    def get_messages():
        thread_id = session.get('thread_id')
        if not thread_id:
            return jsonify({'ok': True, 'messages': []})
        try:
            # Quick fetch of latest messages after a turn is complete
            msgs = client.threads.messages.list(thread_id=thread_id, order='desc', limit=50)
            formatted = []
            for m in reversed(list(msgs)):
                parts = []
                for c in m.content:
                    if c.type == 'text':
                        parts.append(c.text.value)
                formatted.append({'id': m.id, 'role': m.role, 'text': '\n\n'.join(parts)})
            return jsonify({'ok': True, 'messages': formatted})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
