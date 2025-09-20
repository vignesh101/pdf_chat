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
import local_rag


def create_app() -> Flask:
    cfg = load_config()
    app = Flask(__name__)
    app.config['SECRET_KEY'] = cfg.secret_key or os.environ.get('FLASK_SECRET', 'dev-secret-change-me')

    # Build OpenAI client and hold it in app context
    client = build_openai_client(cfg)
    app.config['OPENAI_CLIENT'] = client
    app.config['MODEL_NAME'] = cfg.model_name
    app.config['MODE'] = cfg.mode

    # Ensure vector store and assistant exist at startup (Assistants mode only)
    if cfg.mode == 'assistants':
        ensure_vector_store(client, embedding_model_name=cfg.embedding_model_name)
        assistant_id = ensure_assistant(client, cfg.model_name, embedding_model_name=cfg.embedding_model_name)
        app.config['ASSISTANT_ID'] = assistant_id

    @app.route('/')
    def index():
        # Initialize per-session resources
        if cfg.mode == 'assistants':
            thread_id = session.get('thread_id')
            if not thread_id:
                thread = client.threads.create()
                thread_id = thread.id
                session['thread_id'] = thread_id
        else:
            # Chat mode keeps a local message list
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
                if cfg.mode == 'assistants':
                    local_path = os.path.join(tmp_dir, f.filename)
                    f.save(local_path)
                    file_id = upload_file(client, local_path)
                    saved_ids.append(file_id)
                    try:
                        os.remove(local_path)
                    except OSError:
                        pass
                else:
                    # Chat mode: index file content locally without persisting the file
                    content = f.read()
                    file_id = local_rag.add_file(f.filename, content)
                    saved_ids.append(file_id)
            if saved_ids and cfg.mode == 'assistants':
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
        if cfg.mode == 'assistants':
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
        else:
            # Chat mode: retrieve context locally and call Chat Completions
            k_ctx = 5
            ctx_chunks = local_rag.search(message, k=k_ctx)
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
        if cfg.mode == 'assistants':
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
        else:
            # Chat mode: return local history
            return jsonify({'ok': True, 'messages': session.get('chat_history', [])})

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
