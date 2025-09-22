from __future__ import annotations

import os
from flask import Flask, render_template, request, jsonify, session, Response
from typing import Any, Dict, List

from config_loader import load_config
from openai_client import build_openai_client, build_httpx_client
import embedding_store
import web_embedding_store as web_store
import confluence_embedding_store as conf_store
import octane_embedding_store as oct_store
from confluence_client import ConfluenceAPI
from octane_client import OctaneAPI


def create_app() -> Flask:
    cfg = load_config()
    app = Flask(__name__)
    app.config['SECRET_KEY'] = cfg.secret_key or os.environ.get('FLASK_SECRET', 'dev-secret-change-me')
    # Always reload templates when changed (helps avoid stale UI in dev)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    # Aggressively disable client caching in development to avoid stale HTML/CSS
    disable_cache = (
        bool(app.debug) or
        (os.environ.get('DISABLE_CACHE', '').strip().lower() in ('1', 'true', 'yes', 'on'))
    )
    if disable_cache:
        # Disable static file caching and force template reload behavior
        app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
        try:
            app.jinja_env.auto_reload = True
            # Also drop the in-memory Jinja template cache in dev
            app.jinja_env.cache = {}
        except Exception:
            pass

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

    def _extract_octane_test_ids(text: str) -> List[str]:
        """Return likely Octane test ids from a user message.

        Best-effort patterns: "test id 12345", "test-id: 12345", "test 12345", "TC12345".
        """
        import re
        s = (text or '').strip()
        if not s:
            return []
        out: List[str] = []
        # Pattern 1: explicit "test id" mentions
        for m in re.finditer(r"\btest\s*(?:[-_]?\s*id)?\s*[:#]?\s*(\d{2,})\b", s, flags=re.I):
            try:
                out.append(str(int(m.group(1))))
            except Exception:
                continue
        # Pattern 2: TC12345 or TC-12345
        for m in re.finditer(r"\bTC[-_ ]?(\d{2,})\b", s, flags=re.I):
            try:
                out.append(str(int(m.group(1))))
            except Exception:
                continue
        # Deduplicate, preserve order
        seen = set()
        uniq: List[str] = []
        for tid in out:
            if tid in seen:
                continue
            seen.add(tid)
            uniq.append(tid)
        return uniq

    # Build OpenAI + HTTP clients and hold them in app context (may be None if not configured)
    client = build_openai_client(cfg)
    http_client = build_httpx_client(cfg)
    app.config['OPENAI_CLIENT'] = client
    app.config['HTTP_CLIENT'] = http_client
    # Optional Confluence API client
    conf_api = None
    try:
        if getattr(cfg, 'confluence_base_url', None) and getattr(cfg, 'confluence_access_token', None):
            conf_api = ConfluenceAPI(cfg.confluence_base_url, cfg.confluence_access_token, http_client)
    except Exception:
        conf_api = None
    app.config['CONF_API'] = conf_api
    # Optional Octane API client
    oct_api = None
    try:
        if getattr(cfg, 'octane_base_url', None) and getattr(cfg, 'octane_client_id', None) and getattr(cfg, 'octane_client_secret', None):
            oct_api = OctaneAPI(cfg.octane_base_url, cfg.octane_client_id, cfg.octane_client_secret, http_client)
    except Exception:
        oct_api = None
    app.config['OCT_API'] = oct_api
    app.config['MODEL_NAME'] = cfg.model_name
    # Stamp build/version for visibility in UI
    try:
        import subprocess
        rev = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True)
        app.config['BUILD_REV'] = (rev.stdout or '').strip() or 'unknown'
    except Exception:
        app.config['BUILD_REV'] = 'unknown'
    # Initialize local FAISS embedding store only if client is available
    try:
        if client is not None:
            embedding_store.init(client, embedding_model_name=cfg.embedding_model_name)
            # Initialize separate FAISS store for Web chat
            web_store.init(client, embedding_model_name=cfg.embedding_model_name)
            # Initialize separate FAISS store for Confluence chat
            conf_store.init(client, embedding_model_name=cfg.embedding_model_name)
            # Initialize separate FAISS store for Octane chat
            oct_store.init(client, embedding_model_name=cfg.embedding_model_name)
    except Exception:
        # Do not block app startup; search/upload will surface errors
        pass

    @app.route('/')
    def index():
        # Single mode: Chat with local FAISS RAG
        session.setdefault('chat_history', [])
        session.setdefault('web_chat_history', [])
        session.setdefault('conf_chat_history', [])
        session.setdefault('octane_chat_history', [])
        thread_id = None
        return render_template(
            'index.html',
            model_name=cfg.model_name,
            proxy_url=cfg.proxy_url,
            base_url=cfg.openai_base_url,
            disable_ssl=cfg.disable_ssl,
            client_ready=bool(app.config.get('OPENAI_CLIENT')),
            octane_ready=bool(app.config.get('OCT_API')),
            build_rev=app.config.get('BUILD_REV', 'unknown'),
            thread_id=thread_id,
        )

    # In development, add no-cache headers for HTML responses to prevent
    # browsers/reverse proxies from serving stale index pages.
    if disable_cache:
        @app.after_request
        def _no_cache(resp):  # type: ignore
            try:
                mt = (resp.mimetype or '').lower()
                if mt.startswith('text/html'):
                    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
                    resp.headers['Pragma'] = 'no-cache'
                    resp.headers['Expires'] = '0'
            except Exception:
                pass
            return resp

    # --- Voice: sample upload, status, TTS (OpenAI) ---
    VOICE_DIR = os.path.join(os.getcwd(), 'data', 'voice')
    try:
        os.makedirs(VOICE_DIR, exist_ok=True)
    except Exception:
        pass

    def _voice_sample_path() -> str | None:
        b = session.get('voice_sample_basename')
        if not b:
            return None
        return os.path.join(VOICE_DIR, b)

    @app.get('/voice/status')
    def voice_status():
        p = _voice_sample_path()
        exists = bool(p and os.path.isfile(p))
        return jsonify({
            'ok': True,
            'sample_exists': exists,
            'sample_name': os.path.basename(p) if exists else None,
            'engine': 'local',
        })

    @app.post('/voice/upload_sample')
    def voice_upload_sample():
        f = request.files.get('file')
        if not f:
            return jsonify({'ok': False, 'error': 'No file uploaded'}), 400
        try:
            name = f.filename or 'sample'
            base, ext = os.path.splitext(name)
            ext = ext.lower() if ext else '.mp3'
            safe_ext = ext if ext in ('.mp3', '.wav', '.m4a', '.aac', '.ogg', '.webm') else '.mp3'
            target = os.path.join(VOICE_DIR, f'sample{safe_ext}')
            with open(target, 'wb') as out:
                out.write(f.read())
            session['voice_sample_basename'] = os.path.basename(target)
            return jsonify({'ok': True, 'sample_name': os.path.basename(target)})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    @app.post('/voice/clear')
    def voice_clear():
        try:
            p = _voice_sample_path()
            if p and os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
            session.pop('voice_sample_basename', None)
            return jsonify({'ok': True})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    @app.post('/voice/tts')
    def voice_tts():
        """Synthesize speech locally with optional zero-shot voice cloning from the uploaded sample.
        Preferred engine: Coqui TTS (if installed and model configured). Fallback: espeak/espeak-ng, then pyttsx3.
        Always returns WAV audio.
        """
        import io
        import shutil
        import subprocess
        data = request.get_json(silent=True) or {}
        text = (data.get('text') or '').strip()
        if not text:
            return jsonify({'ok': False, 'error': 'Text is required'}), 400
        requested_voice = (data.get('voice_name') or '').strip()
        use_sample_only = requested_voice.lower() == 'sample'
        sample_wav = _voice_sample_path()
        out_path = os.path.join(VOICE_DIR, 'tts.wav')

        # 1) Coqui TTS voice cloning if available
        coqui_model = getattr(cfg, 'coqui_tts_model', None)
        coqui_device = (getattr(cfg, 'coqui_tts_device', None) or '').strip() or None
        coqui_lang = (getattr(cfg, 'coqui_tts_language', None) or '').strip() or None
        if coqui_model:
            try:
                from TTS.api import TTS as COQUI_TTS  # type: ignore
                # Initialize once and cache
                coqui_inst = app.config.get('COQUI_TTS_INST')
                if not coqui_inst or app.config.get('COQUI_TTS_MODEL_NAME') != coqui_model:
                    coqui_inst = COQUI_TTS(model_name=coqui_model)
                    # Force device if specified
                    if coqui_device:
                        try:
                            coqui_inst.to(coqui_device)
                        except Exception:
                            pass
                    app.config['COQUI_TTS_INST'] = coqui_inst
                    app.config['COQUI_TTS_MODEL_NAME'] = coqui_model
                # Prefer speaker_wav if sample uploaded and model supports it
                kwargs = {}
                if coqui_lang:
                    kwargs['language'] = coqui_lang
                if sample_wav and os.path.isfile(sample_wav):
                    kwargs['speaker_wav'] = sample_wav
                # Some models accept 'speaker' name; requested_voice may map there
                if requested_voice and not use_sample_only:
                    kwargs['speaker'] = requested_voice
                coqui_inst.tts_to_file(text=text, file_path=out_path, **kwargs)
                with open(out_path, 'rb') as f:
                    return Response(f.read(), mimetype='audio/wav')
            except Exception:
                # fall through to other engines
                pass

        # 2) espeak/espeak-ng CLI
        espeak_bin = shutil.which('espeak') or shutil.which('espeak-ng')
        if espeak_bin:
            try:
                cmd = [espeak_bin, '-w', out_path, '-s', '170']
                if requested_voice and not use_sample_only:
                    cmd += ['-v', requested_voice]
                cmd += [text]
                subprocess.run(cmd, check=True)
                with open(out_path, 'rb') as f:
                    return Response(f.read(), mimetype='audio/wav')
            except Exception:
                pass

        # 3) pyttsx3 fallback
        try:
            import pyttsx3  # type: ignore
            eng = pyttsx3.init()
            try:
                if requested_voice:
                    for v in eng.getProperty('voices') or []:
                        vid = getattr(v, 'id', '') or ''
                        name = getattr(v, 'name', '') or ''
                        if requested_voice.lower() in (vid.lower() + ' ' + name.lower()):
                            eng.setProperty('voice', v.id)
                            break
            except Exception:
                pass
            eng.save_to_file(text, out_path)
            eng.runAndWait()
            with open(out_path, 'rb') as f:
                return Response(f.read(), mimetype='audio/wav')
        except Exception:
            pass

        return jsonify({'ok': False, 'error': 'Local TTS unavailable. Install Coqui TTS (TTS), or espeak/espeak-ng, or pyttsx3.'}), 500

    # --- Octane: auth helper to acquire cookies ---
    @app.post('/octane/auth')
    def octane_auth():
        oct_api = app.config.get('OCT_API')
        if oct_api is None:
            return jsonify({'ok': False, 'error': 'Octane is not configured. Set octane_base_url, octane_client_id and octane_client_secret in config.'}), 400
        try:
            res = oct_api.login()
            # Track cookie names in session for visibility
            session['octane_cookie_names'] = res.cookies
            return jsonify({'ok': bool(res.ok), 'cookies': res.cookies, 'note': res.note})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    # --- Octane Chat: fetch items and chat ---
    @app.post('/octanechat')
    def octane_chat():
        data = request.get_json(silent=True) or {}
        message = (data.get('message') or '').strip()
        if not message:
            return jsonify({'ok': False, 'error': 'Empty message'}), 400
        oct_api = app.config.get('OCT_API')
        if oct_api is None:
            return jsonify({'ok': False, 'error': 'Octane is not configured. Set octane config and restart.'}), 400
        try:
            max_results = int(data.get('max_results') or 5)
        except Exception:
            max_results = 5
        try:
            k_ctx = int(data.get('k_ctx') or 5)
        except Exception:
            k_ctx = 5
        project_id = (data.get('project_id') or '').strip()
        workspace_id = (data.get('workspace_id') or '').strip()
        max_results = max(1, min(10, max_results))
        k_ctx = max(1, min(10, k_ctx))

        # Ensure cookies/auth and try to fetch items from Octane and index
        try:
            try:
                oct_api.login()
            except Exception:
                pass
            items = oct_api.fetch_sample_items(project_id, workspace_id, limit=max_results)
            for it in items:
                oct_store.add_item(it.get('key') or '', it.get('title'), it.get('text') or '')
        except Exception:
            # best-effort
            items = []

        # If the user referenced a test id, fetch its script and index it
        try:
            tids = _extract_octane_test_ids(message)
            if tids:
                seen_keys = set(session.get('octane_seen_test_keys', []) or [])
                for tid in tids:
                    res = oct_api.fetch_test_script(project_id, workspace_id, tid)
                    if res and (res.get('text') or '').strip():
                        key = res.get('key') or f"octane:test:{tid}"
                        if key not in seen_keys:
                            oct_store.add_item(key, res.get('title'), res.get('text') or '')
                            seen_keys.add(key)
                session['octane_seen_test_keys'] = list(seen_keys)
        except Exception:
            # non-fatal
            pass

        try:
            ctx_chunks_meta = oct_store.search_with_meta(message, k=k_ctx)
        except Exception:
            ctx_chunks_meta = []
        context_text = "\n\n".join([
            f"[Octane {i+1}] {it.get('file_name') or it.get('file_id')}\n{it.get('text', '')}"
            for i, it in enumerate(ctx_chunks_meta)
        ])
        system_prompt = (
            "You are a helpful assistant with access to ALM Octane item excerpts. "
            "Use the provided Octane content to answer accurately. If the answer isn't present, say you aren't sure.\n\n"
            + (f"Octane Content:\n{context_text}" if context_text else "No Octane content available.")
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

        # Update history
        hist = session.get('octane_chat_history', [])
        hist.append({'id': f'user-{len(hist)+1}', 'role': 'user', 'text': message})
        hist.append({
            'id': f'assistant-{len(hist)+1}',
            'role': 'assistant',
            'text': answer,
            'sources': [
                {
                    'file_id': it.get('file_id'),
                    'file_name': it.get('file_name'),
                    'score': it.get('score'),
                    'text': (it.get('text') or '')[:app.config.get('MAX_SOURCE_PREVIEW', 300)],
                }
                for it in ctx_chunks_meta
            ],
        })
        session['octane_chat_history'] = hist
        return jsonify({'ok': True, 'messages': session.get('octane_chat_history', [])})

    @app.post('/octanechat_stream')
    def octane_chat_stream():
        from flask import Response, stream_with_context
        data = request.get_json(silent=True) or {}
        message = (data.get('message') or '').strip()
        if not message:
            return jsonify({'ok': False, 'error': 'Empty message'}), 400
        oct_api = app.config.get('OCT_API')
        if oct_api is None:
            return jsonify({'ok': False, 'error': 'Octane is not configured. Set octane config and restart.'}), 400
        try:
            max_results = int(data.get('max_results') or 5)
        except Exception:
            max_results = 5
        try:
            k_ctx = int(data.get('k_ctx') or 5)
        except Exception:
            k_ctx = 5
        project_id = (data.get('project_id') or '').strip()
        workspace_id = (data.get('workspace_id') or '').strip()
        max_results = max(1, min(10, max_results))
        k_ctx = max(1, min(10, k_ctx))

        try:
            try:
                oct_api.login()
            except Exception:
                pass
            items = oct_api.fetch_sample_items(project_id, workspace_id, limit=max_results)
            for it in items:
                oct_store.add_item(it.get('key') or '', it.get('title'), it.get('text') or '')
        except Exception:
            pass

        # If the user referenced a test id, fetch its script and index it before retrieval
        try:
            tids = _extract_octane_test_ids(message)
            if tids:
                seen_keys = set(session.get('octane_seen_test_keys', []) or [])
                for tid in tids:
                    res = oct_api.fetch_test_script(project_id, workspace_id, tid)
                    if res and (res.get('text') or '').strip():
                        key = res.get('key') or f"octane:test:{tid}"
                        if key not in seen_keys:
                            oct_store.add_item(key, res.get('title'), res.get('text') or '')
                            seen_keys.add(key)
                session['octane_seen_test_keys'] = list(seen_keys)
        except Exception:
            pass

        try:
            ctx_chunks_meta = oct_store.search_with_meta(message, k=k_ctx)
        except Exception:
            ctx_chunks_meta = []
        context_text = "\n\n".join([
            f"[Octane {i+1}] {it.get('file_name') or it.get('file_id')}\n{it.get('text', '')}"
            for i, it in enumerate(ctx_chunks_meta)
        ])
        system_prompt = (
            "You are a helpful assistant with access to ALM Octane item excerpts. "
            "Use the provided Octane content to answer accurately. If the answer isn't present, say you aren't sure.\n\n"
            + (f"Octane Content:\n{context_text}" if context_text else "No Octane content available.")
        )
        client = app.config.get('OPENAI_CLIENT')
        if client is None:
            return jsonify({'ok': False, 'error': 'OpenAI client not configured. Set OPENAI_API_KEY or update config.toml.'}), 500

        try:
            session['pending_octane_assistant'] = {
                'sources': [
                    {
                        'file_id': it.get('file_id'),
                        'file_name': it.get('file_name'),
                        'score': it.get('score'),
                        'text': (it.get('text') or '')[:app.config.get('MAX_SOURCE_PREVIEW', 300)],
                    }
                    for it in ctx_chunks_meta
                ],
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

    @app.post('/octane/commit')
    def octane_commit():
        data = request.get_json(silent=True) or {}
        question = (data.get('question') or '').strip()
        answer = (data.get('answer') or '').strip()
        if not question or not answer:
            return jsonify({'ok': False, 'error': 'question and answer required'}), 400
        try:
            hist = session.get('octane_chat_history', [])
            pending = session.get('pending_octane_assistant') or {}
            sources = pending.get('sources') or []
            hist.append({'id': f'user-{len(hist)+1}', 'role': 'user', 'text': question})
            hist.append({'id': f'assistant-{len(hist)+1}', 'role': 'assistant', 'text': answer, 'sources': sources})
            session['octane_chat_history'] = hist
            session.pop('pending_octane_assistant', None)
            return jsonify({'ok': True, 'messages': session.get('octane_chat_history', [])})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    @app.get('/octane/messages')
    def octane_get_messages():
        return jsonify({'ok': True, 'messages': session.get('octane_chat_history', [])})

    @app.post('/octane/clear')
    def octane_clear_messages():
        session['octane_chat_history'] = []
        return jsonify({'ok': True, 'messages': []})

    @app.post('/octane/index/clear_all')
    def octane_index_clear_all():
        try:
            st = oct_store.clear_all()
            return jsonify({'ok': True, 'index': st})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

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

    # Track last DuckDuckGo search status for diagnostics
    _DDG_STATUS: Dict[str, Any] = {"ok": None, "note": None}

    def _ddg_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """DuckDuckGo search using duckduckgo_search (ddgs). Honors proxy and SSL settings.

        Returns list of {url,title} and sets _DDG_STATUS for diagnostics.
        """
        results: List[Dict[str, str]] = []
        # Reset status for this call
        try:
            _DDG_STATUS["ok"] = None
            _DDG_STATUS["note"] = None
        except Exception:
            pass
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except Exception:
            try:
                _DDG_STATUS["ok"] = False
                _DDG_STATUS["note"] = "duckduckgo_search not installed or failed to import"
            except Exception:
                pass
            return results
        # Prepare kwargs with proxy and SSL behavior if supported, accounting for ddgs version differences
        base_kwargs: Dict[str, Any] = {
            'timeout': 30.0,
            'headers': {'User-Agent': 'document-chat/1.0 (+https://localhost)'},
        }
        proxy_val = getattr(cfg, 'proxy_url', None) or None

        # Build a list of argument variants to try, to support ddgs versions
        # that expect either 'proxies' (modern) or 'proxy' (legacy), and
        # those that do or do not accept 'verify'.
        attempts: List[Dict[str, Any]] = []

        def add_attempt(proxy_key: str | None, include_verify: bool) -> None:
            kw = dict(base_kwargs)
            if proxy_key and proxy_val:
                kw[proxy_key] = proxy_val
            if include_verify and getattr(cfg, 'disable_ssl', False):
                kw['verify'] = False
            attempts.append(kw)

        # Prefer modern API names first
        add_attempt('proxies', True)
        add_attempt('proxies', False)
        # Fall back to legacy param name
        add_attempt('proxy', True)
        add_attempt('proxy', False)
        # As a last resort, try without any proxy configured
        add_attempt(None, True)
        add_attempt(None, False)

        last_err: Exception | None = None
        for kw in attempts:
            try:
                with DDGS(**kw) as ddgs:
                    for item in ddgs.text(
                        query,
                        region="wt-wt",
                        safesearch="moderate",
                        timelimit=None,
                        max_results=max_results,
                    ):
                        try:
                            url = (item.get('href') or '').strip()
                            title = (item.get('title') or '').strip()
                            if url:
                                results.append({'url': url, 'title': title})
                        except Exception:
                            continue
                # If we succeeded with this kw set, stop trying further variants
                last_err = None
                break
            except TypeError as e:
                # Likely due to unexpected keyword (e.g., 'verify', 'proxies', or 'proxy')
                last_err = e
                continue
            except Exception as e:
                last_err = e
                break

        if last_err is not None and not results:
            try:
                _DDG_STATUS["ok"] = False
                _DDG_STATUS["note"] = f"search failed: {str(last_err)[:140]}"
            except Exception:
                pass
        # Mark status if not already set
        try:
            if _DDG_STATUS.get("ok") is None:
                _DDG_STATUS["ok"] = True
                _DDG_STATUS["note"] = "0 results" if not results else None
        except Exception:
            pass
        return results

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
        if not message:
            return jsonify({'ok': False, 'error': 'Empty message'}), 400

        # Options
        try:
            max_results = int(data.get('max_results') or 5)
        except Exception:
            max_results = 5
        try:
            k_ctx = int(data.get('k_ctx') or 5)
        except Exception:
            k_ctx = 5
        # Toggle for using web search (DuckDuckGo)
        search_enabled = True
        try:
            se = data.get('search_enabled')
            if isinstance(se, bool):
                search_enabled = se
            elif isinstance(se, (int,)):
                search_enabled = bool(se)
            elif isinstance(se, str):
                search_enabled = se.strip().lower() in ('1', 'true', 'yes', 'on')
        except Exception:
            pass
        # Clamp
        max_results = max(1, min(10, max_results))
        k_ctx = max(1, min(10, k_ctx))

        # Build candidate URLs via DDG + any URLs present in the message
        ddg_hits = _ddg_search(message, max_results=max_results) if search_enabled else []
        ddg_urls = [h.get('url') for h in ddg_hits if isinstance(h.get('url'), str)]
        typed_urls = _extract_urls_from_text(message)
        urls = list(dict.fromkeys((typed_urls + ddg_urls)))[:max_results]

        # Record search context for UI visibility
        search_info = {
            'engine': 'DuckDuckGo' if search_enabled else 'Manual',
            'query': message,
            'results': ddg_hits,
            'used_urls': urls,
            'status': dict(_DDG_STATUS) if search_enabled else {'ok': True, 'note': None},
        }

        # Fetch pages and index into web-only FAISS store
        web_pages = _fetch_urls_text(urls)
        try:
            for wp in web_pages:
                web_store.add_page(wp.get('url') or '', wp.get('title'), wp.get('text') or '')
        except Exception:
            pass

        # Retrieve best context from web index
        try:
            ctx_chunks_meta = web_store.search_with_meta(message, k=k_ctx)
        except Exception:
            ctx_chunks_meta = []
        context_text = "\n\n".join([
            f"[Web {i+1}] {it.get('file_name') or it.get('file_id')}\n{it.get('text', '')}"
            for i, it in enumerate(ctx_chunks_meta)
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
            'sources': [
                {
                    'file_id': it.get('file_id'),
                    'file_name': it.get('file_name'),
                    'score': it.get('score'),
                    'text': (it.get('text') or '')[:app.config.get('MAX_SOURCE_PREVIEW', 300)],
                }
                for it in ctx_chunks_meta
            ],
            'search': search_info,
        })
        session['web_chat_history'] = hist
        return jsonify({'ok': True, 'messages': session.get('web_chat_history', [])})

    # --- Confluence Chat: search your Confluence and chat ---
    @app.post('/confchat')
    def confchat():
        data = request.get_json(silent=True) or {}
        message = (data.get('message') or '').strip()
        if not message:
            return jsonify({'ok': False, 'error': 'Empty message'}), 400

        conf_api = app.config.get('CONF_API')
        if conf_api is None:
            return jsonify({'ok': False, 'error': 'Confluence is not configured. Set confluence_base_url and confluence_access_token in config.'}), 400

        try:
            max_results = int(data.get('max_results') or 5)
        except Exception:
            max_results = 5
        try:
            k_ctx = int(data.get('k_ctx') or 5)
        except Exception:
            k_ctx = 5
        spaces_raw = (data.get('spaces') or '').strip()
        spaces = [s.strip() for s in spaces_raw.split(',') if s and s.strip()] if spaces_raw else []
        max_results = max(1, min(10, max_results))
        k_ctx = max(1, min(10, k_ctx))

        # Search Confluence and fetch page contents
        hits = []
        try:
            hits = conf_api.search_pages(message, spaces=spaces, limit=max_results)
        except Exception:
            hits = []

        try:
            for h in hits:
                pid = str(h.get('id') or '')
                if not pid:
                    continue
                text, title = conf_api.get_page_content_text(pid)
                url = h.get('url') or ''
                conf_store.add_page(url or (title or pid), title, text or '')
        except Exception:
            pass

        # Retrieve best context from Confluence index
        try:
            ctx_chunks_meta = conf_store.search_with_meta(message, k=k_ctx)
        except Exception:
            ctx_chunks_meta = []
        context_text = "\n\n".join([
            f"[Confluence {i+1}] {it.get('file_name') or it.get('file_id')}\n{it.get('text', '')}"
            for i, it in enumerate(ctx_chunks_meta)
        ])
        system_prompt = (
            "You are a helpful assistant with access to Confluence page excerpts. "
            "Use the provided Confluence content to answer accurately. If the answer isn't present, say you aren't sure.\n\n"
            + (f"Confluence Content:\n{context_text}" if context_text else "No Confluence content available.")
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

        # Update confluence history
        hist = session.get('conf_chat_history', [])
        hist.append({'id': f'user-{len(hist)+1}', 'role': 'user', 'text': message})
        hist.append({
            'id': f'assistant-{len(hist)+1}',
            'role': 'assistant',
            'text': answer,
            'sources': [
                {
                    'file_id': it.get('file_id'),
                    'file_name': it.get('file_name'),
                    'score': it.get('score'),
                    'text': (it.get('text') or '')[:app.config.get('MAX_SOURCE_PREVIEW', 300)],
                }
                for it in ctx_chunks_meta
            ],
        })
        session['conf_chat_history'] = hist
        return jsonify({'ok': True, 'messages': session.get('conf_chat_history', [])})

    @app.post('/confchat_stream')
    def confchat_stream():
        from flask import Response, stream_with_context
        data = request.get_json(silent=True) or {}
        message = (data.get('message') or '').strip()
        if not message:
            return jsonify({'ok': False, 'error': 'Empty message'}), 400

        conf_api = app.config.get('CONF_API')
        if conf_api is None:
            return jsonify({'ok': False, 'error': 'Confluence is not configured. Set confluence_base_url and confluence_access_token in config.'}), 400

        try:
            max_results = int(data.get('max_results') or 5)
        except Exception:
            max_results = 5
        try:
            k_ctx = int(data.get('k_ctx') or 5)
        except Exception:
            k_ctx = 5
        spaces_raw = (data.get('spaces') or '').strip()
        spaces = [s.strip() for s in spaces_raw.split(',') if s and s.strip()] if spaces_raw else []
        max_results = max(1, min(10, max_results))
        k_ctx = max(1, min(10, k_ctx))

        hits = []
        try:
            hits = conf_api.search_pages(message, spaces=spaces, limit=max_results)
        except Exception:
            hits = []
        try:
            for h in hits:
                pid = str(h.get('id') or '')
                if not pid:
                    continue
                text, title = conf_api.get_page_content_text(pid)
                url = h.get('url') or ''
                conf_store.add_page(url or (title or pid), title, text or '')
        except Exception:
            pass

        try:
            ctx_chunks_meta = conf_store.search_with_meta(message, k=k_ctx)
        except Exception:
            ctx_chunks_meta = []
        context_text = "\n\n".join([
            f"[Confluence {i+1}] {it.get('file_name') or it.get('file_id')}\n{it.get('text', '')}"
            for i, it in enumerate(ctx_chunks_meta)
        ])
        system_prompt = (
            "You are a helpful assistant with access to Confluence page excerpts. "
            "Use the provided Confluence content to answer accurately. If the answer isn't present, say you aren't sure.\n\n"
            + (f"Confluence Content:\n{context_text}" if context_text else "No Confluence content available.")
        )
        client = app.config.get('OPENAI_CLIENT')
        if client is None:
            return jsonify({'ok': False, 'error': 'OpenAI client not configured. Set OPENAI_API_KEY or update config.toml.'}), 500

        try:
            session['pending_conf_assistant'] = {
                'sources': [
                    {
                        'file_id': it.get('file_id'),
                        'file_name': it.get('file_name'),
                        'score': it.get('score'),
                        'text': (it.get('text') or '')[:app.config.get('MAX_SOURCE_PREVIEW', 300)],
                    }
                    for it in ctx_chunks_meta
                ],
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

    @app.post('/conf/commit')
    def conf_commit():
        data = request.get_json(silent=True) or {}
        question = (data.get('question') or '').strip()
        answer = (data.get('answer') or '').strip()
        if not question or not answer:
            return jsonify({'ok': False, 'error': 'question and answer required'}), 400
        try:
            hist = session.get('conf_chat_history', [])
            pending = session.get('pending_conf_assistant') or {}
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
            session['conf_chat_history'] = hist
            session.pop('pending_conf_assistant', None)
            return jsonify({'ok': True, 'messages': session.get('conf_chat_history', [])})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    @app.get('/conf/messages')
    def conf_get_messages():
        return jsonify({'ok': True, 'messages': session.get('conf_chat_history', [])})

    @app.post('/conf/clear')
    def conf_clear_messages():
        session['conf_chat_history'] = []
        return jsonify({'ok': True, 'messages': []})

    @app.post('/conf/index/clear_all')
    def conf_index_clear_all():
        try:
            st = conf_store.clear_all()
            return jsonify({'ok': True, 'index': st})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    @app.post('/webchat_stream')
    def webchat_stream():
        from flask import Response, stream_with_context
        data = request.get_json(silent=True) or {}
        message = (data.get('message') or '').strip()
        if not message:
            return jsonify({'ok': False, 'error': 'Empty message'}), 400

        # Options
        try:
            max_results = int(data.get('max_results') or 5)
        except Exception:
            max_results = 5
        try:
            k_ctx = int(data.get('k_ctx') or 5)
        except Exception:
            k_ctx = 5
        # Toggle for using web search (DuckDuckGo)
        search_enabled = True
        try:
            se = data.get('search_enabled')
            if isinstance(se, bool):
                search_enabled = se
            elif isinstance(se, (int,)):
                search_enabled = bool(se)
            elif isinstance(se, str):
                search_enabled = se.strip().lower() in ('1', 'true', 'yes', 'on')
        except Exception:
            pass
        max_results = max(1, min(10, max_results))
        k_ctx = max(1, min(10, k_ctx))

        ddg_hits = _ddg_search(message, max_results=max_results) if search_enabled else []
        ddg_urls = [h.get('url') for h in ddg_hits if isinstance(h.get('url'), str)]
        typed_urls = _extract_urls_from_text(message)
        urls = list(dict.fromkeys((typed_urls + ddg_urls)))[:max_results]

        # Record search context for UI visibility
        search_info = {
            'engine': 'DuckDuckGo' if search_enabled else 'Manual',
            'query': message,
            'results': ddg_hits,
            'used_urls': urls,
            'status': dict(_DDG_STATUS) if search_enabled else {'ok': True, 'note': None},
        }

        web_pages = _fetch_urls_text(urls)
        try:
            for wp in web_pages:
                web_store.add_page(wp.get('url') or '', wp.get('title'), wp.get('text') or '')
        except Exception:
            pass

        try:
            ctx_chunks_meta = web_store.search_with_meta(message, k=k_ctx)
        except Exception:
            ctx_chunks_meta = []
        context_text = "\n\n".join([
            f"[Web {i+1}] {it.get('file_name') or it.get('file_id')}\n{it.get('text', '')}"
            for i, it in enumerate(ctx_chunks_meta)
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
                'sources': [
                    {
                        'file_id': it.get('file_id'),
                        'file_name': it.get('file_name'),
                        'score': it.get('score'),
                        'text': (it.get('text') or '')[:app.config.get('MAX_SOURCE_PREVIEW', 300)],
                    }
                    for it in ctx_chunks_meta
                ],
                'search': search_info,
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

    @app.post('/web/search_info')
    def web_search_info():
        data = request.get_json(silent=True) or {}
        message = (data.get('message') or '').strip()
        if not message:
            return jsonify({'ok': False, 'error': 'Empty message'}), 400
        try:
            max_results = int(data.get('max_results') or 5)
        except Exception:
            max_results = 5
        max_results = max(1, min(10, max_results))
        # Toggle for using web search (DuckDuckGo)
        search_enabled = True
        try:
            se = data.get('search_enabled')
            if isinstance(se, bool):
                search_enabled = se
            elif isinstance(se, (int,)):
                search_enabled = bool(se)
            elif isinstance(se, str):
                search_enabled = se.strip().lower() in ('1', 'true', 'yes', 'on')
        except Exception:
            pass
        try:
            ddg_hits = _ddg_search(message, max_results=max_results) if search_enabled else []
        except Exception:
            ddg_hits = []
        ddg_urls = [h.get('url') for h in ddg_hits if isinstance(h.get('url'), str)]
        typed_urls = _extract_urls_from_text(message)
        urls = list(dict.fromkeys((typed_urls + ddg_urls)))[:max_results]
        search_info = {
            'engine': 'DuckDuckGo' if search_enabled else 'Manual',
            'query': message,
            'results': ddg_hits,
            'used_urls': urls,
            'status': dict(_DDG_STATUS) if search_enabled else {'ok': True, 'note': None},
        }
        return jsonify({'ok': True, 'search': search_info})

    @app.post('/web/index/clear_all')
    def web_index_clear_all():
        try:
            st = web_store.clear_all()
            return jsonify({'ok': True, 'index': st})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

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
            search_info = pending.get('search') or None
            hist.append({'id': f'user-{len(hist)+1}', 'role': 'user', 'text': question})
            msg = {'id': f'assistant-{len(hist)+1}', 'role': 'assistant', 'text': answer, 'sources': sources}
            if search_info is not None:
                msg['search'] = search_info
            hist.append(msg)
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
