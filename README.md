# Document Chat (Embeddings + FAISS)

A simple Python web app to upload documents and chat with them using OpenAI Embeddings for retrieval and a local FAISS vector index. The UI provides two tabs only: Document Chat and Settings.

## Features

- Upload files and index them locally (FAISS)
- PDF ingestion locally (requires `pypdf` or `PyPDF2`)
- Retrieve top matching chunks via embeddings similarity
- Chat with grounded answers using retrieved snippets
- Sources panel under replies with snippet previews and scores
- Drag & drop uploads and copy-to-clipboard for messages
- Configurable via `config.toml` and environment variables
- Optional HTTP proxy and SSL verification toggle

## Configuration

Create a `config.toml` at the project root or export environment variables. Supported keys (environment variable in parentheses):

- `proxy_url` (`PROXY_URL`) — HTTP/HTTPS proxy URL (e.g., `http://localhost:7890`).
- `openai_base_url` (`OPENAI_BASE_URL`) — Override the OpenAI API base URL (e.g., for gateways).
- `openai_api_key` (`OPENAI_API_KEY`) — Your OpenAI API key.
- `disable_ssl` (`DISABLE_SSL`) — Set to `true` to disable SSL verification (trusted setups only).
- `mode` (`MODE`) — Only `chat` is supported (local FAISS retrieval).
- `model_name` (`MODEL_NAME`) — Chat model to use, e.g., `gpt-4o-mini`.
- `embedding_model_name` (`EMBEDDING_MODEL_NAME`) — Embedding model, e.g., `text-embedding-3-small`.
- `secret_key` (`SECRET_KEY`) — Flask session secret (required for production).

An example config is provided in `config.example.toml`.

## Quickstart

1. Create and fill in a config:

   ```sh
   cp config.example.toml config.toml
   # Edit config.toml with your values
   ```

   Or export environment variables instead of using `config.toml`.

2. Create a virtual environment and install deps:

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Run the app:

   ```sh
   # Optional: enable PDF support
   # pip install pypdf
   FLASK_APP=app.py FLASK_ENV=development flask run --port 5000
   ```

4. Open `http://localhost:5000` in your browser.

## Notes

- This app uses OpenAI Embeddings to build a local FAISS index under `data/` and Chat Completions for responses. It does not use OpenAI Vector Stores or Files APIs.
- PDF uploads are parsed locally if `pypdf` (or `PyPDF2`) is installed; otherwise you receive a clear error on upload.
- For production, set a strong `SECRET_KEY` or configure Flask-Session.
- Sessions: when `Flask-Session` is installed, data is stored under `data/flask_sessions`. Without it, the app falls back to signed cookies and automatically trims chat history to stay small.

## Docker

This image reads all settings from `config.toml` inside the container.

Prepare config:

```
cp config.example.toml config.toml
# Edit config.toml with your values (including secret_key)
```

Build:

```
docker build -t document-chat .
```

Run:

```
docker run --rm -p 5000:5000 document-chat
```

Then open http://localhost:5000.

Notes:
- The image bundles the `config.toml` present at build time. To change settings without rebuilding, mount a config file: `-v $(pwd)/config.toml:/app/config.toml:ro`.
- To persist indices and sessions across restarts, mount the data directory: `-v $(pwd)/data:/app/data`.

