# Document Chat (Embeddings + FAISS)

A simple Python web app to upload documents and chat with them using OpenAI Embeddings for retrieval and a local FAISS vector index. Includes a clean, responsive chat UI (bubbles, markdown, copy-to-clipboard, sources panel) and a flexible config supporting custom proxies and base URLs.

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

Create a `config.toml` at the project root or export environment variables. The following keys are supported (environment variable in parentheses):

- `proxy_url` (`PROXY_URL`) — HTTP/HTTPS proxy URL (e.g., `http://localhost:7890`).
- `openai_base_url` (`OPENAI_BASE_URL`) — Override the OpenAI API base URL (e.g., for gateways).
- `openai_api_key` (`OPENAI_API_KEY`) — Your OpenAI API key.
- `disable_ssl` (`DISABLE_SSL`) — Set to `true` to disable SSL verification (use only for trusted setups).
- `mode` (`MODE`) — Only `chat` is supported. Retrieval uses local FAISS + Embeddings.
- `model_name` (`MODEL_NAME`) — The model to use (Assistant or Chat depending on mode), e.g., `gpt-4o-mini`.
- `embedding_model_name` (`EMBEDDING_MODEL_NAME`) — The embedding model to use (e.g., `text-embedding-3-small`).

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

3. Install FAISS (CPU) and run the app:

   ```sh
   pip install -r requirements.txt
   # Optional: enable PDF support
   # pip install pypdf
   FLASK_APP=app.py FLASK_ENV=development flask run --port 5000
   ```

4. Open `http://localhost:5000` in your browser.

## Notes

- This app does not use the Vector Stores or Files APIs. It calls the Embeddings endpoint to build a local FAISS index under `data/` and uses Chat Completions for responses.
- PDF uploads are parsed locally if `pypdf` (or `PyPDF2`) is installed. Without it, uploading PDFs will return a clear error message.
- If you use a proxy or custom base URL (e.g., gateways), set them in the config.
- For production, set a `SECRET_KEY` environment variable for Flask sessions.
