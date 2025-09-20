# Document Chat (Embeddings + FAISS)

A simple Python web app to upload documents and chat with them using OpenAI Embeddings for retrieval and a local FAISS vector index. Includes a clean, responsive chat UI (bubbles, markdown, copy-to-clipboard, sources panel) and a flexible config supporting custom proxies and base URLs.

## Features

- Upload files and index them locally (FAISS)
- PDF ingestion locally (requires `pypdf` or `PyPDF2`)
- Retrieve top matching chunks via embeddings similarity
- Chat with grounded answers using retrieved snippets
- Web Chat: ask questions answered from the internet via DuckDuckGo search (RAG over fetched pages)
- Confluence Chat: search Confluence pages (via REST API), cache content locally and answer grounded in those excerpts
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
 - `confluence_base_url` (`CONFLUENCE_BASE_URL`) — Base URL of your Confluence site (e.g., `https://your-domain.atlassian.net`). The app tries both `/wiki/rest/api` and `/rest/api` paths.
 - `confluence_access_token` (`CONFLUENCE_ACCESS_TOKEN`) — Token used to authenticate with Confluence. If the value contains a colon (e.g., `email@example.com:API_TOKEN`) it is sent as Basic auth. Otherwise it is sent as a Bearer token (for DC PATs).

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

## Web Chat (Internet RAG)

This app includes a separate “Web” chat mode that uses DuckDuckGo Search to discover relevant web pages, fetches them over HTTP, extracts readable text, stores it in a dedicated FAISS index, and then answers your question grounded in those excerpts.

- No API key needed for search: it uses the `duckduckgo_search` library (DDGS).
- Proxies and SSL verification settings are honored (see `proxy_url` and `disable_ssl`).
- Retrieved web content is stored under `data/web` as a separate FAISS index.

How it works at a glance:
- Perform a DDG text search for your query and combine with any URLs you typed.
- Fetch up to 5 pages, strip HTML to text, chunk, embed, and add to the web index.
- Retrieve the top matches from the web index and pass them to Chat Completions as context.
- The reply shows a Sources panel with page titles/URLs and snippet previews.

Using it:
- In the UI, click the “Web” tab above the input box and ask your question.
- For streaming responses the app uses `/webchat_stream` with a follow-up `/webchat/commit` to save history; non-streaming uses `/webchat`.
- Clear just the Web chat history via “New Chat” or the Clear button; the web FAISS store persists in `data/web` between sessions.
- Adjust retrieval:
  - “Web pages” controls how many pages to fetch/search (1–10).
  - “Context k” controls how many chunks are retrieved as context (1–10).
- Manage cache:
  - Use “Clear Web Cache” (in the Status panel) to delete the web FAISS index and metadata under `data/web`.

Notes and limits:
- Some sites may block automated fetching; results may be partial or missing.
- Respect target sites’ terms of use and robots policies.
- Network errors are handled gracefully (the answer may indicate missing web content).

## Confluence Chat

This app includes a “Confluence” chat mode that searches your Confluence using the REST API, fetches matching pages, stores their text in a dedicated FAISS index, and then answers questions grounded in those excerpts.

- Configure `confluence_base_url` and `confluence_access_token` in `config.toml` or environment.
- Proxies and SSL verification settings are honored (see `proxy_url` and `disable_ssl`).
- Retrieved Confluence content is stored under `data/confluence` as a separate FAISS index.

How it works:
- Build a Confluence CQL query for your prompt and optional space filters.
- Fetch up to N pages’ rendered HTML and convert to plain text.
- Chunk, embed, and add to the Confluence FAISS index.
- Retrieve the top matches and pass as context to Chat Completions.

Using it:
- In the UI, click the “Confluence” tab. Optionally limit to specific spaces (comma-separated), adjust “Pages” and “Context k”, then ask your question.
- Clear just the Confluence cache via the “Clear Confluence Cache” button in the Status panel.

Notes and limits:
- Access depends on your token’s permissions; some pages may be inaccessible.
- HTML-to-text is best-effort, so some formatting/tables may be simplified.

## Notes

- This app does not use the Vector Stores or Files APIs. It calls the Embeddings endpoint to build a local FAISS index under `data/` and uses Chat Completions for responses.
- PDF uploads are parsed locally if `pypdf` (or `PyPDF2`) is installed. Without it, uploading PDFs will return a clear error message.
- If you use a proxy or custom base URL (e.g., gateways), set them in the config.
- For production, set a `SECRET_KEY` environment variable for Flask sessions.
- Sessions: this app prefers server-side sessions when `Flask-Session` is installed. If you see a warning about the session cookie being too large, install the extra:
  ```sh
  pip install Flask-Session
  ```
  With `Flask-Session` present, the app stores session data on disk under `data/flask_sessions` and keeps the browser cookie small. Without it, the app falls back to signed cookies and automatically trims chat history and source previews to stay under browser cookie limits.

  Note: Some Flask-Session/Werkzeug version combinations can cause a TypeError about a bytes-like cookie value when signing the server-side session id. To maximize compatibility, this app disables signing for server-side sessions by default (the cookie only contains a random id). If you prefer signing, set `SESSION_USE_SIGNER=1` in the environment before starting the app.
