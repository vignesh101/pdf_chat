# Document Chat (OpenAI Assistants + Files)

A simple Python web app to upload documents and chat with them using the OpenAI Assistants API (v2) with file search. Includes a minimal yet polished chat UI and a flexible config supporting custom proxies and base URLs.

## Features

- Upload files to OpenAI and attach to a vector store
- Assistant with `file_search` tool to ground answers in uploaded files
- Per-session threads; persistent assistant/vector store
- Configurable via `config.toml` and environment variables
- Optional HTTP proxy and SSL verification toggle

## Configuration

Create a `config.toml` at the project root or export environment variables. The following keys are supported (environment variable in parentheses):

- `proxy_url` (`PROXY_URL`) — HTTP/HTTPS proxy URL (e.g., `http://localhost:7890`).
- `openai_base_url` (`OPENAI_BASE_URL`) — Override the OpenAI API base URL (e.g., for gateways).
- `openai_api_key` (`OPENAI_API_KEY`) — Your OpenAI API key.
- `disable_ssl` (`DISABLE_SSL`) — Set to `true` to disable SSL verification (use only for trusted setups).
- `model_name` (`MODEL_NAME`) — The model to use for the Assistant (e.g., `gpt-4o-mini`).

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
   FLASK_APP=app.py FLASK_ENV=development flask run --port 5000
   ```

4. Open `http://localhost:5000` in your browser.

## Notes

- This app uses Assistants API v2 with `file_search` and a vector store. First run creates and persists the assistant/vector store IDs under `data/state.json`.
- Files uploaded through the UI are sent to the OpenAI Files API with `purpose="assistants"`.
- If you use a proxy or custom base URL (e.g., gateways), set them in the config.
- For production, set a `SECRET_KEY` environment variable for Flask sessions.

