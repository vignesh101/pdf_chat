ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE} AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python deps. Optionally include heavy TTS stack at build time.
COPY requirements.txt ./
ARG INSTALL_TTS=false
RUN set -eux; \
    if [ "$INSTALL_TTS" = "true" ]; then \
      pip install --no-cache-dir -r requirements.txt; \
    else \
      # Build a reduced requirements list without TTS extras to keep image slim
      awk 'BEGIN{IGNORECASE=1} !/^\s*(TTS|pyttsx3)\b/{print}' requirements.txt > /tmp/requirements-no-tts.txt; \
      pip install --no-cache-dir -r /tmp/requirements-no-tts.txt; \
    fi

# Copy app source
COPY . .

# Default Flask settings; override as needed at runtime
ENV FLASK_APP=app.py \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=5000

EXPOSE 5000

# For production consider using gunicorn: 
#   CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:create_app()"]
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port", "5000"]

