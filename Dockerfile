# ── Full stack — PC / server (x86_64) ────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System libraries required by OpenCV and WebRTC/av
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgstreamer1.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-backend.txt requirements-frontend.txt ./
RUN pip install --no-cache-dir \
    -r requirements-backend.txt \
    -r requirements-frontend.txt

# Copy application source (data/models/outputs are mounted as volumes)
COPY app.py config.py pipeline.py streamlit_callback.py main.py ./
COPY components/ ./components/
COPY src/ ./src/

# Create mount-point directories (populated by docker-compose volumes)
RUN mkdir -p data models outputs

# Configure Streamlit via ENV — .streamlit/ is gitignored so not available after clone
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
