FROM python:3.11-slim

# System deps needed by PyMuPDF + Google Cloud SDK
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Google credentials: the JSON content is passed as an env var
# GOOGLE_CREDENTIALS_JSON at runtime → written to a temp file
# This avoids baking secrets into the image
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/bin/sh", "/entrypoint.sh"]
