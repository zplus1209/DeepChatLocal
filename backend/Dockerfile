FROM python:3.11-slim

# System deps: LibreOffice for .docx/.pptx conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
        libreoffice-writer \
        libreoffice-impress \
        libreoffice-calc \
        libgl1 \
        libglib2.0-0 \
        curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Data volume
RUN mkdir -p /data/qdrant_db /data/parser_output

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
