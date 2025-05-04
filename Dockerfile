FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    openssl \
    curl \
    ca-certificates \
    gnupg \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set up MariaDB's Python connector dependencies
RUN curl -LsSO https://r.mariadb.com/downloads/mariadb_repo_setup && \
    echo "c4a0f3dade02c51a6a28ca3609a13d7a0f8910cccbb90935a2f218454d3a914a  mariadb_repo_setup" | sha256sum -c - && \
    chmod +x mariadb_repo_setup && \
    ./mariadb_repo_setup --mariadb-server-version="mariadb-11.7" && \
    rm mariadb_repo_setup && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libmariadb3 \
    libmariadb-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy project files
COPY . /app

# Install project dependencies
RUN uv sync

EXPOSE 8000

CMD ["uv", "run", "mcp-server-mariadb-vector", "--host", "0.0.0.0", "--transport", "sse"]
