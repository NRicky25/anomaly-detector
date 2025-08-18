FROM python:3.11-slim-bullseye

WORKDIR /app

# Install system dependencies for pyodbc
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        apt-transport-https \
        gnupg \
        python3-dev \
        gcc \
        unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# Add Microsoft repository and install ODBC driver
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql18 && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

EXPOSE 80

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]
