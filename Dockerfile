FROM python:3.11-slim-bullseye

WORKDIR /app

# Install system dependencies for pyodbc and Azure SQL
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev gcc unixodbc-dev curl apt-transport-https gnupg && \
    # Add Microsoft repo and install ODBC driver
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql18 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]
