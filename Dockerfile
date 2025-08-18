# Use a lightweight official Python image as the base
FROM python:3.11-slim-bullseye

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends python3-dev gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first to leverage Docker cache
# This installs dependencies before copying the rest of the code,
# so if only code changes, dependencies don't need to be reinstalled.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container's /app directory
# The '.' after 'COPY' means copy from the current context (your project root)
# The '.' after '/app' means copy to the current working directory in the container (/app)
COPY . .

# Expose the port that the FastAPI application will run on
EXPOSE 80

# Define the command to run your FastAPI application using Uvicorn
# --host 0.0.0.0 makes the app accessible from outside the container
# --port 8000 is the port inside the container
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]