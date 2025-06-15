# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (for nltk, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data at build time (punkt for tokenization)
RUN python -m nltk.downloader punkt

# Copy the application code
COPY ./api ./api

# Expose the port (optional, for documentation)
EXPOSE 10000

# Use the PORT environment variable if set (for Render), else default to 10000
ENV PORT=10000

CMD ["python", "-m", "api.main"]