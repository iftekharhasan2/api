# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# Copy the rest of the application
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Set environment variables
ENV FLASK_APP=backend.api
ENV FLASK_ENV=production
ENV PORT=10000

# Expose the port
EXPOSE 10000

# Run the application with gunicorn
CMD ["gunicorn", "--config", "backend/gunicorn_config.py", "backend.api:app"] 