FROM python:3.11-slim

# Install Tesseract and required system libraries
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better Docker caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .
 

# Start the application
# CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "10000"]
CMD uvicorn backend.app:app --host 0.0.0.0 --port $PORT