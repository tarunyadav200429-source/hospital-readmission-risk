# Dockerfile -- packages the prediction API into a portable container so it runs
# identically on any machine ("works on my laptop" -> "works everywhere").
#
# Build:  docker build -t readmission-api .
# Run:    docker run -p 8000:8000 readmission-api
# Then open http://127.0.0.1:8000/docs

# Start from a small official Python image.
FROM python:3.12-slim

# libgomp1 is the OpenMP runtime that XGBoost and LightGBM need at run time.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (this layer is cached unless requirements change).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code, config and the trained model artifacts into the image.
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/

# Document the port the API listens on.
EXPOSE 8000

# Start the API server. 0.0.0.0 = listen on all interfaces (required in Docker).
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
