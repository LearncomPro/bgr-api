FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download silueta ONNX model (~43MB) - same quality as u2net, compressed size
RUN mkdir -p /app/models && \
    curl -L -o /app/models/u2net.onnx "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx"

COPY . .

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]