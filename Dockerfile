FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the ONNX model at build time (~44MB)
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('danielgatis/rembg', 'u2net.onnx', cache_dir='/app/models')"

COPY . .

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]