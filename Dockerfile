FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model at build time so it's baked into the image
RUN python -c "from transformers import pipeline; pipeline('image-segmentation', model='briaai/RMBG-1.4', trust_remote_code=True)"

# Copy app
COPY . .

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
