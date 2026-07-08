from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from PIL import Image
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import io
import time
import threading
import urllib.request

app = FastAPI()

# Load ONNX model once at startup
print("Loading U2Net ONNX model...")
start = time.time()
model_path = hf_hub_download("danielgatis/rembg", "u2net.onnx", cache_dir="/app/models")
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print(f"Model loaded in {time.time() - start:.1f}s")


def preprocess(image, size=320):
    """Resize and normalize image for U2Net."""
    img = image.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
    return arr.astype(np.float32)


def postprocess(output, original_size):
    """Convert model output to alpha mask."""
    mask = output[0][0, 0]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.resize(original_size, Image.BILINEAR)
    return mask_img


# Self-ping to prevent Render free tier from sleeping
def keep_alive():
    import os
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        return
    while True:
        time.sleep(600)
        try:
            urllib.request.urlopen(f"{url}/health", timeout=10)
            print("Keep-alive ping sent")
        except Exception as e:
            print(f"Keep-alive ping failed: {e}")

threading.Thread(target=keep_alive, daemon=True).start()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    original_size = image.size

    input_tensor = preprocess(image)
    outputs = session.run(None, {input_name: input_tensor})
    mask = postprocess(outputs, original_size)

    image_rgba = image.convert("RGBA")
    image_rgba.putalpha(mask)

    buf = io.BytesIO()
    image_rgba.save(buf, format="PNG", optimize=True)
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=no_bg.png"},
    )


# Static files LAST
app.mount("/", StaticFiles(directory="static", html=True), name="static")