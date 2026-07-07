from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from PIL import Image
from transformers import pipeline
import io
import time
import threading
import urllib.request

app = FastAPI()

# Load model once at startup
print("Loading RMBG-1.4 model...")
start = time.time()
segmenter = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
print(f"Model loaded in {time.time() - start:.1f}s")


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

    result = segmenter(image)

    mask = result[0]["mask"]

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


# Static files LAST — so API routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")