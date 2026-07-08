from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from rembg import remove, new_session
import io
import time
import threading
import urllib.request

app = FastAPI()

print("Loading rembg session...")
start = time.time()
session = new_session("silueta")
print(f"Session loaded in {time.time() - start:.1f}s")


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
    input_bytes = await file.read()

    output_bytes = remove(
        input_bytes,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
    )

    return Response(
        content=output_bytes,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=no_bg.png"},
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")