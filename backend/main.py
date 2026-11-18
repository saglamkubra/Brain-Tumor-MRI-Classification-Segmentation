from contextlib import asynccontextmanager
from io import BytesIO
import math

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .models import CLASSES, ROOT, Models
from .segmentation import load_sam, png, sam_image, segment

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 4096 * 4096
TYPES = {"image/png": "PNG", "image/jpeg": "JPEG"}


@asynccontextmanager
async def lifespan(app):
    app.state.models = await run_in_threadpool(Models)
    await run_in_threadpool(load_sam, app.state.models)
    yield


app = FastAPI(title="NEURO AI LAB", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=ROOT / "frontend"), name="arayuz")
app.mount("/results", StaticFiles(directory=ROOT / "results"), name="deneyler")


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/health")
def health(request: Request):
    models = request.app.state.models
    return {
        "status": "hazır" if not models.errors else "eksik",
        "device": models.device,
        "swin_profile": models.swin_profile,
        "classes": CLASSES,
        "models": [{"name": name, "loaded": name in models.models,
                    "device": ("cpu" if name in models.cpu_only else models.device)
                    if name in models.models else None, "error": models.errors.get(name)}
                   for name in (*models.models, *models.errors)],
    }


async def uploaded_image(request):
    mime = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if mime not in TYPES:
        raise HTTPException(415, "PNG veya JPG görüntüsü yükleyin.")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_BYTES:
            raise HTTPException(413, "Görüntü en fazla 10 MiB olabilir.")
        body.extend(chunk)
    try:
        with Image.open(BytesIO(body)) as image:
            if image.format != TYPES[mime]:
                raise HTTPException(415, "Dosya içeriği bildirilen PNG/JPG türüyle uyuşmuyor.")
            if max(image.size) > 4096 or image.width * image.height > MAX_PIXELS:
                raise HTTPException(413, "Görüntü boyutları en fazla 4096 × 4096 olabilir.")
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(422, "Görüntü okunamadı veya bozuk.") from None


@app.post("/predict")
async def predict(request: Request):
    image = await uploaded_image(request)
    models = request.app.state.models
    predictions, errors = [], {}
    for name in ("CNN", "ViT", "Swin-Tiny"):
        try:
            predictions.append(await run_in_threadpool(models.predict, name, image))
        except (RuntimeError, ValueError) as exc:
            errors[name] = str(exc)
    if not predictions:
        raise HTTPException(503, {"message": "Tahmin üretilemedi.", "errors": errors})
    return {"classes": CLASSES, "predictions": predictions, "errors": errors, "preview": png(sam_image(image))}

@app.post("/segment")
async def segment_image(request: Request, mode: str = "auto", box: str | None = None,
                        method: str = "gradcam", stage: int = 3):
    if mode not in {"auto", "manual"} or (mode == "auto" and box is not None):
        raise HTTPException(422, "Segmentasyon yöntemi auto veya manual olmalı.")
    if method not in {"gradcam", "intensity"} or stage not in {3, 4}:
        raise HTTPException(422, "Yöntem gradcam veya intensity, aşama 3 veya 4 olmalı.")
    coordinates = None
    if mode == "manual":
        try:
            xs, ys, xe, ye = map(float, (box or "").split(","))
            coordinates = [xs, ys, xe, ye]
            if not (all(map(math.isfinite, coordinates)) and 0 <= xs < xe <= 223 and 0 <= ys < ye <= 223):
                raise ValueError
        except ValueError:
            raise HTTPException(422, "Manuel kutu 224 × 224 görüntüde geçerli x1,y1,x2,y2 koordinatları içermeli.") from None
    image = await uploaded_image(request)
    try:
        return await run_in_threadpool(segment, request.app.state.models, image, coordinates, method, stage)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(503, f"Grad-CAM oluşturulamadı: {exc}") from None
