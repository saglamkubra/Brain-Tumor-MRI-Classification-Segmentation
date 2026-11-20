from datetime import datetime, timezone
import json
from urllib.error import URLError

from fastapi import Path as RoutePath, Query
from fastapi.responses import Response

from .research import TOPICS, abstract, research
from xml.etree.ElementTree import ParseError
from .report import report

from contextlib import asynccontextmanager
from io import BytesIO
import math

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .language import message
from .explanation_lab import input_previews, intervention, validate_box
from .models import CLASSES, ROOT, Models
from .segmentation import explain, load_sam, png, sam_image, segment, swin_stage3_layer_name

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


def request_language(request):
    return "en" if request.headers.get("accept-language", "").lower().startswith("en") else "tr"


@app.exception_handler(HTTPException)
async def localized_http_error(request: Request, error: HTTPException):
    language = request_language(request)
    detail = error.detail
    if isinstance(detail, dict) and "message" in detail:
        detail = {**detail, "message": message(detail["message"], language)}
    else:
        detail = message(detail, language)
    return JSONResponse({"detail": detail}, status_code=error.status_code, headers=error.headers)


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
        "swin_stage3_target": models.swin_stage3_target,
        "swin_stage3_layer": swin_stage3_layer_name(models),
        "classes": CLASSES,
        "cnn_integrity": models.cnn_integrity if "CNN" in models.models else None,
        "models": [{"name": name, "checkpoint": models.versions.get(name),
                    "profile": ("v2" if name == "CNN" else models.swin_profile if name == "Swin-Tiny" else None), "loaded": name in models.models,
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
    for name in ("CNN", "Swin-Tiny"):
        try:
            predictions.append(await run_in_threadpool(models.predict, name, image))
        except (RuntimeError, ValueError) as exc:
            errors[name] = str(exc)
    if not predictions:
        raise HTTPException(503, {"message": "Tahmin üretilemedi.", "errors": errors})
    return {"timestamp": datetime.now(timezone.utc).isoformat(), "classes": CLASSES, "predictions": predictions, "errors": errors, "preview": png(sam_image(image))}

@app.post("/segment")
async def segment_image(request: Request, mode: str = "auto", box: str | None = None,
                        method: str = "gradcam", stage: int = 3, points: str = Query("", max_length=250),
                        allow_fallback: bool = False):
    if mode not in {"auto", "manual"} or (mode == "auto" and (box is not None or points)):
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
    prompts = []
    if points:
        try:
            for item in points.split(";"):
                x, y, label = map(int, item.split(","))
                if not (0 <= x < 224 and 0 <= y < 224 and label in (0, 1)):
                    raise ValueError
                if label == 1 and not (xs <= x <= xe and ys <= y <= ye):
                    raise ValueError
                prompts.append((x, y, label))
            if len(prompts) > 12 or len(set(prompts)) != len({(x, y) for x, y, _ in prompts}):
                raise ValueError
        except ValueError:
            raise HTTPException(422, "Noktaları görüntü üzerinde seçin. Dahil etme noktaları kutu içinde olmalı.") from None
    image = await uploaded_image(request)
    try:
        result = await run_in_threadpool(segment, request.app.state.models, image, coordinates, method, stage, prompts, allow_fallback)
        for field in ("message", "warning", "box_warning", "box_reason"):
            if field in result:
                result[field] = message(result[field], request_language(request))
        return result
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(503, f"Grad-CAM oluşturulamadı: {exc}") from None



@app.post("/explain")
async def explain_image(request: Request, stage: int = Query(3, ge=3, le=4),
                        model: str = Query("swin", pattern="^(swin|cnn)$")):
    image = await uploaded_image(request)
    name = "CNN" if model == "cnn" else "Swin-Tiny"
    try:
        label, _, device, overlay, layer = await run_in_threadpool(
            explain, request.app.state.models, image, sam_image(image), stage, name)
        return dict(classification=label, model=name, stage=stage if model == "swin" else None, device=device, overlay=overlay,
                    gradcam_layer=layer)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(503, f"Grad-CAM oluşturulamadı: {exc}") from None


@app.post("/model-inputs")
async def model_inputs(request: Request):
    image = await uploaded_image(request)
    try:
        return {"inputs": await run_in_threadpool(input_previews, request.app.state.models, image)}
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from None


@app.post("/intervene")
async def intervene_image(request: Request, mode: str = Query("box", pattern="^(box|gradcam|random)$"),
                          fill: str = Query("gray", pattern="^(gray|gaussian)$"),
                          stage: int = Query(3, ge=3, le=4), box: str | None = Query(None, max_length=100)):
    image = await uploaded_image(request)
    try:
        coordinates = validate_box(image, box) if mode == "box" or (mode == "random" and box is not None) else None
        if mode == "gradcam" and box is not None:
            raise ValueError("Bu müdahale için kutu gönderilmemelidir.")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    try:
        return await run_in_threadpool(intervention, request.app.state.models, image, mode, fill, coordinates, stage)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from None


@app.get("/performance")
def performance():
    path = ROOT / "results" / "web_v3_performance.json"
    if not path.is_file():
        raise HTTPException(404, "Bu test çalıştırmasının doğrulanmış ölçümleri henüz mevcut değil.")
    return FileResponse(path, media_type="application/json")


@app.get("/research")
def publications(label: str, q: str = Query("", max_length=120),
                 years: int = Query(0, ge=0, le=30), sort: str = "relevance"):
    if label not in TOPICS or sort not in {"relevance", "pub_date"}:
        raise HTTPException(422, "Geçersiz sınıf veya sıralama.")
    try:
        return {"articles": research(label, q.strip(), years, sort)}
    except (URLError, OSError, ValueError, KeyError):
        raise HTTPException(502, "PubMed’e şu anda erişilemiyor. Bağlantıyı kontrol edip yeniden deneyin.") from None


@app.get("/research/{pmid}")
def publication_abstract(pmid: str = RoutePath(pattern=r"^[0-9]{1,12}$")):
    try:
        return {"sections": abstract(pmid)}
    except (URLError, OSError, ValueError, ParseError):
        raise HTTPException(502, "PubMed özeti alınamadı.") from None


@app.post("/report")
async def download_report(request: Request):
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > 4_000_000:
            raise HTTPException(413, "Rapor isteği en fazla 4 MB olabilir.")
        body.extend(chunk)
    try:
        layer = swin_stage3_layer_name(request.app.state.models)
        pdf = await run_in_threadpool(report, json.loads(body), request_language(request), layer)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise HTTPException(422, f"Rapor hazırlanamadı: {exc}") from None
    return Response(pdf, media_type="application/pdf", headers={
        "Content-Disposition": 'attachment; filename="NEURO_AI_LAB_rapor.pdf"'})
