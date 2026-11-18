"""Test the running local server with the original MRI dataset ZIP."""
import argparse
import base64
from io import BytesIO
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zipfile import ZipFile

from PIL import Image

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
SAMPLE = "Testing/meningioma/Te-me_263.jpg"


def exchange(root, route, content=None, mime="image/jpeg", expected=200):
    headers = {"Content-Type": mime} if content is not None else {}
    try:
        response = urlopen(Request(root + route, content, headers), timeout=180)
    except HTTPError as exc:
        response = exc
    with response:
        assert response.status == expected, (route, response.status, response.read())
        body = response.read()
        return json.loads(body) if response.headers.get_content_type() == "application/json" else body


def image_result(uri):
    prefix, payload = uri.split(",", 1)
    assert prefix == "data:image/png;base64"
    result = Image.open(BytesIO(base64.b64decode(payload)))
    assert result.size == (224, 224)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--root", default="http://127.0.0.1:8000")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    def request(route, *params, **kwargs):
        return exchange(args.root, route, *params, **kwargs)

    with ZipFile(args.dataset) as archive:
        sample = archive.read(SAMPLE)
        healthy = archive.read(next(path for path in archive.namelist()
                                    if path.startswith("Testing/notumor/") and path.endswith(".jpg")))
    assert b'/assets/app.js' in request("/")
    assert b'showVisuals' in request("/assets/app.js")
    assert request("/results/swin_confusion_matrix.png").startswith(b"\x89PNG")
    health = request("/health")
    assert health["device"] == args.device, health
    assert len(health["models"]) == 4 and all(model["loaded"] for model in health["models"]), health
    predictions = request("/predict", sample)
    assert predictions["classes"] == CLASSES and not predictions["errors"], predictions
    assert len(predictions["predictions"]) == 3
    image_result(predictions["preview"])
    for prediction in predictions["predictions"]:
        assert prediction["class"] == "meningioma" and prediction["device"] == args.device, prediction
        assert list(prediction["probabilities"]) == CLASSES
        assert abs(sum(prediction["probabilities"].values()) - 1) < 1e-6
    print("PREDICT", predictions["predictions"])

    for route in ["/segment?mode=auto", "/segment?mode=manual&box=115,85,160,135"]:
        result = request(route, sample)
        assert result["status"] == "hazır" and result["device"] == args.device, result.get("message")
        if "mode=auto" in route:
            image_result(result["gradcam"])
            assert result["gradcam_device"] == args.device
        image_result(result["overlay"])
        mask = image_result(result["mask"])
        assert 0 < mask.histogram()[255] == result["pixels"] < 224 * 224
        print("SEGMENT", route, {key: result[key] for key in ("box", "score", "pixels", "device")})

    skipped = request("/segment?mode=auto", healthy)
    assert skipped["classification"] == "notumor" and skipped["status"] == "atlandı", skipped.get("message")
    assert "mask" not in skipped and "overlay" not in skipped
    image_result(skipped["gradcam"])
    png_input = BytesIO()
    Image.open(BytesIO(sample)).save(png_input, format="PNG")
    assert len(request("/predict", png_input.getvalue(), "image/png")["predictions"]) == 3
    request("/segment?mode=manual&box=2,2,1,1", sample, expected=422)
    request("/predict", sample, "image/png", expected=415)
    request("/predict", b"bozuk", expected=422)
    request("/predict", b"x" * (10 * 1024 * 1024 + 1), expected=413)
    oversized = BytesIO()
    Image.new("RGB", (4097, 1)).save(oversized, format="PNG")
    request("/predict", oversized.getvalue(), "image/png", expected=413)
    print("PNG, notumor skip and upload / box validation passed.")


if __name__ == "__main__":
    main()
