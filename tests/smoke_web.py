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
    parser.add_argument('--pubmed', action='store_true', help='Check the live PubMed service (internet required).')
    args = parser.parse_args()

    def request(route, *params, **kwargs):
        return exchange(args.root, route, *params, **kwargs)

    with ZipFile(args.dataset) as archive:
        sample = archive.read(SAMPLE)
        healthy = archive.read(next(path for path in archive.namelist()
                                    if path.startswith("Testing/notumor/") and path.endswith(".jpg")))
    assert b'/assets/app.js' in request("/")
    assert b'camResult' in request("/assets/app.js")
    assert request("/results/swin_confusion_matrix.png").startswith(b"\x89PNG")
    health = request("/health")
    assert health["device"] == args.device, health
    assert len(health["models"]) == 3 and all(model["loaded"] for model in health["models"]), health
    predictions = request("/predict", sample)
    assert predictions["classes"] == CLASSES and not predictions["errors"], predictions
    assert len(predictions["predictions"]) == 2
    image_result(predictions["preview"])
    for prediction in predictions["predictions"]:
        assert prediction["class"] == "meningioma" and prediction["device"] == args.device, prediction
        assert list(prediction["probabilities"]) == CLASSES
        assert abs(sum(prediction["probabilities"].values()) - 1) < 1e-6
    print("PREDICT", predictions["predictions"])
    cams = [request(f"/explain?stage={stage}", sample)["overlay"] for stage in (3, 4)]
    assert cams[0] != cams[1]
    for cam in cams:
        image_result(cam)

    cnn_cam = request("/explain?model=cnn", sample)
    cnn = next(item for item in predictions["predictions"] if item["model"] == "CNN")
    assert cnn_cam["model"] == "CNN" and cnn_cam["stage"] is None
    assert cnn_cam["classification"] == cnn["class"]
    assert image_result(cnn_cam["gradcam_layer"]).mode == "RGBA"
    request("/explain?model=unknown", sample, expected=422)
    for image in ("/results/CNN_V2%20sonu%C3%A7lar/07_confusion_matrix.png",
                  "/results/swin_v3/swin_sam_grafikleri/10_sam_ve_gradcam_dort_sinif.png"):
        assert request(image).startswith(b"\x89PNG")

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
    assert len(request("/predict", png_input.getvalue(), "image/png")["predictions"]) == 2
    request("/segment?mode=manual&box=2,2,1,1", sample, expected=422)
    request("/predict", sample, "image/png", expected=415)
    request("/predict", b"bozuk", expected=422)
    request("/predict", b"x" * (10 * 1024 * 1024 + 1), expected=413)
    oversized = BytesIO()
    Image.new("RGB", (4097, 1)).save(oversized, format="PNG")
    request("/predict", oversized.getvalue(), "image/png", expected=413)
    references = []
    if args.pubmed:
        articles = request("/research?label=glioma&years=5&sort=relevance")["articles"]
        assert articles and all(row["id"].isdigit() for row in articles)
        for article in articles[:4]:
            sections = request(f'/research/{article["id"]}')["sections"]
            if sections:
                references = [article]
                break
        assert references, "PubMed returned no abstracts for the first four articles."
        print("PubMed and abstract", references[0]["id"])

    for include_images in (False, True):
        visuals = {"original": {"image": predictions["preview"]},
                   "cnn_cam": {"image": cnn_cam["overlay"]}} if include_images else {}
        payload = json.dumps({"analysis": predictions, "visuals": visuals, "references": references}).encode("utf8")
        pdf = request("/report", payload, "application/json")
        assert pdf.startswith(b"%PDF") and (b"/Subtype /Image" in pdf) == include_images
    print("PDF, with and without images, passed.")
    print("PNG, notumor skip and upload / box validation passed.")


if __name__ == "__main__":
    main()
