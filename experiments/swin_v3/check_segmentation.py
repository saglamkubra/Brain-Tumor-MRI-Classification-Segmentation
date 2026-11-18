"""Check real checkpoint inference and segmentation against archived V3 samples."""
import argparse
import json
import os
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from backend.models import Models
from backend.segmentation import load_sam, segment

SAMPLES = {
    "meningioma": "Testing/meningioma/Te-me_263.jpg",
    "glioma": "Testing/glioma/Te-gl_1.jpg",
    "pituitary": "Testing/pituitary/Te-pi_1.jpg",
    "notumor": "Testing/notumor/Te-no_1.jpg",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    os.environ["NEURO_SWIN_PROFILE"] = "v3"
    models = Models()
    load_sam(models)
    assert not models.errors, models.errors
    results = {}
    with ZipFile(args.archive) as archive:
        for label, path in SAMPLES.items():
            with archive.open(path) as handle:
                image = Image.open(handle).convert("RGB")
            records = {}
            for stage in (3, 4):
                result = segment(models, image, None, "intensity", stage)
                assert result["gradcam"].startswith("data:image/png;base64,")
                assert ("mask" in result) == (result["status"] == "hazır"), result["status"]
                if result["classification"] == "notumor":
                    assert result["warning"]
                records[f"intensity_stage_{stage}"] = {
                    key: result[key] for key in ("status", "classification", "box", "score", "pixels", "warning")
                    if key in result
                }
            assert records["intensity_stage_3"].get("box") == records["intensity_stage_4"].get("box")
            results[label] = records
        with archive.open(SAMPLES["meningioma"]) as handle:
            image = Image.open(handle).convert("RGB")
        manual = segment(models, image, [115, 85, 160, 135])
        assert manual["status"] == "hazır", manual.get("message")
        results["manual_meningioma"] = {"status": manual["status"], "pixels": manual["pixels"]}
    blank = segment(models, Image.new("RGB", (224, 224)), None, "intensity")
    assert blank["status"] == "kutu_yok" and "mask" not in blank
    results["blank"] = blank["status"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
