"""Compare the pre-guard and current SAM alternative on a fixed V3 stratified sample."""
import argparse
import csv
from io import BytesIO
from pathlib import Path
import random
from types import ModuleType
from zipfile import ZipFile

from PIL import Image

from backend.models import CLASSES, Models
from backend import segmentation

EXCLUDED = {
    "Te-gl_22.jpg", "Te-aug-me_7.jpg", "Te-gl_1.jpg", "Te-me_263.jpg",
    "Te-pi_1.jpg", "Te-no_1.jpg", "Te-gl_103.jpg", "Te-gl_102.jpg",
    "glioma-original.jpg", "meningioma-original.jpg", "pituitary-original.jpg",
    "notumor-original.jpg",
}
SEED = 20261119


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("previous", type=Path, help="Trusted pre-guard backend/segmentation.py snapshot.")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    previous = ModuleType("backend.segmentation_before_guard")
    exec(compile(args.previous.read_bytes(), str(args.previous), "exec"), previous.__dict__)
    with args.split.open(encoding="utf-8-sig", newline="") as source:
        split = list(csv.DictReader(source))
    rng = random.Random(SEED)
    models = Models()
    segmentation.load_sam(models)
    assert models.swin_profile == "v3" and {"Swin-Tiny", "SAM ViT-B"} <= models.models.keys(), models.errors
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.archive) as dataset, args.output.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["class", "image", "prediction", "old_status", "old_source", "old_box",
                         "old_pixels", "new_status", "new_source", "new_box", "new_pixels"])
        for label in CLASSES:
            available = sorted(row["dosya"] for row in split if row["sinif"] == label
                               and Path(row["dosya"]).name not in EXCLUDED)
            for path in rng.sample(available, 5):
                with Image.open(BytesIO(dataset.read(path))) as source:
                    image = source.convert("RGB")
                old = previous.segment(models, image, None, stage=3, allow_fallback=True)
                new = segmentation.segment(models, image, None, stage=3, allow_fallback=True)
                assert old["classification"] == new["classification"]
                def cells(result):
                    return [result["status"], result.get("box_source"), result.get("box"), result.get("pixels")]
                record = [label, Path(path).name, new["classification"], *cells(old), *cells(new)]
                writer.writerow(record)
                output.flush()
                print(record, flush=True)


if __name__ == "__main__":
    main()
