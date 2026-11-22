import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException, Request
from PIL import Image

from backend.explanation_lab import region_mask, random_pixels
from backend.main import app, intervene_image


class InterventionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        buffer = BytesIO()
        Image.new("RGB", (200, 100)).save(buffer, format="PNG")
        self.data = buffer.getvalue()

    async def post(self, box, fill="gray"):
        async def receive():
            return {"type": "http.request", "body": self.data, "more_body": False}

        request = Request({"type": "http", "headers": [(b"content-type", b"image/png")],
                           "app": app}, receive)
        return await intervene_image(request, mode="box", fill=fill, stage=3, box=box)

    async def test_invalid_boxes_rejected_before_inference(self):
        with patch("backend.main.intervention") as infer:
            for box in ("", "0,0,224,224", "20,100,10,120", "nan,60,100,90"):
                with self.subTest(box=box), self.assertRaises(HTTPException) as error:
                    await self.post(box)
                self.assertEqual(error.exception.status_code, 422)
            infer.assert_not_called()

    async def test_large_box_rejected(self):
        with patch("backend.main.intervention") as infer, self.assertRaises(HTTPException) as error:
            await self.post("0,56,224,168")
        self.assertEqual(error.exception.status_code, 422)
        self.assertIn("yarısını", error.exception.detail)
        infer.assert_not_called()

    async def test_valid_box_maps_to_native_coordinates(self):
        with patch.object(app.state, "models", SimpleNamespace(), create=True), patch(
            "backend.main.intervention", return_value={"results": []}
        ) as infer:
            response = await self.post("0,56,112,112", fill="gaussian")
        self.assertEqual(response, {"results": []})
        self.assertEqual(infer.call_args.args[2:], ("box", "gaussian", (0, 0, 100, 50), 3))

    def test_equal_area_random_pixels_are_repeatable(self):
        image = Image.new("RGB", (200, 100))
        heat = np.arange(224*224, dtype=np.float32).reshape(224, 224)
        with patch("backend.explanation_lab.explain", return_value=(None, heat, None, None, None)):
            cam = region_mask(None, image, "gradcam", None, 3)
        random = region_mask(None, image, "random", None, 3)
        self.assertEqual(int(cam.sum()), 2000)
        self.assertEqual(int(random.sum()), int(cam.sum()))
        np.testing.assert_array_equal(random, region_mask(None, image, "random", None, 3))
        self.assertEqual(len(set(random_pixels(8, 5, 12))), 12)
        occupied = np.flatnonzero(random.any(axis=1))
        self.assertEqual(len(occupied), occupied[-1] - occupied[0] + 1)

    def test_random_control_matches_a_non_default_box_area(self):
        image = Image.new("RGB", (200, 100))
        box = (5, 7, 49, 27)
        primary = region_mask(None, image, "box", box, 3)
        control = region_mask(None, image, "random", box, 3)
        self.assertEqual(int(control.sum()), int(primary.sum()))
        self.assertEqual(int(control.sum()), 880)
        np.testing.assert_array_equal(control, region_mask(None, image, "random", box, 3))

    def test_random_control_never_crosses_primary_box_across_seeds(self):
        width, height = 80, 60
        box = (10, 13, 59, 39)
        count = (box[2]-box[0])*(box[3]-box[1])
        for seed in range(20):
            indices = random_pixels(width, height, count, exclude=box, seed=seed)
            self.assertEqual(len(np.unique(indices)), count)
            y, x = np.divmod(indices, width)
            self.assertFalse(((x >= box[0]) & (x < box[2]) &
                              (y >= box[1]) & (y < box[3])).any())

    def test_random_control_reports_no_available_patch(self):
        with self.assertRaisesRegex(ValueError, "kontrol yaması bulunamadı"):
            random_pixels(80, 60, 1800, exclude=(10, 15, 70, 45))
