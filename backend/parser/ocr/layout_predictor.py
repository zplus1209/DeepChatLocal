from __future__ import annotations

import cv2
import logging
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, Image.Image, np.ndarray]


def is_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


class LayoutBox(BaseModel):
    label: str
    box: List[float] = Field(min_length=4, max_length=4)
    confidence: float = Field(ge=0.0, le=1.0)
    image: Optional[object] = Field(default=None, exclude=True)

    @field_validator("box")
    @classmethod
    def _box_valid(cls, v: List[float]) -> List[float]:
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid box {v}: x2 > x1 and y2 > y1 required.")
        return [float(c) for c in v]

    @property
    def x1(self) -> float:
        return self.box[0]

    @property
    def y1(self) -> float:
        return self.box[1]

    @property
    def x2(self) -> float:
        return self.box[2]

    @property
    def y2(self) -> float:
        return self.box[3]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def pil_image(self) -> Optional[Image.Image]:
        if self.image is None:
            return None
        return Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))


class LayoutPredictor:
    def __init__(
        self,
        *,
        layout_model_name: str = "yolov10_layout",
        doc_orientation_model_name: str = "PP-LCNet_x1_0_doc_ori",
        text_line_orientation_model_name: str = "PP-LCNet_x1_0_textline_ori",
        use_gpu: bool | None = None,
    ):
        self.layout_model_name = layout_model_name
        self.doc_orientation_model_name = doc_orientation_model_name
        self.text_line_orientation_model_name = text_line_orientation_model_name

        if use_gpu is None:
            self.use_gpu = is_cuda_available()
        else:
            self.use_gpu = use_gpu

    def predict(self, image: ImageInput) -> List[LayoutBox]:
        self.image = image
        bgr = self._to_bgr(self.image)

        if self.layout_model_name == "yolov10_layout":
            regions = self._yolov10_layout(bgr)
        else:
            regions = self._paddle_layout(bgr)

        corrected: List[LayoutBox] = []
        for box in regions:
            if self.text_line_orientation_model_name:
                corrected.extend(self._text_line_orientation(box))
            else:
                corrected.extend(self._image_orientation_cls(box))

        self.regions = corrected
        return corrected

    def plot(
        self,
        show_label: bool = True,
        color_map: Optional[Dict[str, tuple]] = None,
        save_path: Optional[str | Path] = None
    ) -> None:
        bgr = self._to_bgr(self.image)
        image_plot = bgr.copy()
        img_h, img_w = image_plot.shape[:2]

        default_colors = {
            "title": (1, 0, 0),
            "section-header": (0, 0, 1),
            "text": (0, 0.5, 0),
            "picture": (1, 0.5, 0),
            "image": (1, 0.5, 0),
            "table": (0.5, 0, 0.5),
            "page-footer": (0.5, 0.5, 0.5),
        }

        cmap = color_map if color_map is not None else default_colors

        def rgb_float_to_bgr_int(rgb):
            r, g, b = rgb
            return (int(b * 255), int(g * 255), int(r * 255))

        for region in self.regions:
            x1 = max(0, min(int(region.x1), img_w - 1))
            y1 = max(0, min(int(region.y1), img_h - 1))
            x2 = max(0, min(int(region.x2), img_w - 1))
            y2 = max(0, min(int(region.y2), img_h - 1))

            rgb = cmap.get(region.label, (0, 1, 0))
            color = rgb_float_to_bgr_int(rgb)

            cv2.rectangle(image_plot, (x1, y1), (x2, y2), color, 2)

            if show_label:
                cv2.putText(
                    image_plot,
                    f"{region.label} {region.confidence:.2f}",
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        rgb_image = cv2.cvtColor(image_plot, cv2.COLOR_BGR2RGB)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb_image).save(save_path)

        plt.figure(figsize=(10, 12))
        plt.imshow(rgb_image)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def _yolov10_layout(self, bgr: np.ndarray) -> List[LayoutBox]:
        from huggingface_hub import hf_hub_download
        from doclayout_yolo import YOLOv10

        ckpt = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        )
        model = YOLOv10(ckpt)
        device = "cuda" if self.use_gpu else "cpu"
        results = model.predict(bgr, imgsz=1024, conf=0.2, device=device)[0]

        raw_boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        regions = [
            LayoutBox(
                label=model.names[cls_id],
                box=box.tolist(),
                confidence=float(score),
                image=self._crop(bgr, box),
            )
            for box, score, cls_id in zip(raw_boxes, scores, class_ids)
        ]

        reading_order = self._get_reading_order(regions)
        regions = [regions[i] for i in reading_order]
        return regions

    def _paddle_layout(self, bgr: np.ndarray) -> List[LayoutBox]:
        from paddleocr import LayoutDetection

        layout_model = LayoutDetection(model_name=self.layout_model_name)
        output = layout_model.predict(bgr, batch_size=1, layout_nms=True)

        regions = [
            LayoutBox(
                label=block["label"],
                box=list(map(float, block["coordinate"])),
                confidence=float(block["score"]),
                image=self._crop(bgr, block["coordinate"]),
            )
            for res in output
            for block in res["boxes"]
        ]

        return regions

    def _image_orientation_cls(self, region: LayoutBox) -> List[LayoutBox]:
        from paddleocr import DocImgOrientationClassification

        model = DocImgOrientationClassification(
            model_name=self.doc_orientation_model_name
        )

        return [
            region.model_copy(
                update={
                    "image": self._rotate(region.image, res["label_names"][0])
                }
            )
            for res in model.predict(region.image, batch_size=1)
        ]

    def _text_line_orientation(self, region: LayoutBox) -> List[LayoutBox]:
        from paddleocr import TextLineOrientationClassification

        model = TextLineOrientationClassification(
            model_name=self.text_line_orientation_model_name
        )

        return [
            region.model_copy(
                update={
                    "image": self._rotate(region.image, res["label_names"][0])
                }
            )
            for res in model.predict(region.image, batch_size=1)
        ]

    @staticmethod
    def _get_reading_order(regions: List[LayoutBox]) -> List[int]:
        import torch
        from transformers import LayoutLMv3ForTokenClassification
        from .layoutreader.v3.helpers import (
            prepare_inputs,
            boxes2inputs,
            parse_logits,
        )

        if not regions:
            return []

        layout_order_model = LayoutLMv3ForTokenClassification.from_pretrained(
            "hantian/layoutreader"
        )
        layout_order_model.eval()

        max_x = max(r.x2 for r in regions) * 1.1 or 1.0
        max_y = max(r.y2 for r in regions) * 1.1 or 1.0

        norm_boxes = [
            [
                int((r.x1 / max_x) * 1000),
                int((r.y1 / max_y) * 1000),
                int((r.x2 / max_x) * 1000),
                int((r.y2 / max_y) * 1000),
            ]
            for r in regions
        ]

        inputs = prepare_inputs(boxes2inputs(norm_boxes), layout_order_model)

        with torch.no_grad():
            logits = layout_order_model(**inputs).logits.cpu().squeeze(0)

        reading_order = parse_logits(logits, len(norm_boxes))
        return reading_order

    @staticmethod
    def _to_bgr(image: ImageInput) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        path = str(image)
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return bgr

    @staticmethod
    def _crop(bgr: np.ndarray, box: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, box)
        h, w = bgr.shape[:2]
        return bgr[max(0, y1): min(h, y2), max(0, x1): min(w, x2)]

    @staticmethod
    def _rotate(img: np.ndarray, angle: str) -> np.ndarray:
        return {
            "0": img,
            "90": cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            "180": cv2.rotate(img, cv2.ROTATE_180),
            "270": cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
        }.get(angle, img)