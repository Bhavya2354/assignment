import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255),
    (255, 165, 0), (128, 0, 128), (0, 255, 255),
]


def apply_nms(detections: list, iou_threshold: float = 0.5) -> list:
    """
    Non-Maximum Suppression to remove overlapping duplicate detections.
    Keeps the highest confidence box when two boxes overlap above iou_threshold.
    """
    if len(detections) == 0:
        return detections

    boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
    scores = np.array([d["confidence"] for d in detections], dtype=np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou < iou_threshold]

    return [detections[i] for i in keep]


def draw_detections(image: Image.Image, detections: list) -> str:
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for i, det in enumerate(detections):
        color = COLORS[i % len(COLORS)]
        x_min, y_min, x_max, y_max = [int(v) for v in det["bbox"]]
        label = det["label"]
        conf = det["confidence"]

        cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), color, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_cv, (x_min, y_min - th - 8), (x_min + tw + 4, y_min), color, -1)
        cv2.putText(img_cv, text, (x_min + 2, y_min - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cx, cy = int(det["center"][0]), int(det["center"][1])
        cv2.circle(img_cv, (cx, cy), 4, color, -1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"result_{timestamp}.png")
    cv2.imwrite(output_path, img_cv)
    return output_path