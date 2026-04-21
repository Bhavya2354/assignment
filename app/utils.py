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