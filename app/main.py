from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import os

from app.detector import run_detection
from app.utils import draw_detections, apply_nms

app = FastAPI(title="Open-Vocabulary Object Detection API")

IMAGE_PATH = os.path.join("data", "input.png")


class DetectionRequest(BaseModel):
    query: str
    threshold: float = 0.3


@app.post("/detect")
def detect(request: DetectionRequest):
    if not os.path.exists(IMAGE_PATH):
        raise HTTPException(status_code=404, detail=f"Input image not found at {IMAGE_PATH}")

    image = Image.open(IMAGE_PATH).convert("RGB")

    # Pass the query exactly as given by user — no expansion, no hardcoding
    # Multi-label queries like "book, bottle" are split inside run_detection
    detections = run_detection(image, request.query, threshold=request.threshold)

    # Remove overlapping duplicate boxes from multi-label separate forward passes
    detections = apply_nms(detections, iou_threshold=0.5)

    if not detections:
        return {
            "detections": [],
            "output_image_path": None,
            "message": "No objects detected. Try lowering the threshold or rephrasing the query.",
        }

    output_path = draw_detections(image, detections)

    return {
        "detections": detections,
        "output_image_path": output_path,
    }


@app.get("/result")
def get_result(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Result image not found.")
    return FileResponse(path, media_type="image/png")


@app.get("/")
def health():
    return {"status": "running", "message": "Object Detection API is live."}