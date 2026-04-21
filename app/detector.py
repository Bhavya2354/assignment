import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image

MODEL_ID = "IDEA-Research/grounding-dino-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = None
model = None

def load_model():
    global processor, model
    if model is None:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)


def _run_single_detection(image: Image.Image, query: str, threshold: float):
    text_prompt = query.strip().rstrip(".") + "."
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    h, w = image.size[1], image.size[0]
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        target_sizes=[(h, w)],
    )[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score.item() < threshold:
            continue
        x_min, y_min, x_max, y_max = [round(v.item(), 2) for v in box]
        detections.append({
            "label": label.strip(),
            "bbox": [x_min, y_min, x_max, y_max],
            "center": [round((x_min + x_max) / 2, 2), round((y_min + y_max) / 2, 2)],
            "confidence": round(score.item(), 4),
        })
    return detections


def run_detection(image: Image.Image, query: str, threshold: float = 0.3):
    load_model()

    sub_queries = [q.strip() for q in query.split(",") if q.strip()]

    all_detections = []
    for sub_query in sub_queries:
        detections = _run_single_detection(image, sub_query, threshold)
        all_detections.extend(detections)

    return all_detections