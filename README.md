DEMO_VIDEO = https://drive.google.com/file/d/1mXvHxaNF0hfPoyiGrhllKII_4-5edIRs/view?usp=sharing

Open-Vocabulary Object Detection API
Overview
This project is a text-driven object detection system built using GroundingDINO and FastAPI. The system accepts a natural language query describing an object, detects it in a given static image, and returns structured results including bounding boxes, center coordinates, and confidence scores. Every detection is also saved as an annotated image with boxes and labels drawn on it.

The Assignment
The task was to build an API where a user provides a text description of an object and the system detects it in a provided image using an open-vocabulary detection model. The image contains a book, a cardboard cargo box, a glass liquor bottle, a plastic water bottle, a marker, and a wine glass. The system had to handle natural language queries like "glass bottle with liquor in it", "plastic water bottle", and "white book" and return bounding boxes, center pixel coordinates, and confidence scores for each detected object.

Approach
The model used is IDEA-Research/grounding-dino-base loaded from HuggingFace. GroundingDINO is an open-vocabulary detection model that works by fusing a text encoder and an image encoder through cross-attention. It grounds natural language descriptions onto image regions without needing any fixed class labels or retraining. Whatever you describe in text, it tries to find in the image.
The system is structured into three components. The FastAPI server in main.py handles all HTTP routing and request processing. The detection logic in detector.py manages model loading and inference. The utility functions in utils.py handle post-processing and visualization.

Multi-Query Handling
One key design decision was how to handle comma-separated multi-label queries like "book, cargo box". Running both labels together in a single forward pass causes the model to treat them as one combined token context, where the dominant label suppresses the other. To solve this, each comma-separated label is split and run as a completely independent forward pass through the model. The results from all passes are then merged into one unified response. This ensures every label gets the model's full attention and nothing gets missed.

Challenges
The glass liquor bottle and the plastic water bottle in the image are both transparent with no strong visual difference between them. Short queries like "glass bottle" and "plastic bottle" are not enough for the model to differentiate them reliably. Since GroundingDINO uses the full text embedding of the query, descriptive phrases with shape or size context work better — the text side compensates for what the visual features cannot distinguish alone.
Running separate forward passes per label also means the same object can sometimes be detected twice with slightly overlapping boxes. Non-Maximum Suppression is applied after merging all detections to clean this up. It keeps the highest confidence box and removes any other box that overlaps it beyond a 50 percent IoU threshold, resulting in one clean detection per object.

Project Structure
//
assignment/
├── app/
│   ├── main.py        # FastAPI server
│   ├── detector.py    # Model inference
│   └── utils.py       # NMS and visualization
├── data/
│   └── input.png      # Input image
├── outputs/           # Saved annotated results
└── requirements.txt   # Dependencies
//

Setup and Running
pip install -r requirements.txt
uvicorn app.main:app --reload
API docs available at http://127.0.0.1:8000/docs

API
POST /detect
Accepts a query string and an optional threshold (default 0.3). Returns a list of detections each containing a label, bounding box as [x_min, y_min, x_max, y_max], center as [x_center, y_center], confidence score, and the path to the saved annotated image.
GET /result
Returns the annotated output image by file path.
GET /
Health check.

Conclusion
This project demonstrates a working end-to-end open-vocabulary object detection pipeline where natural language is the only input needed to detect any object in an image. The core engineering decisions — running separate forward passes per label, applying NMS for deduplication, and using descriptive queries to handle visually ambiguous objects — address the real limitations of running GroundingDINO base on a CPU with complex multi-object scenes. The result is a clean, modular API that fulfills all assignment requirements and handles edge cases without any hardcoding or model fine-tuning.
