# pizza_monitoring/detection_service/config.py

RABBITMQ_HOST = "localhost"
INPUT_QUEUE = "video_frames"
OUTPUT_QUEUE = "detections"
PROCESSED_QUEUE = "processed_frames"
MODEL_PATH = "best.pt"  # use your fine-tuned model
VIOLATION_DIR = "frames"
DB_PATH = "violations.db"