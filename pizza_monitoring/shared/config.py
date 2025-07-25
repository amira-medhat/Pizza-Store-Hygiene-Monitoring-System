import os

# Get RabbitMQ host from environment variable or use default
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")

# Queue names
PROCESSED_QUEUE = "processed_frames"
RABBITMQ_QUEUE = "video_frames"

# Use relative paths for Docker compatibility
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/weights/best.pt")
DB_PATH = os.environ.get("DB_PATH", "/app/shared/violations.db")

# Video source - can be a file, RTSP stream, or webcam
# VIDEO_SOURCE = os.environ.get("VIDEO_SOURCE", "0")  # Default to webcam
VIDEO_SOURCE = os.environ.get("VIDEO_SOURCE", "/app/video_source.mp4")