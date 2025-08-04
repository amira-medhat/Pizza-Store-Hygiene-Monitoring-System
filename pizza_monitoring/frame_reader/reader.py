import sys
import os
# Add root directory to sys.path (pizza_monitoring)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.config import RABBITMQ_HOST, RABBITMQ_QUEUE, VIDEO_SOURCE

import cv2
import pika
import base64
import time



def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def connect_rabbitmq():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=RABBITMQ_QUEUE, durable=False)
    return channel

def main():
    print(f"[FrameReader] 🎥 Reading from: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("[FrameReader] ❌ Failed to open video source.")
        return

    channel = connect_rabbitmq()
    print(f"[FrameReader] 🟢 Connected to RabbitMQ queue '{RABBITMQ_QUEUE}'")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[FrameReader] 📺 End of video or error.")
            break

        frame_data = encode_frame(frame)
        channel.basic_publish(
            exchange='',
            routing_key=RABBITMQ_QUEUE,
            body=frame_data.encode('utf-8')
        )
        print("[FrameReader] 📤 Frame sent.")

    cap.release()
    print("[FrameReader] ✅ Done.")

if __name__ == "__main__":
    main()
