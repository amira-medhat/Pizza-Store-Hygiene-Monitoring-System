import sys
import os
# Add root directory to sys.path (pizza_monitoring)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pika
import json
import base64
import cv2
import time
from concurrent.futures import ThreadPoolExecutor

from detection_logic import process_frame
from utils import decode_base64_frame
from shared.config import RABBITMQ_HOST, RABBITMQ_QUEUE, PROCESSED_QUEUE

# Use a single worker for detection_logic to avoid race conditions with global variables
# But use a separate thread pool for encoding/publishing to maintain throughput
executor = ThreadPoolExecutor(max_workers=1)  # Single worker for detection logic
publish_executor = ThreadPoolExecutor(max_workers=3)  # Multiple workers for publishing

# Shared state dictionary that will be passed to process_frame
detection_state = None
current_frame_id = 0

def encode_frame(frame):
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"[Detector] ❌ Error encoding frame: {e}")
        return ""

def publish_result(result, frame_b64, start_time):
    """Handle publishing results to RabbitMQ (runs in separate thread pool)"""
    message = {
        "timestamp": result["timestamp"],
        "frame_id": result["frame_id"],
        "is_violation": result["is_violation"],
        "is_safe_pickup": result["is_safe_pickup"],
        "labels": result["labels"],
        "boxes": result["boxes"],
        "frame": frame_b64
    }

    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
        channel = connection.channel()
        channel.queue_declare(queue=PROCESSED_QUEUE, durable=True)
        channel.confirm_delivery()

        message_body = json.dumps(message).encode('utf-8')
        channel.basic_publish(
            exchange='',
            routing_key=PROCESSED_QUEUE,
            body=message_body
        )

        end_time = time.time()
        print(f"[Detector] 🟢 Frame {result['frame_id']} processed and sent in {end_time - start_time:.2f} seconds")
        print("[Detector] ✅ Message published to processed_frames")

    except Exception as e:
        print(f"[Detector] ❌ Failed to publish message: {e}")
    finally:
        if 'connection' in locals() and connection.is_open:
            connection.close()

def handle_detection_task(body):
    """Process frame detection in a single thread to avoid race conditions"""
    global detection_state, current_frame_id
    
    start_time = time.time()
    print("[Detector] 🟢 Received frame for processing...")
    
    # Increment frame ID
    current_frame_id += 1
    frame_id = current_frame_id
    
    # Decode frame
    frame = decode_base64_frame(body)
    if frame is None:
        print("[Detector] ❌ Failed to decode frame")
        return
    
    # Process frame with stateless function, passing in and getting back state
    result, updated_state = process_frame(frame, frame_id, detection_state)
    
    # Update shared state
    detection_state = updated_state
    
    # Encode annotated frame
    frame_b64 = encode_frame(result["annotated_frame"])
    
    # Submit publishing to a separate thread pool to maintain throughput
    publish_executor.submit(publish_result, result, frame_b64, start_time)

# Dispatch to thread pool
def callback(ch, method, properties, body):
    executor.submit(handle_detection_task, body)

def run_detector():
    start_time = time.time()
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)
    
    # Set prefetch count to 1 to ensure we process one frame at a time
    # This helps maintain the correct order of frames and avoid overwhelming the detector
    channel.basic_qos(prefetch_count=1)
    
    channel.basic_consume(queue=RABBITMQ_QUEUE, on_message_callback=callback, auto_ack=True)
    print("[Detector] 🟢 Started consuming frames...")
    print(f"[Detector] 🕒 Initialization took {time.time() - start_time:.2f} seconds")
    print(f"[Detector] 💡 Using 1 worker for detection and {publish_executor._max_workers} workers for publishing")
    channel.start_consuming()

if __name__ == "__main__":
    try:
        run_detector()
    except KeyboardInterrupt:
        print("[Detector] ❌ Stopped by user.")
    except Exception as e:
        print(f"[Detector] ❌ Error: {e}")
    finally:
        # Shutdown thread pools gracefully
        executor.shutdown(wait=False)
        publish_executor.shutdown(wait=False)
        print("[Detector] ✅ Thread pools and connection closed.")