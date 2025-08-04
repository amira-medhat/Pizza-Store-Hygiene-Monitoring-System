import sys
import os
# Add root directory to sys.path (pizza_monitoring)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pika
import json
import base64
import cv2
import numpy as np
from threading import Thread
import queue
import time
import state
from shared.config import PROCESSED_QUEUE

# Frame queue
frame_queue = queue.Queue(maxsize=60)

# ─────────────────────────────────────────────
# Decode frame from Base64
# ─────────────────────────────────────────────
def decode_base64_frame(b64_str):
    try:
        start = time.time()
        byte_data = base64.b64decode(b64_str)
        np_array = np.frombuffer(byte_data, dtype=np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        end = time.time()
        print(f"[Stream Consumer] 🕒 Decoded frame in {end - start:.4f} seconds.")
        return frame
    except Exception as e:
        print(f"[Stream Consumer] ❌ Decode error: {e}")
        return None

# ─────────────────────────────────────────────
# Frame processor thread
# ─────────────────────────────────────────────
def process_frame_worker(worker_id=0):
    while True:
        try:
            frame_data = frame_queue.get(timeout=5)
            start = time.time()

            frame = decode_base64_frame(frame_data)
            if frame is not None:
                state.latest_frame = frame

            print(f"[Worker {worker_id}] 🕒 Processed in {time.time() - start:.4f} sec")
            frame_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Worker {worker_id}] ❌ Processing error: {e}")

# ─────────────────────────────────────────────
# RabbitMQ Consumer
# ─────────────────────────────────────────────
def consume_frames():
    def callback(ch, method, properties, body):
        try:
            data = json.loads(body)
            b64_frame = data.get("frame")

            if b64_frame:
                try:
                    frame_queue.put_nowait(b64_frame)
                except queue.Full:
                    print("[Stream Consumer] ⚠️ Frame queue full, dropping frame")

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            print(f"[Stream Consumer] ❌ Callback error: {e}")
            ch.basic_ack(delivery_tag=method.delivery_tag)

    try:
        connection_params = pika.ConnectionParameters(
            host=os.environ.get("RABBITMQ_HOST", "localhost"),
            heartbeat=600,
            blocked_connection_timeout=300,
            socket_timeout=10.0
        )
        connection = pika.BlockingConnection(connection_params)
        channel = connection.channel()

        channel.queue_declare(queue=PROCESSED_QUEUE, durable=False)
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=PROCESSED_QUEUE, on_message_callback=callback, auto_ack=False)

        print("[Stream Consumer] 🟢 Connected and consuming...")
        channel.start_consuming()

    except pika.exceptions.AMQPConnectionError as e:
        print(f"[Stream Consumer] ❌ RabbitMQ connection failed: {e}")
    except Exception as e:
        print(f"[Stream Consumer] ❌ Unexpected error: {e}")

# ─────────────────────────────────────────────
# Start Threads (Consumers + Workers)
# ─────────────────────────────────────────────
def start_consumer_thread(worker_count=3):
    # Start worker threads for decoding
    for i in range(worker_count):
        Thread(target=process_frame_worker, args=(i,), daemon=True).start()

    # Start the consumer
    Thread(target=consume_frames, daemon=True).start()
