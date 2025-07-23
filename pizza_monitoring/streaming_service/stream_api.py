from flask import Flask, Response, jsonify, render_template_string
from waitress import serve
import time
import cv2
import sqlite3
from config import DB_PATH
import state
from rabbit_consumer import start_consumer_thread

app = Flask(__name__)

# ───────────────────────
# Start RabbitMQ Consumer
# ───────────────────────
start_consumer_thread()

# ───────────────────────
# MJPEG Stream Generator
# ───────────────────────
def generate_frames():
    prev_time = time.time()
    last_encoded = None
    last_frame_time = 0
    frame_count = 0

    while True:
        start = time.time()
        frame = getattr(state, 'latest_frame', None)

        if frame is None:
            # Show last good frame if available (not older than 0.5s)
            if last_encoded and (time.time() - last_frame_time < 0.5):
                yield last_encoded
            time.sleep(0.01)
            continue

        # Resize if needed
        if frame.shape[1] > 640:
            frame = cv2.resize(frame, (640, 480))

        # Encode frame as JPEG
        success, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            time.sleep(0.005)
            continue

        encoded = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        last_encoded = encoded
        last_frame_time = time.time()

        # FPS and latency log
        frame_count += 1
        elapsed = time.time() - prev_time
        if elapsed > 0:
            fps = 1 / elapsed
        else:
            fps = 0
        latency = (time.time() - start) * 1000
        prev_time = time.time()
        if frame_count % 30 == 0:
            print(f"[Live Stream] FPS: {fps:.2f} | Latency: {latency:.2f} ms")

        yield encoded

        # Adaptive sleep for ~30 FPS
        process_time = time.time() - start
        time.sleep(max(0, (1 / 30) - process_time))


# ────────────────
# HTML Dashboard
# ────────────────
dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>🍕 Pizza Monitoring Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f8f9fa;
            color: #333;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .dashboard {
            display: flex;
            flex-direction: row;
            justify-content: center;
            align-items: flex-start;
            padding: 30px;
            gap: 40px;
        }
        .stats {
            background-color: #fff;
            padding: 20px 30px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            min-width: 240px;
        }
        .stats h1 {
            font-size: 1.5em;
            margin-bottom: 20px;
            text-align: left;
        }
        .stats p {
            font-size: 1.1em;
            margin: 12px 0;
            text-align: left;
        }
        .video {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        img {
            border-radius: 10px;
            border: 3px solid #ccc;
            max-width: 100%;
            height: auto;
        }
        @media (max-width: 1000px) {
            .dashboard {
                flex-direction: column;
                align-items: center;
            }
        }
    </style>
    <script>
        async function updateStats() {
            try {
                const res = await fetch('/summary');
                const data = await res.json();
                document.getElementById('safe').innerText = data.total_safe_pickups;
                document.getElementById('violations').innerText = data.total_violations;
            } catch (e) {
                console.error("Failed to load stats:", e);
            }
        }

        setInterval(updateStats, 1000);
        window.onload = updateStats;
    </script>
</head>
<body>
    <div class="dashboard">
        <div class="stats">
            <h1>🍕 Pizza Monitoring</h1>
            <p><strong>✅ Safe Pickups:</strong> <span id="safe">0</span></p>
            <p><strong>❌ Violations:</strong> <span id="violations">0</span></p>
        </div>
        <div class="video">
            <h2>📺 Live Stream</h2>
            <img src="/video" alt="Live Stream" style="width: 720px;">
        </div>
    </div>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(dashboard_html)


# ───────────────────────
# Summary API (Cached)
# ───────────────────────
summary_cache = {
    "violations": 0,
    "safe_pickups": 0,
    "last_updated": 0
}

@app.route("/summary")
def get_summary():
    try:
        now = time.time()
        if now - summary_cache["last_updated"] > 1:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM violations WHERE is_violation = 1")
            summary_cache["violations"] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM violations WHERE is_safe_pickup = 1")
            summary_cache["safe_pickups"] = cursor.fetchone()[0]
            conn.close()
            summary_cache["last_updated"] = now

        return jsonify({
            "total_violations": summary_cache["violations"],
            "total_safe_pickups": summary_cache["safe_pickups"]
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ───────────────────────
# Video Streaming Route
# ───────────────────────
@app.route("/video")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ───────────────────────
# Start Waitress Server
# ───────────────────────
if __name__ == "__main__":
    print("🚀 Running Flask server with Waitress on http://0.0.0.0:8000 ...")
    serve(app, host="0.0.0.0", port=8000)
