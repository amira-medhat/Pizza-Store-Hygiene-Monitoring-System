import cv2
from ultralytics import YOLO
import numpy as np
import time

# Paths
VIDEO_PATH = "Sah w b3dha ghalt (2).mp4"
MODEL_PATH = "best.pt"
OUTPUT_PATH = "output_violation_logic.mp4"

# Load model
model = YOLO(MODEL_PATH)
print(model.names)

# Video capture & writer setup
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")
width, height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_id = 0
violation_count = 0
messages = []  # Store messages to display on video

# ROI (over all containers)
SCOOPER_CONTAINERS = [
    (0, (480, 270, 525, 320)),
    (1, (460, 317, 515, 350)),
    (2, (460, 350, 510, 400)),
]

# Event buffer
potential_pickups = []  # List of dicts with: start_frame, hand, scooper_touched, processed
grace_duration = int(fps * 2)  # 2 seconds
last_safe_frame = -999  # initialize globally
violation_hands = []  # Track hands in violation for persistent marking


# ------------------ Helper functions ------------------

def is_inside(box, roi):
    """Check if center of box is inside ROI."""
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

def boxes_overlap(box1, box2, iou_thresh=0.1):
    """Check if two boxes overlap using IoU."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou > iou_thresh

def match_tracked_hand(event_hand, current_hands, iou_thresh=0.3):
    for hand in current_hands:
        if boxes_overlap(event_hand, hand, iou_thresh):
            return hand
    return None

# === Frame Loop ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    results = model.predict(frame, conf=0.003, iou=0.4, verbose=False)[0]
    hands, scoopers, pizzas = [], [], []

    # Extract detections
    for box in results.boxes:
        cls_id = int(box.cls.cpu().numpy().squeeze())
        label = model.names[cls_id]
        coords = box.xyxy.cpu().numpy().squeeze()

        if label == 'hand':
            hands.append(coords)
        elif label == 'scooper':
            scoopers.append(coords)
        elif label == 'pizza':
            pizzas.append(coords)

    # === Step 1: Detect hand entering ROI only if no active event ===
    active_event = any(not e["processed"] for e in potential_pickups)

    if not active_event and (frame_id - last_safe_frame > grace_duration):
        for hand in hands:
            for cid, roi in SCOOPER_CONTAINERS:
                if is_inside(hand, roi):
                    video_time = frame_id / fps
                    minutes = int(video_time // 60)
                    seconds = int(video_time % 60)
                    msg = f"[{minutes:02d}:{seconds:02d}] Hand detected in container ROI: C{cid}"
                    print(msg)
                    messages.append(msg)

                    potential_pickups.append({
                        "start_frame": frame_id,
                        "hand": hand,
                        "scooper_touched": False,
                        "processed": False
                    })

            if not active_event and (frame_id - last_safe_frame > grace_duration):
                break  # Exit outer loop after one detection

    # === Step 2: Evaluate pickup after 40 frames
    for event in potential_pickups:
        if event["processed"]:
            continue

        age = frame_id - event["start_frame"]
        if age <= 40:
            scooper_used = any(
                any(boxes_overlap(pizza, scooper, iou_thresh=0.0001) for scooper in scoopers)
                for pizza in pizzas
            )
            if scooper_used:
                event["scooper_touched"] = True
        else:
            # Evaluate and mark processed
            video_time = event["start_frame"] / fps
            minutes = int(video_time // 60)
            seconds = int(video_time % 60)

            if not event["scooper_touched"]:
                msg = f"[{minutes:02d}:{seconds:02d}] Violation detected!"
                print(msg)
                messages.append(msg)
                violation_count += 1

                # Draw violation box
                violating_hand = event.get("hand")
                if violating_hand is not None:
                    cv2.putText(frame, "❌ Violation!",
                                (int(violating_hand[0]), int(violating_hand[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.rectangle(frame,
                                (int(violating_hand[0]), int(violating_hand[1])),
                                (int(violating_hand[2]), int(violating_hand[3])),
                                (0, 0, 255), 3)
            else:
                msg = f"[{minutes:02d}:{seconds:02d}] Safe pickup with scooper"
                print(msg)
                messages.append(msg)
                last_safe_frame = frame_id  # mark safe time

            event["processed"] = True

    # === Step 3: Clean up processed events
    potential_pickups = [e for e in potential_pickups if not e["processed"]]

    # === Draw annotations ===
    for cid, roi in SCOOPER_CONTAINERS:
        cv2.rectangle(frame, roi[:2], roi[2:], (255, 255, 0), 2)
        cv2.putText(frame, f"C{cid}", (roi[0], roi[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    for box in hands:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(frame, "Hand", (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for box in pizzas:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 165, 255), 2)
        cv2.putText(frame, "Pizza", (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    for box in scoopers:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 2)
        cv2.putText(frame, "Scooper", (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    cv2.putText(frame, f"Violations: {violation_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Show most recent message
    for i, msg in enumerate(messages[-1:]):
        cv2.putText(frame, msg, (10, 70 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out.write(frame)

# === Cleanup ===
cap.release()
out.release()
print(f"[✅] Done. Violations: {violation_count}, video saved to {OUTPUT_PATH}")
