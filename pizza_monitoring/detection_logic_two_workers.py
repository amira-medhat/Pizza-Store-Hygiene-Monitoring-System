import cv2
from ultralytics import YOLO
import numpy as np
import time

# Paths
VIDEO_PATH = r"D:\PizzaStore_Task\Sah w b3dha ghalt (2).mp4"
MODEL_PATH = r"D:\PizzaStore_Task\best.pt"
OUTPUT_PATH = "output_Video_2nd.mp4"

# Load model with tracking enabled
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
messages = []

# ROI (over all containers)
SCOOPER_CONTAINERS = [
    (0, (480, 270, 525, 320)),
    (1, (460, 317, 515, 350)),
    (2, (460, 350, 510, 400)),
]
# SCOOPER_CONTAINERS = [
#     (0, (470, 270, 525, 310)),
#     (1, (460, 310, 520, 350)),
#     (2, (450, 355, 500, 390)),
# ]
# Event buffer
person_events = {}  # person_id -> list of events
fps_grace = int(fps * 0) # Allow 2 seconds grace period for safe pickups
last_safe_frame = {}  # person_id -> last safe frame
worker_stats = {}  # Track statistics per worker
worker_in_roi = {}  # Track which workers are currently in ROI

# Track worker positions to maintain consistent IDs
worker_positions = {}  # track_id -> last position (x, y)
worker_id_map = {}     # track_id -> consistent worker ID
next_worker_id = 1     # Counter for assigning consistent worker IDs


def is_inside(box, roi):
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

def boxes_overlap(box1, box2, iou_thresh=0.1):
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

def get_consistent_worker_id(track_id, position):
    """Map tracking IDs to consistent worker IDs based on position"""
    global next_worker_id
    
    # If we've seen this track_id before, return its mapped worker_id
    if track_id in worker_id_map:
        # Update position
        worker_positions[track_id] = position
        return worker_id_map[track_id]
    
    # Check if this position is close to an existing worker (re-identification)
    for existing_track_id, existing_pos in worker_positions.items():
        if existing_track_id in worker_id_map:
            dist = np.sqrt((position[0] - existing_pos[0])**2 + (position[1] - existing_pos[1])**2)
            if dist < 150:  # If within 150 pixels, consider it the same worker
                worker_id_map[track_id] = worker_id_map[existing_track_id]
                worker_positions[track_id] = position
                return worker_id_map[track_id]
    
    # New track_id, assign a new consistent worker ID
    worker_id_map[track_id] = next_worker_id
    worker_positions[track_id] = position
    next_worker_id += 1
    return worker_id_map[track_id]

def assign_hand_to_person(hand_box, person_boxes):
    """Assign a hand to the closest person"""
    hx1, hy1, hx2, hy2 = hand_box
    hcx = (hx1 + hx2) / 2
    hcy = (hy1 + hy2) / 2
    min_dist = float('inf')
    assigned_id = None
    
    # Maximum distance threshold for hand-person assignment
    max_dist_threshold = 300
    
    for person in person_boxes:
        if hasattr(person, 'id') and person.id is not None:
            px1, py1, px2, py2 = person.xyxy[0].cpu().numpy()
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2
            
            # Calculate Euclidean distance between hand and person centers
            dist = np.sqrt((hcx - pcx)**2 + (hcy - pcy)**2)
            
            # Check if hand is within reasonable distance of the person
            if dist < min_dist and dist < max_dist_threshold:
                min_dist = dist
                track_id = int(person.id.item())
                # Get consistent worker ID for this track_id
                assigned_id = get_consistent_worker_id(track_id, (pcx, pcy))
    
    return assigned_id

# === Frame Loop ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.0001, iou=0.3, verbose=False)[0]

    hands, scoopers, pizzas, persons = [], [], [], []

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
        elif label == 'person':
            persons.append(box)

    # Track which hands belong to which worker
    person_hands = {}
    
    # Process persons first to establish consistent IDs
    for person in persons:
        if hasattr(person, 'id') and person.id is not None:
            track_id = int(person.id.item())
            px1, py1, px2, py2 = person.xyxy[0].cpu().numpy()
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2
            # Get consistent worker ID
            get_consistent_worker_id(track_id, (pcx, pcy))
    
    # Assign hands to persons
    for hand in hands:
        worker_id = assign_hand_to_person(hand, persons)
        if worker_id is None:
            continue
            
        if worker_id not in person_hands:
            person_hands[worker_id] = []
        person_hands[worker_id].append(hand)
    
    # Track ROI entry/exit and record events on exit
    current_frame_in_roi = set()
    for worker_id, worker_hands in person_hands.items():
        worker_in_roi_this_frame = False
        for hand in worker_hands:
            for cid, roi in SCOOPER_CONTAINERS:
                if is_inside(hand, roi):
                    worker_in_roi_this_frame = True
                    current_frame_in_roi.add(worker_id)
                    # Start tracking if not already
                    if worker_id not in worker_in_roi:
                        worker_in_roi[worker_id] = {
                            "start_frame": frame_id,
                            "hand": hand,
                            "roi_id": cid,
                            "scooper_touched": False,
                            "pizza_touched": False
                        }
                    break
            if worker_in_roi_this_frame:
                break
    
    # Check for workers who exited ROI and record events
    for worker_id in list(worker_in_roi.keys()):
        if worker_id not in current_frame_in_roi:
            # Worker exited ROI - record the event only if no unprocessed events exist
            if worker_id not in person_events:
                person_events[worker_id] = []
            
            # Check if there's already an unprocessed event for this worker
            if not any(not e['processed'] for e in person_events[worker_id]):
                roi_data = worker_in_roi[worker_id]
                person_events[worker_id].append({
                    "start_frame": roi_data["start_frame"],
                    "end_frame": frame_id,
                    "hand": roi_data["hand"],
                    "roi_id": roi_data["roi_id"],
                    "scooper_touched": roi_data["scooper_touched"],
                    "pizza_touched": roi_data["pizza_touched"],
                    "processed": False,
                    "worker_id": worker_id
                })
            del worker_in_roi[worker_id]


    
    # Process completed events (when worker exited ROI)
    for pid, events in person_events.items():
        for event in events:
            if event['processed']:
                continue
            
            age_since_exit = frame_id - event['end_frame']
            if age_since_exit <= int(3 * fps):
                current_worker_hands = person_hands[worker_id]
                pizza_touched = any(
                    any(boxes_overlap(current_hand, pizza, iou_thresh=0.1) for pizza in pizzas)
                    for current_hand in current_worker_hands
                )
                if pizza_touched:
                    event['pizza_touched'] = True
                # Continue checking for scooper usage during grace period
                scooper_used = any(
                    any(boxes_overlap(pizza, scooper, iou_thresh=0.1) for scooper in scoopers)
                    for pizza in pizzas
                )
                if scooper_used:
                    event['scooper_touched'] = True
            else:
                
                video_time = event["end_frame"] / fps
                minutes = int(video_time // 60)
                seconds = int(video_time % 60)
                worker_id = event['worker_id']
                
                # Initialize worker stats if not exists
                if worker_id not in worker_stats:
                    worker_stats[worker_id] = {
                        "violations": 0,
                        "safe_pickups": 0
                    }
                
                # Only consider it a violation if worker touched pizza but didn't use a scooper
                if event['pizza_touched'] and not event['scooper_touched']:
                    violation_count += 1
                    worker_stats[worker_id]["violations"] += 1
                    msg = f"[{minutes:02d}:{seconds:02d}] Violation detected for Worker #{worker_id}!"
                    print(msg)
                    messages.append(msg)
                    hand_box = event['hand']
                    cv2.putText(frame, f"❌ Violation! W#{worker_id}", (int(hand_box[0]), int(hand_box[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.rectangle(frame, (int(hand_box[0]), int(hand_box[1])),
                                (int(hand_box[2]), int(hand_box[3])), (0, 0, 255), 3)
                elif event['pizza_touched'] and event['scooper_touched']:
                    worker_stats[worker_id]["safe_pickups"] += 1
                    msg = f"[{minutes:02d}:{seconds:02d}] Safe pickup by Worker #{worker_id}"
                    print(msg)
                    messages.append(msg)
                    last_safe_frame[worker_id] = frame_id
                event['processed'] = True

    for cid, roi in SCOOPER_CONTAINERS:
        cv2.rectangle(frame, roi[:2], roi[2:], (255, 255, 0), 2)
        cv2.putText(frame, f"C{cid}", (roi[0], roi[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    for box in hands:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(frame, "Hand", (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for box in pizzas:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 165, 255), 2)
        cv2.putText(frame, "Pizza", (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    for box in scoopers:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 2)
        cv2.putText(frame, "Scooper", (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    for person in persons:
        if hasattr(person, 'id') and person.id is not None:
            track_id = int(person.id.item())
            coords = person.xyxy[0].cpu().numpy()
            pcx = (coords[0] + coords[2]) / 2
            pcy = (coords[1] + coords[3]) / 2
            
            # Get consistent worker ID
            worker_id = get_consistent_worker_id(track_id, (pcx, pcy))
            
            cv2.rectangle(frame, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (255, 255, 255), 2)
            cv2.putText(frame, f"Worker #{worker_id}", (int(coords[0]), int(coords[1]) - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display overall violation count
    cv2.putText(frame, f"Total Violations: {violation_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Display per-worker statistics
    y_offset = 60
    for worker_id, stats in worker_stats.items():
        cv2.putText(frame, f"Worker #{worker_id}: {stats['violations']} violations, {stats['safe_pickups']} safe", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25

    # Display recent messages
    for i, msg in enumerate(messages[-1:]):
        cv2.putText(frame, msg, (10, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out.write(frame)

cap.release()
out.release()

# Print final statistics
print(f"[✅] Done. Total violations: {violation_count}, video saved to {OUTPUT_PATH}")
print("\nWorker statistics:")
for worker_id, stats in worker_stats.items():
    print(f"Worker #{worker_id}: {stats['violations']} violations, {stats['safe_pickups']} safe pickups")
