import sys
import os
# Add root directory to sys.path (pizza_monitoring)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
from ultralytics import YOLO
import numpy as np
import time
from database import save_violation
from datetime import datetime
from shared.config import DB_PATH, MODEL_PATH, VIDEO_SOURCE
# Load model with tracking enabled
model = YOLO(MODEL_PATH)
print(model.names)
model.to('cuda')
model.model.half()  # Use FP16 for faster inference

# Video capture setup to get FPS
cap = cv2.VideoCapture(VIDEO_SOURCE)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
print(f"Video FPS: {fps}")

# ROI (over all containers)
SCOOPER_CONTAINERS = [
    (0, (480, 270, 525, 320)),
    (1, (460, 317, 515, 350)),
    (2, (460, 350, 510, 400)),
]

# Tracking state - kept outside process_frame but passed in/out as needed
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

def get_consistent_worker_id(track_id, position, worker_id_map, worker_positions, next_worker_id):
    """Map tracking IDs to consistent worker IDs based on position"""
    # If we've seen this track_id before, return its mapped worker_id
    if track_id in worker_id_map:
        # Update position
        worker_positions[track_id] = position
        return worker_id_map[track_id], worker_id_map, worker_positions, next_worker_id
    
    # New track_id, assign a new consistent worker ID
    worker_id_map[track_id] = next_worker_id
    worker_positions[track_id] = position
    next_worker_id += 1
    return worker_id_map[track_id], worker_id_map, worker_positions, next_worker_id

def assign_hand_to_person(hand_box, person_boxes, worker_id_map, worker_positions, next_worker_id):
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
                assigned_id, worker_id_map, worker_positions, next_worker_id = get_consistent_worker_id(
                    track_id, (pcx, pcy), worker_id_map, worker_positions, next_worker_id)
    
    return assigned_id, worker_id_map, worker_positions, next_worker_id

def process_frame(frame, frame_id, state=None):
    """
    Process a single frame with stateless logic
    
    Args:
        frame: The video frame to process
        frame_id: Current frame ID
        state: Dictionary containing persistent state (tracking info, etc.)
    
    Returns:
        result: Dictionary with detection results
        state: Updated state dictionary
    """
    start_time = time.time()
    
    # Initialize state if not provided
    if state is None:
        state = {
            'worker_id_map': {},
            'worker_positions': {},
            'next_worker_id': 1,
            'person_events': {},
            'last_safe_frame': {},
            'worker_stats': {},
            'processed_violations': set(),
            'violation_count': 0,
            'messages': []
        }
    
    # Extract state variables
    worker_id_map = state['worker_id_map']
    worker_positions = state['worker_positions']
    next_worker_id = state['next_worker_id']
    person_events = state['person_events']
    last_safe_frame = state['last_safe_frame']
    worker_stats = state['worker_stats']
    processed_violations = state['processed_violations']
    violation_count = state['violation_count']
    messages = state['messages']
    
    # Initialize frame-specific variables
    is_violation = False
    is_safe_pickup = False
    
    # Run object detection
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.01, iou=0.3, verbose=False)[0]

    hands, scoopers, pizzas, persons, labels_in_frame, boxes_in_frame = [], [], [], [], [], []

    # Process detection results
    for box in results.boxes:
        cls_id = int(box.cls.cpu().numpy().squeeze())
        label = model.names[cls_id]
        coords = box.xyxy.cpu().numpy().squeeze().tolist()

        labels_in_frame.append(label)
        boxes_in_frame.append(coords)

        if label == 'hand':
            hands.append(coords)
        elif label == 'scooper':
            scoopers.append(coords)
        elif label == 'pizza':
            pizzas.append(coords)
        elif label == 'person':
            persons.append(box)

    # Create timestamp in ISO format
    timestamp = datetime.now().isoformat()

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
            _, worker_id_map, worker_positions, next_worker_id = get_consistent_worker_id(
                track_id, (pcx, pcy), worker_id_map, worker_positions, next_worker_id)
    
    # Assign hands to persons
    for hand in hands:
        worker_id, worker_id_map, worker_positions, next_worker_id = assign_hand_to_person(
            hand, persons, worker_id_map, worker_positions, next_worker_id)
        if worker_id is None:
            continue
            
        if worker_id not in person_hands:
            person_hands[worker_id] = []
        person_hands[worker_id].append(hand)
    
    # Process each worker's hands
    for worker_id, worker_hands in person_hands.items():
        for hand in worker_hands:
            for cid, roi in SCOOPER_CONTAINERS:
                if is_inside(hand, roi):
                    if worker_id not in person_events:
                        person_events[worker_id] = []
                    if worker_id in last_safe_frame and frame_id - last_safe_frame[worker_id] <= int(fps * 3):
                        break
                    if any(not e['processed'] for e in person_events[worker_id]):
                        break
                    person_events[worker_id].append({
                        "start_frame": frame_id,
                        "hand": hand,
                        "roi_id": cid,
                        "scooper_touched": False,
                        "pizza_touched": False,
                        "processed": False,
                        "worker_id": worker_id
                    })
                    break

    # Associate scoopers with workers based on proximity
    worker_scoopers = {}
    for scooper in scoopers:
        worker_id, worker_id_map, worker_positions, next_worker_id = assign_hand_to_person(
            scooper, persons, worker_id_map, worker_positions, next_worker_id)
        if worker_id is not None:
            if worker_id not in worker_scoopers:
                worker_scoopers[worker_id] = []
            worker_scoopers[worker_id].append(scooper)
    
    # Process events for each worker
    for pid, events in person_events.items():
        for event in events:
            if event['processed']:
                continue
            age = frame_id - event['start_frame']
            if age <= 50:
                # Check if this worker is using a scooper
                worker_id = event['worker_id']
                hand = event['hand']
                
                # Check if any of this worker's current hands are touching a pizza
                if worker_id in person_hands:
                    current_worker_hands = person_hands[worker_id]
                    pizza_touched = any(
                        any(boxes_overlap(current_hand, pizza, iou_thresh=0.0001) for pizza in pizzas)
                        for current_hand in current_worker_hands
                    )
                    if pizza_touched:
                        event['pizza_touched'] = True
                
                # Check if worker has scoopers assigned
                if worker_id in worker_scoopers and worker_scoopers[worker_id]:
                    event['scooper_touched'] = True
                
                # Fallback to checking pizza-scooper overlap
                else:
                    scooper_used = any(
                        any(boxes_overlap(pizza, scooper, iou_thresh=0.0001) for scooper in scoopers)
                        for pizza in pizzas
                    )
                    if scooper_used:
                        event['scooper_touched'] = True
            else:
                video_time = event["start_frame"] / fps
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
                violation_id = f"{worker_id}_{event['start_frame']}"
                
                if event['pizza_touched'] and not event['scooper_touched'] and violation_id not in processed_violations:
                    violation_count += 1
                    worker_stats[worker_id]["violations"] += 1
                    msg = f"[{minutes:02d}:{seconds:02d}] Violation detected for Worker #{worker_id}!"
                    print(f"\n[VIOLATION DETECTED] {msg}\n")
                    messages.append(msg)
                    is_violation = True
                    processed_violations.add(violation_id)  # Mark this violation as processed

                elif event['pizza_touched'] and event['scooper_touched']:
                    # Create a unique safe pickup ID
                    safe_pickup_id = f"{worker_id}_{event['start_frame']}"
                    
                    if safe_pickup_id not in processed_violations:
                        worker_stats[worker_id]["safe_pickups"] += 1
                        is_safe_pickup = True
                        processed_violations.add(safe_pickup_id)  # Mark this safe pickup as processed
                        msg = f"[{minutes:02d}:{seconds:02d}] Safe pickup by Worker #{worker_id}"
                        print(f"\n[SAFE PICKUP] {msg}\n")
                        messages.append(msg)
                        last_safe_frame[worker_id] = frame_id
                
                event['processed'] = True

    # Save to database with status only if there's an actual event
    db_path = DB_PATH
    
    # Check if we have a violation or safe pickup to record
    if is_violation or is_safe_pickup:
        # Create a unique event ID for this frame to prevent duplicate database entries
        event_id = f"frame_{frame_id}"
        
        # Only save if we haven't already saved this exact frame event
        if event_id not in processed_violations:
            save_violation(timestamp, "", labels_in_frame, boxes_in_frame, 
                          is_violation, is_safe_pickup, db_path)
            processed_violations.add(event_id)  # Mark this frame as processed
    
    # Draw bounding boxes on the frame
    # Draw ROI containers
    for cid, roi in SCOOPER_CONTAINERS:
        cv2.rectangle(frame, roi[:2], roi[2:], (255, 255, 0), 2)
        cv2.putText(frame, f"C{cid}", (roi[0], roi[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Draw hands
    for box in hands:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(frame, "Hand", (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw pizzas
    for box in pizzas:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 165, 255), 2)
        cv2.putText(frame, "Pizza", (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    # Draw scoopers
    for box in scoopers:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 2)
        cv2.putText(frame, "Scooper", (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Draw persons
    for person in persons:
        if hasattr(person, 'id') and person.id is not None:
            track_id = int(person.id.item())
            coords = person.xyxy[0].cpu().numpy()
            pcx = (coords[0] + coords[2]) / 2
            pcy = (coords[1] + coords[3]) / 2
            
            # Get consistent worker ID
            worker_id = worker_id_map.get(track_id, 0)
            
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
    
    # Add violation indicators
    if is_violation:
        cv2.rectangle(frame, (10, 10), (frame.shape[1]-10, frame.shape[0]-10), (0, 0, 255), 5)
    
    # Update state
    updated_state = {
        'worker_id_map': worker_id_map,
        'worker_positions': worker_positions,
        'next_worker_id': next_worker_id,
        'person_events': person_events,
        'last_safe_frame': last_safe_frame,
        'worker_stats': worker_stats,
        'processed_violations': processed_violations,
        'violation_count': violation_count,
        'messages': messages
    }
    
    # Return result
    return {
        "timestamp": timestamp,
        "frame_id": frame_id,
        "is_violation": is_violation,
        "is_safe_pickup": is_safe_pickup,
        "labels": labels_in_frame,
        "boxes": boxes_in_frame,
        "annotated_frame": frame,
        "processing_time": time.time() - start_time
    }, updated_state