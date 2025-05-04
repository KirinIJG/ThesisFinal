
import numpy as np
import cv2
import collections
from config import FRAME_TIME, MIN_DISTANCE, FPS
from scipy.spatial.distance import cdist
import logging
import warnings
import math
import joblib
import obd

#warnings.filterwarnings("ignore", category="UserWarning", module="numpy")

connection = obd.OBD()
# Speed tracking history
speed_history = collections.defaultdict(lambda: collections.deque(maxlen=5))

# Setup logger
logger = logging.getLogger("utils")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load trained models for pixel-to-distance conversion
HORIZONTAL_MODEL_PATH = "./horizontal_model.joblib"
VERTICAL_MODEL_PATH = "./vertical_model.joblib"
horizontal_model = joblib.load(HORIZONTAL_MODEL_PATH)
vertical_model = joblib.load(VERTICAL_MODEL_PATH)

logger.info("Trained models for distance estimation loaded successfully.")

def pixel_to_distance(x, y):
    """
    Convert pixel coordinates to real-world distances using the trained models.
    """
    pixel_coords = np.array([[x, y]])
    horizontal_distance = horizontal_model.predict(pixel_coords)[0]
    vertical_distance = vertical_model.predict(pixel_coords)[0]
    vertical_distance = max(vertical_distance, 0)  # Ensure vertical distance is non-negative
    return horizontal_distance, vertical_distance

def calculate_speed(pos1, pos2, camera_index):
    """
    Calculate speed (m/s) based on real-world distance and frame time.
    """
    # Convert pixel coordinates to real-world distances
    horizontal1, vertical1 = pixel_to_distance(pos1[0], pos1[1])
    horizontal2, vertical2 = pixel_to_distance(pos2[0], pos2[1])

    # Calculate real-world distance
    real_distance = np.linalg.norm([horizontal2 - horizontal1, vertical2 - vertical1])  # In cm

    # Convert distance to meters and calculate speed
    distance_in_meters = real_distance / 100.0  # Convert cm to meters
    if distance_in_meters < MIN_DISTANCE:
       return 0.0
    return distance_in_meters / FRAME_TIME

def calculate_smoothed_speed(track_id, new_speed):
    """
    Calculate smoothed speed using a moving average.
    """
    speed_history[track_id].append(new_speed)
    return np.mean(speed_history[track_id])

def direction_from_displacement(dx, dy, camera_index, angle):
    """
    Determine the motion direction based on displacement.
    """
    if dx < 0.1 and dy < 0.1:
        return "Stationary"
    else:
        dx = math.floor(dx)
        dy = math.floor(dy)
        if camera_index == 3:  # for front camera
            if 25 <= abs(angle) <= 90:
                return "Forward" if dy < 0 else "Backward"
            else:  # Horizontal motion dominates
                return "Rightward" if dx < 0 else "Leftward"

        elif camera_index == 0:  # for back camera
            if 25 <= abs(angle) <= 90:
                return "Forward" if dy > 0 else "Backward"
            else:  # Horizontal motion dominates
                return "Rightward" if dx > 0 else "Leftward"

        elif camera_index == 2:  # for left camera
            if 0 <= abs(angle) <= 45:
                return "Forward" if dx > 0 else "Backward"
            else:  # Horizontal motion dominates
                return "Rightward" if dy < 0 else "Leftward"

def calculate_optical_flow_cuda(prev_gray, gray_frame, points):
    """
    Calculate optical flow using CUDA-accelerated SparsePyrLK.
    """
    if prev_gray is None or prev_gray.size == 0:
        raise ValueError("Invalid input: prev_gray is None or empty.")
    if gray_frame is None or gray_frame.size == 0:
        raise ValueError("Invalid input: gray_frame is None or empty.")
    if points is None or len(points) == 0:
        raise ValueError("Invalid input: points array is None or empty.")

    try:
        gpu_flow = cv2.cuda.SparsePyrLKOpticalFlow.create(winSize=(15, 15), maxLevel=3)
        prev_gray_gpu = cv2.cuda_GpuMat()
        gray_frame_gpu = cv2.cuda_GpuMat()
        prev_gray_gpu.upload(prev_gray)
        gray_frame_gpu.upload(gray_frame)

        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        points_gpu = cv2.cuda_GpuMat()
        points_gpu.upload(points)

        points_new_gpu, status_gpu, _ = gpu_flow.calc(prev_gray_gpu, gray_frame_gpu, points_gpu, None)

        points_new = points_new_gpu.download()
        status = status_gpu.download()

        return points_new, status
    except Exception as e:
        logger.error(f"Optical flow calculation failed: {e}")
        return points, np.zeros_like(points, dtype=np.uint8)

def match_tracks(detections, object_positions, distance_threshold):
    """
    Match detections to existing object positions or assign new track IDs.
    """
    detection_centers = np.array(
        [[d[0] + (d[2] - d[0]) / 2, d[1] + (d[3] - d[1]) / 2] for d in detections],
        dtype=np.float32,
    )
    prev_centers = []
    for pos in object_positions.values():
        if len(pos) > 0:
            last_position = pos[-1]
            if isinstance(last_position, (list, np.ndarray)) and len(last_position) == 2:
                prev_centers.append(last_position)

    prev_centers = np.array(prev_centers, dtype=np.float32)
    if detection_centers.ndim == 1:
        detection_centers = detection_centers.reshape(-1, 2)
    if prev_centers.size == 0:
        prev_centers = np.zeros((0, 2), dtype=np.float32)

    matches = {}
    if len(prev_centers) > 0:
        distances = cdist(detection_centers, prev_centers)
        for det_idx, _ in enumerate(detection_centers):
            closest_idx = np.argmin(distances[det_idx])
            closest_distance = distances[det_idx, closest_idx]

            if closest_distance < distance_threshold:
                prev_track_id = list(object_positions.keys())[closest_idx]
                matches[det_idx] = prev_track_id
            else:
                matches[det_idx] = f"new_object_{det_idx}"
    else:
        matches = {det_idx: f"new_object_{det_idx}" for det_idx in range(len(detections))}

    return matches


def draw_predictions(frame, detections, gray_frame, prev_gray, object_positions, object_tracking_count, matches, last_speeds, frame_count, camera_index):
    """
    Draw bounding boxes, labels, speed estimates, and directions with stable track IDs.
    """
    #print(f"Frame count: {frame_count}")
    #print(f"Detections: {detections}")
    #print(f"Object positions before processing:")
    for key, pos in object_positions.items():
        print(f"Track ID: {key}, Positions: {pos}, Shape: {np.array(pos).shape if isinstance(pos, np.ndarray) else 'Not an array'}")
    for track_id, position in object_positions.items():
        print(f"Track {track_id}: {np.array(position).shape}")
        
    #prev_direction = None
    
    for det_idx, detection in enumerate(detections):
        x_min, y_min, x_max, y_max, confidence, class_id = map(int, detection)
        center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
        current_position = np.array([center_x, center_y], dtype=np.float32)

        # Assign track ID from matches or create a new one
        track_id = matches.get(det_idx, f"object_{class_id}_{x_min}_{y_min}")

        try:
            # Optical flow tracking
            if prev_gray is not None and track_id in object_positions:
                prev_center = np.array(object_positions[track_id], dtype=np.float32).reshape(-1, 1, 2)
                points_new, status = calculate_optical_flow_cuda(prev_gray, gray_frame, prev_center)

                if status is not None and np.any(status == 1):
                    current_position = points_new[0].flatten()

                displacement = current_position - prev_center[0].flatten()
                dx, dy = displacement
                angle = math.degrees(math.atan2(dy, dx))
                direction = direction_from_displacement(dx, dy, camera_index, angle)

                cv2.putText(frame, f"Direction: {direction}", (center_x, center_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            # Speed calculation (CALIBRATE)
            if frame_count % (FPS // 30) == 0 or frame_count == 1:  # Every 1/3 second
                if track_id in object_positions:
                    speed_mps = calculate_speed(object_positions[track_id], current_position, camera_index)
                    smoothed_speed_kmh = calculate_smoothed_speed(track_id, speed_mps * 3.6)
                    response = connection.query(obd.commands.SPEED)
                    response = response.value.to('kph')
                    smoothed_speed_kmh = smoothed_speed_kph - response
                    smoothed_speed_kph = abs(smoothed_speed_kph)
                    last_speeds[track_id] = smoothed_speed_kmh
                else:
                    last_speeds[track_id] = 0.0

            # Display speed
            if track_id in last_speeds:
                speed_text = f"Speed: {last_speeds[track_id]:.2f} km/h"
                cv2.putText(frame, speed_text, (center_x, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # Update object positions
            object_positions[track_id] = current_position

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        except Exception as e:
            logger.error(f"Error drawing predictions for track ID {track_id}: {e}")

    return gray_frame
