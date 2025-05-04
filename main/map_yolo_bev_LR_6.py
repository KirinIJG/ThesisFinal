import cv2
import numpy as np
import joblib
import os
import time
from config import FRAME_TIME

# Constants
BEV_WIDTH = 800
BEV_HEIGHT = 480
vehicle_w_m = 1.8
vehicle_h_m = 5.3
scale = 42.48
cam_offset = vehicle_h_m / 2 - 2.55

vehicle_w_px = int(vehicle_w_m * scale)
vehicle_h_px = int(vehicle_h_m * scale)
center_x = BEV_WIDTH // 2
center_y = BEV_HEIGHT // 2

SIDES = ['FRONT', 'BACK', 'LEFT', 'RIGHT']

# Load regression models
model_folder = "./models"
regression_models = {}
for side in SIDES:
    model_path = os.path.join(model_folder, f"{side}_pixel_to_topdown_model.pkl")
    if os.path.exists(model_path):
        regression_models[side] = joblib.load(model_path)
    else:
        print(f"[BEV] Missing regression model for side: {side}")

# Load 2D PNG icons for each class
icon_folder = "./icons"
class_icons = {}
host_vehicle_icon = None
for class_name in os.listdir(icon_folder):
    if class_name.endswith(".png"):
        key = class_name.replace(".png", "")
        icon_path = os.path.join(icon_folder, class_name)
        icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
        if icon is not None:
            if key == "host_vehicle":
                host_vehicle_icon = icon
            else:
                class_icons[key] = icon

def draw_background():
    bg = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
    colors = [(0, 255, 255), (0, 165, 255), (0, 0, 255)]
    for i in range(3, 0, -1):
        w = int((vehicle_w_m + 2 * i) * scale)
        h = int((vehicle_h_m + 2 * i) * scale)
        cv2.rectangle(bg, (center_x - w // 2, center_y - h // 2), (center_x + w // 2, center_y + h // 2), colors[3 - i], -1)
    resized_host = cv2.resize(host_vehicle_icon, (vehicle_w_px, vehicle_h_px))
    overlay_icon(bg, resized_host, center_x, center_y)
    #cv2.rectangle(bg, (center_x - vehicle_w_px // 2, center_y - vehicle_h_px // 2),
                  #(center_x + vehicle_w_px // 2, center_y + vehicle_h_px // 2), (50, 50, 50), -1)
    #cv2.rectangle(bg, (center_x - vehicle_w_px // 2, center_y - vehicle_h_px // 2),
                  #(center_x + vehicle_w_px // 2, center_y + vehicle_h_px // 2), (255, 255, 255), 2)
    return bg

def map_to_bev(hor_m, ver_m, side):
    if side == 'FRONT':
        px = center_x + hor_m * scale
        py = center_y - (ver_m + (vehicle_h_m / 2)) * scale
    elif side == 'BACK':
        px = center_x - hor_m * scale
        py = center_y + (ver_m + (vehicle_h_m / 2)) * scale
    elif side == 'LEFT':
        px = center_x - (ver_m + cam_offset + vehicle_w_m / 2) * scale
        py = center_y - hor_m * scale
    elif side == 'RIGHT':
        px = center_x + (ver_m + cam_offset + vehicle_w_m / 2) * scale
        py = center_y + hor_m * scale
    else:
        return None
    return int(px), int(py)

def overlay_icon(canvas, icon, px, py):
    ih, iw = icon.shape[:2]
    x1 = px - iw // 2
    y1 = py - ih // 2
    x2 = x1 + iw
    y2 = y1 + ih

    if x1 < 0 or y1 < 0 or x2 > canvas.shape[1] or y2 > canvas.shape[0]:
        return

    alpha = icon[:, :, 3] / 255.0
    for c in range(3):
        canvas[y1:y2, x1:x2, c] = (1 - alpha) * canvas[y1:y2, x1:x2, c] + alpha * icon[:, :, c]

def run_mapping_process(queues, shared_data):
    static_canvas = draw_background()
    persistent_canvas = static_canvas.copy()
    fade_timeout = 2.5

    while not shared_data["exit_flag"].value:
        persistent_canvas = cv2.addWeighted(persistent_canvas, 0.9, static_canvas, 0.1, 0)
        now = time.time()

        if "last_seen" in shared_data:
            expired = [tid for tid, ts in shared_data["last_seen"].items() if now - ts > fade_timeout]
            for tid in expired:
                shared_data["track_positions"].pop(tid, None)
                shared_data["last_speeds"].pop(tid, None)
                shared_data["last_seen"].pop(tid, None)

        for track_id, info in shared_data.get("track_positions", {}).items():
            try:
                cam_index = int(str(track_id).split('_')[0])
                side = SIDES[cam_index] if cam_index < len(SIDES) else "UNKNOWN"
                if side == "UNKNOWN" or side not in regression_models:
                    continue

                cx, cy = info["pos"]
                class_name = info.get("cls", "default")

                input_point = np.array([[cx, cy]], dtype=np.float32)
                hor_m, ver_m = regression_models[side].predict(input_point)[0]

                px, py = map_to_bev(hor_m, ver_m, side)
                if px is None or py is None:
                    continue

                icon = class_icons.get(class_name)
                if icon is not None:
                    resized_icon = cv2.resize(icon, (40, 40))
                    overlay_icon(persistent_canvas, resized_icon, px, py)

                speed_kph = shared_data["last_speeds"].get(track_id, 0.0)
                cv2.putText(persistent_canvas, f"{speed_kph:.1f} km/h", (px - 30, py - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            except Exception as e:
                print(f"[BEV ERROR] {track_id}: {e}")

        resized = cv2.resize(persistent_canvas, (800, 400))
        shared_data["bev_display_frame"] = resized

        time.sleep(FRAME_TIME)
