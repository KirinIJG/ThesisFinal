import cv2
import time
import os
import psutil
from model1 import load_model
from utils import draw_predictions, pixel_to_distance
from deep_sort_realtime.deepsort_tracker import DeepSort
import warnings
import numpy as np
from datetime import datetime
import csv
import obd

warnings.filterwarnings("ignore", category=UserWarning, module="numpy")


def letterbox_image(image, target_size=(320, 320), color=(0, 0, 0)):
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_x = target_width - new_width
    pad_y = target_height - new_height
    top = pad_y // 2
    bottom = pad_y - top
    left = pad_x // 2
    right = pad_x - left
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded_image


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def get_next_test_folder(base_path="logs"):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    existing = [d for d in os.listdir(base_path) if d.startswith("test") and os.path.isdir(os.path.join(base_path, d))]
    test_nums = [int(name[4:]) for name in existing if name[4:].isdigit()]
    next_num = max(test_nums, default=0) + 1
    new_folder = os.path.join(base_path, f"test{next_num}")
    os.makedirs(new_folder)
    return new_folder


def camera_process(camera_index, queue, shared_data, test_folder):
    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        print(f"Starting process for Camera {camera_index + 1}...")

        model = load_model()
        if model is None:
            raise RuntimeError("Failed to load YOLO model.")

        deepsort = DeepSort(max_age=30, n_init=3)
        prev_gray = None
        frame_count = 0
        skip_frames = 2

        
        log_dir = os.path.join(test_folder, f"camera{camera_index + 1}")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        processing_log_path = os.path.join(log_dir, "processing_log.csv")
        object_log_path = os.path.join(log_dir, "objects_log.csv")
        count_log_path = os.path.join(log_dir, "detection_count.csv")
        system_log_path = os.path.join(log_dir, "system_usage.csv")
        inference_log_path = os.path.join(log_dir, "inference_log.csv")

        with open(processing_log_path, "w", newline="") as p_log, \
            open(object_log_path, "w", newline="") as o_log, \
            open(count_log_path, "w", newline="") as c_log, \
            open(system_log_path, "w", newline="") as s_log, \
            open(inference_log_path, "w", newline="") as i_log:

            processing_writer = csv.writer(p_log)
            object_writer = csv.writer(o_log)
            count_writer = csv.writer(c_log)
            system_writer = csv.writer(s_log)
            inference_writer = csv.writer(i_log)

            processing_writer.writerow(["Timestamp_ms", "Frame", "ProcessingTime_s"])
            object_writer.writerow(["Timestamp_ms", "TrackID", "ClassID", "Speed_kph", "Distance_m", "HostSpeed_kph", "RelativeSpeed_kph"])
            count_writer.writerow(["Timestamp_ms", "Frame", "NumDetections"])
            system_writer.writerow(["Timestamp_ms", "CPU_percent", "Memory_percent"])
            inference_writer.writerow(["Timestamp_ms", "Frame", "YOLO_InferenceTime_s"])

            while True:
                if shared_data["exit_flag"].value:
                    print(f"[Camera {camera_index}] Kill switch activated. Exiting.")
                    break
                start_time = time.perf_counter()
                if not queue.empty():
                    frame = queue.get()
                    if frame is None:
                        continue

                    frame = letterbox_image(frame, target_size=(320, 320))
                    frame_count += 1
                    if frame_count % skip_frames != 0:
                        continue

                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                    try:
                        yolo_start = time.perf_counter()
                        results = model.predict(frame, conf=0.5)
                        yolo_end = time.perf_counter()
                        inference_time = yolo_end - yolo_start
                        inference_writer.writerow([timestamp_now, frame_count, f"{inference_time:.4f}"])

                        deepsort_detections = []
                        detection_boxes = []
                        total_detections = 0

                        for r in results:
                            boxes = r.boxes
                            total_detections += len(boxes)
                            for box in boxes:
                                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].cpu().numpy())
                                confidence = float(box.conf[0].cpu().numpy())
                                class_id = int(box.cls[0].cpu().numpy())
                                w = x_max - x_min
                                h = y_max - y_min
                                deepsort_detections.append(([x_min, y_min, w, h], confidence, class_id))
                                detection_boxes.append((x_min, y_min, x_max, y_max, confidence, class_id))

                        count_writer.writerow([timestamp_now, frame_count, total_detections])

                        tracks = deepsort.update_tracks(deepsort_detections, frame=frame)
                        matched_tracks = {}
                        drawing_detections = []

                        for track in tracks:
                            if not track.is_confirmed():
                                continue

                            track_id = f"{camera_index}_{track.track_id}"
                            best_iou = 0
                            best_box = None
                            for det in detection_boxes:
                                iou = compute_iou(track.to_ltrb(), det[:4])
                                if iou > best_iou:
                                    best_iou = iou
                                    best_box = det

                            if best_box:
                                x_min, y_min, x_max, y_max, conf, cls_id = best_box
                            else:
                                x_min, y_min, x_max, y_max = map(int, track.to_ltrb())
                                conf, cls_id = 1.0, 0

                            drawing_detections.append([x_min, y_min, x_max, y_max, conf, cls_id])
                            matched_tracks[len(matched_tracks)] = track_id

                            if "last_seen" not in shared_data:
                                shared_data["last_seen"] = {}
                            shared_data["last_seen"][track_id] = time.time()

                            cx = (x_min + x_max) // 2
                            cy = y_min
                            class_name = model.names[cls_id]
                            shared_data["track_positions"][track_id] = {
                                "pos": (cx, cy),
                                "cls": class_name
                            }

                            speed = shared_data["last_speeds"].get(track_id, 0.0)
                            _, vertical_distance = pixel_to_distance(cx, cy)
                            distance_m = vertical_distance / 100.0
                            host_speed_kph = shared_data.get("host_speed", 0.0)
                            relative_speed_kph = shared_data["last_speeds"].get(track_id, 0.0)
                            object_writer.writerow([
    timestamp_now, track_id, cls_id,
    f"{speed:.2f}", f"{distance_m:.2f}",
    f"{host_speed_kph:.2f}", f"{relative_speed_kph:.2f}"
])

                    except Exception as e:
                        print(f"[Camera {camera_index + 1}] Error during processing: {e}")

                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time

                    cpu = psutil.cpu_percent()
                    mem = psutil.virtual_memory().percent
                    system_writer.writerow([timestamp_now, cpu, mem])
                    processing_writer.writerow([timestamp_now, frame_count, f"{elapsed_time:.4f}"])

                    active_ids = set(matched_tracks.values())
                    existing_ids = set(shared_data["track_positions"].keys())
                    stale_ids = [tid for tid in existing_ids if tid.startswith(f"{camera_index}_") and tid not in active_ids]
                    for tid in stale_ids:
                        shared_data["track_positions"].pop(tid, None)
                        shared_data["last_speeds"].pop(tid, None)
                        shared_data["last_seen"].pop(tid, None)

                    prev_gray = draw_predictions(
                        frame, drawing_detections, gray_frame, prev_gray,
                        shared_data["track_positions"], {},
                        matched_tracks, shared_data["last_speeds"],
                        frame_count, camera_index, shared_data.get("host_speed", 0.0)
                    )

                    #cv2.namedWindow(f"Camera {camera_index + 1}", cv2.WINDOW_NORMAL)
                    #cv2.resizeWindow(f"Camera {camera_index + 1}", 350, 240)
                    shared_data["camera_frames"][camera_index] = frame

                    #if cv2.waitKey(1) & 0xFF == ord('q'):
                       # print(f"Stopping process for Camera {camera_index + 1}...")
                        #break

        #cv2.destroyAllWindows()

    finally:
        profiler.disable()
        profile_output = os.path.join(log_dir, "cprofile_stats.prof")
        profiler.dump_stats(profile_output)
        print(f"Profiler saved: {profile_output}")
