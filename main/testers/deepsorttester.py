import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- CONFIG ---
VIDEO_SOURCE = r"../recording8.mp4"  # Use 0 for webcam or replace with a file path like 'sample.mp4'
MODEL_PATH = r"../best.pt" # Or use your own YOLO model like 'best.pt'
CONF_THRESHOLD = 0.5

def main():
    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Initialize DeepSort
    tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.3)

    # Open video stream
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Error: Cannot open video stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=CONF_THRESHOLD)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_id))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.imshow("YOLO + DeepSORT", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
