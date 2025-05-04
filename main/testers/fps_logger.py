import cv2
import time
import csv
import os

# ESP32 Stream URL
ESP32_IP = '192.168.0.113'  # <-- your ESP32 IP

# GStreamer pipeline string for MJPEG
gst_pipeline = (
    f'souphttpsrc location=http://{ESP32_IP}:81/stream ! '
    f'multipartdemux ! '
    f'jpegdec ! '
    f'videoconvert ! '
    f'appsink'
)

# Output folder and file
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)
log_file = os.path.join(log_folder, "frame_log_gstreamer.csv")

# OpenCV VideoCapture using GStreamer
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("[ERROR] Cannot open ESP32 stream via GStreamer!")
    exit()

print("[INFO] Connected to ESP32 Stream (GStreamer mode).")

# Initialize FPS tracking
frame_count = 0
start_time = time.time()
LOG_INTERVAL = 5  # seconds

# Open CSV for logging
with open(log_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Timestamp", "FPS"])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame.")
            continue

        frame_count += 1

        now = time.time()

        # Log every LOG_INTERVAL seconds
        if now - start_time >= LOG_INTERVAL:
            fps = frame_count / (now - start_time)
            timestamp_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))

            print(f"[{timestamp_now}] FPS: {fps:.2f}")
            writer.writerow([timestamp_now, f"{fps:.2f}"])
            csvfile.flush()

            frame_count = 0
            start_time = now

        # Display the frame
        cv2.imshow("ESP32 Stream (GStreamer)", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Logging completed. File saved at: {log_file}")
