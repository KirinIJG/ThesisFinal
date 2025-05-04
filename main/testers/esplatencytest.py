from flask import Flask, request
import cv2
import numpy as np
import time
from datetime import datetime
import os
import csv

app = Flask(__name__)
frames_dir = "./frames"
csv_file = "latency_log.csv"

# Ensure frames directory exists
os.makedirs(frames_dir, exist_ok=True)

# Create CSV file with header if it doesn't exist
if not os.path.isfile(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ESP_Timestamp_ms", "Jetson_Timestamp_ms", "Latency_ms", "Datetime"])

@app.route('/upload', methods=['POST'])
def upload_frame():
    try:
        # Get ESP32 timestamp
        esp_timestamp_ms = int(request.headers.get('X-Timestamp', '0'))
        jetson_timestamp_ms = int(time.time() * 1000)
        latency_ms = jetson_timestamp_ms - esp_timestamp_ms
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Decode image
        img_data = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

        # Save frame
        filename = f"frame_{esp_timestamp_ms}.jpg"
        cv2.imwrite(os.path.join(frames_dir, filename), frame)

        # Append to CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([esp_timestamp_ms, jetson_timestamp_ms, latency_ms, now])

        print(f"[{now}] Latency: {latency_ms} ms")
        return "OK", 200

    except Exception as e:
        print("Error:", e)
        return "Error", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
