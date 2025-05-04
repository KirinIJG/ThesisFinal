import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# List of camera IDs to monitor
CAMERA_IDS = [1,2,3,4]  # Adjust based on your setup

# Set up plots
num_cams = len(CAMERA_IDS)
fig, axs = plt.subplots(num_cams, 4, figsize=(16, 2.5 * num_cams))
fig.suptitle("Real-Time Performance Monitor - Multi-Camera", fontsize=16)

# Ensure axs is always 2D for consistent indexing
if num_cams == 1:
    axs = [axs]

# Column titles using figure-level annotation (persistent, raised above)
metrics = ["CPU Usage (%)", "RAM Usage (%)", "GPU Memory (MB)", "YOLO Inference Time (s)"]
title_positions = [0.14, 0.39, 0.63, 0.88]  # Adjusted further to reduce overlap with suptitle
for pos, title in zip(title_positions, metrics):
    fig.text(pos, 0.9, title, ha='center', va='bottom', fontsize=12, weight='bold')

# Animate function
def animate(i):
    for idx, cam_id in enumerate(CAMERA_IDS):
        log_dir = f"logs/camera{cam_id}"
        system_log = os.path.join(log_dir, "system_usage.csv")
        inference_log = os.path.join(log_dir, "inference_log.csv")

        try:
            df_sys = pd.read_csv(system_log).tail(100)
            df_inf = pd.read_csv(inference_log).tail(100)
        except Exception:
            continue

        axs[idx][0].cla()
        axs[idx][1].cla()
        axs[idx][2].cla()
        axs[idx][3].cla()

        axs[idx][0].plot(df_sys["CPU_percent"], label=f"CPU %", color='tab:red')
        axs[idx][1].plot(df_sys["Memory_percent"], label=f"RAM %", color='tab:blue')
        axs[idx][2].plot(df_sys["GPU_memory_MB"], label=f"GPU %", color='tab:green')
        axs[idx][3].plot(df_inf["YOLO_InferenceTime_s"], label=f"YOLO Time", color='tab:purple')

        for j in range(4):
            axs[idx][j].legend()
            axs[idx][j].grid(True)
            if idx == len(CAMERA_IDS) - 1:
                axs[idx][j].set_xlabel("Frame")
            if j == 0:
                axs[idx][j].set_ylabel(f"Cam {cam_id}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

# Start animation
ani = FuncAnimation(fig, animate, interval=1000)
plt.show()