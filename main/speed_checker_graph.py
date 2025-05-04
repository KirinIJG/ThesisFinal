import matplotlib.pyplot as plt
import pandas as pd

# Replace with actual file path
csv_file = "speed_logs_new_object_1.csv"

df = pd.read_csv(csv_file, names=["frame", "raw_speed", "adjusted_speed"])

plt.figure(figsize=(10, 5))
plt.plot(df["frame"], df["raw_speed"], label="Raw Pixel Speed (km/h)", linestyle="--")
plt.plot(df["frame"], df["adjusted_speed"], label="Distance-Aware Speed (km/h)", linewidth=2)
plt.xlabel("Frame Number")
plt.ylabel("Speed (km/h)")
plt.title("Comparison of Speed Estimation Methods")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
