import pstats
import io
import csv

# === Inputs ===
test_number = int(input("Test Number: "))
camera_index = int(input("Camera Number: "))

# === Profile file path ===
profile_path = f"logs/test{test_number}/camera{camera_index}/cprofile_stats.prof"

# === Capture pstats output ===
stream = io.StringIO()
p = pstats.Stats(profile_path, stream=stream)
p.strip_dirs().sort_stats("cumulative").print_stats()

# === Process lines from stats ===
lines = stream.getvalue().splitlines()
data_lines = [line for line in lines if line.strip() and line[0].isdigit()]

# === Save to CSV ===
csv_path = f"logs/test{test_number}/camera{camera_index}/profile_output.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ncalls", "tottime", "percall1", "cumtime", "percall2", "filename:function"])
    
    for line in data_lines:
        try:
            parts = line.split(None, 5)
            writer.writerow(parts)
        except Exception:
            pass  # skip malformed lines

print(f"CSV saved to {csv_path}")
