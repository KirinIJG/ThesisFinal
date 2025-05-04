import os
import glob
import csv

def combine_logs(log_type: str, output_file: str):
    """
    Combines all logs of a given type (e.g., processing_log.csv) into a single CSV.
    Each row is tagged with the originating camera name.
    """
    combined_rows = []
    header_written = False

    for filepath in sorted(glob.glob(f"logs/*_{log_type}")):
        camera_id = os.path.basename(filepath).split('_')[0]  # e.g., 'camera1'

        with open(filepath, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            if not header_written:
                combined_rows.append(["Camera"] + header)
                header_written = True

            for row in reader:
                combined_rows.append([camera_id] + row)

    with open(output_file, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerows(combined_rows)

if __name__ == "__main__":
    os.makedirs("logs/combined", exist_ok=True)

    combine_logs("processing_log.csv", "logs/combined/processing_log_combined.csv")
    combine_logs("objects_log.csv", "logs/combined/objects_log_combined.csv")
    combine_logs("detection_count.csv", "logs/combined/detection_count_combined.csv")
    combine_logs("system_usage.csv", "logs/combined/system_usage_combined.csv")

    print("? All logs combined into logs/combined/")
