from jtop import jtop
import time

print("Starting GPU memory monitor...\n")

try:
    with jtop() as jetson:
        while jetson.ok():
            stats = jetson.stats
            print("-" * 40)
            print("Raw jtop.stats['GPU'] =", stats.get("GPU"))
            try:
                mem_used = stats["GPU"]["mem_used"]
                print(f"GPU Memory Used: {mem_used} MB")
            except (KeyError, TypeError):
                print("?? Could not access 'GPU' ? 'mem_used'")
            time.sleep(1)
except Exception as e:
    print(f"Error initializing jtop: {e}")
