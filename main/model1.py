from ultralytics import YOLO
import torch

def load_model():
    """Load the YOLO model and send it to GPU if available."""
    model_path = "./best1.pt"
    try:
        model = YOLO(model_path)
        if torch.cuda.is_available():
            model.to("cuda")
        print("YOLO model loaded successfully.")
        print("CUDA Available:", torch.cuda.is_available())
        print("Current Device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
        print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None
