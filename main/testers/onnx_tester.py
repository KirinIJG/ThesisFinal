import onnxruntime as ort
import numpy as np
import cv2

from onnx_postprocess import parse_yolov8_output

# Load image and preprocess
img_path = "./pics/test.png"
img = cv2.imread(img_path)
original_shape = img.shape[:2]  # (H, W)
img_resized = cv2.resize(img, (416, 416))  # Match ONNX input size

# Normalize and convert to NCHW with float16 (ONNX expects float16)
img_norm = img_resized.astype(np.float32) / 255.0
img_trans = np.transpose(img_norm, (2, 0, 1))  # HWC to CHW
img_input = np.expand_dims(img_trans, axis=0)
input_tensor = np.ascontiguousarray(img_input)

# Load ONNX model
session = ort.InferenceSession("./best1.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_tensor})

# Postprocess
print("Output shape:", outputs[0].shape)
print("Sample row:", outputs[0][0][:5])

detections = parse_yolov8_output(outputs[0], original_shape, conf_thres=0.25, iou_thres=0.45, num_classes=12)

# Draw detections
for x1, y1, x2, y2, conf, cls in detections:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{cls}: {conf:.2f}"
    cv2.putText(img, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imshow("ONNX Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
