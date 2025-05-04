import numpy as np
import cv2

def parse_yolov8_output(output, original_shape, conf_thres=0.25, iou_thres=0.45, num_classes=12):
    """
    Parses YOLOv8 ONNX output and returns bounding boxes after NMS.

    Args:
        output: The raw model output (numpy array)
        original_shape: (H, W) of the original image before resize
        conf_thres: Confidence threshold to filter detections
        iou_thres: IoU threshold for NMS
        num_classes: Number of classes in your model

    Returns:
        List of [x1, y1, x2, y2, conf, class_id]
    """
    # Handle output shape like (1, 17, 3549)
    if output.shape[1] <= 100 and output.shape[2] > 100:
        output = np.transpose(output, (0, 2, 1))  # (1, 3549, 17)

    preds = output[0]  # shape: (N, 17)
    boxes = preds[:, :4]
    obj_conf = preds[:, 4]
    cls_scores = preds[:, 5:]

    cls_ids = np.argmax(cls_scores, axis=1)
    cls_confs = np.max(cls_scores, axis=1)
    final_scores = obj_conf * cls_confs

    # Filter by confidence threshold
    mask = final_scores >= conf_thres
    boxes = boxes[mask]
    final_scores = final_scores[mask]
    cls_ids = cls_ids[mask]

    # Convert xywh ? xyxy
    xy = boxes[:, :2]
    wh = boxes[:, 2:]
    xy1 = xy - wh / 2
    xy2 = xy + wh / 2
    boxes_xyxy = np.concatenate([xy1, xy2], axis=1)

    # Scale boxes back to original image size
    input_size = 416  # from letterbox
    h0, w0 = original_shape
    r = min(input_size / w0, input_size / h0)
    dw, dh = (input_size - r * w0) / 2, (input_size - r * h0) / 2

    boxes_xyxy[:, [0, 2]] -= dw
    boxes_xyxy[:, [1, 3]] -= dh
    boxes_xyxy /= r

    boxes_xyxy = boxes_xyxy.clip(min=0)
    boxes_int = boxes_xyxy.astype(int)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_int.tolist(),
        scores=final_scores.tolist(),
        score_threshold=conf_thres,
        nms_threshold=iou_thres
    )

    result = []
    if isinstance(indices, (tuple, list)) and len(indices) > 0:
        indices = np.array(indices).flatten()
    elif isinstance(indices, np.ndarray):
        indices = indices.flatten()
    else:
        indices = []

    for i in indices:
        x1, y1, x2, y2 = boxes_int[i]
        conf = final_scores[i]
        cls = int(cls_ids[i])
        result.append([x1, y1, x2, y2, conf, cls])

    return result
