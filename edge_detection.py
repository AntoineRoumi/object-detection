import numpy as np
import cv2
import model

def edge_detection_rectangle_on_frame(frame: np.ndarray, bbox: model.BoundingBox, canny_low: int = 100, canny_high: int = 200):
    x0, y0, x1, y1 = bbox
    part_of_frame = frame[y0:y1, x0:x1]
    return cv2.Canny(part_of_frame, canny_low, canny_high)
