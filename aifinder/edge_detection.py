import numpy as np
import cv2
from .bounding_box import BoundingBox

def edge_detection_rectangle_on_frame(frame: np.ndarray, bbox: BoundingBox, canny_low: int = 100, canny_high: int = 200):
    part_of_frame = frame[bbox.y0:bbox.y1, bbox.x0:bbox.x1]
    img_blur = cv2.blur(part_of_frame, (3,3))
    edges = cv2.Canny(img_blur, canny_low, canny_high)
    return edges
