import numpy as np

from .model import BoundingBox

def extract_area_from_image(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    """Extract a rectangle area of an image and puts it in another image.

    image: 3D Numpy array representing a RGB image 
    bbox: coordinates of the rectangle on the image 

    Returns a 3D Numpy array with the extracted area (in RGB)."""

    x0, y0, x1, y1 = bbox

    width = x1 - x0
    height = y1 - y0
    extracted_image = np.empty((height, width, 3), dtype=image.dtype)
    for y in range(0, height):
        extracted_image[y] = np.array(image[y0 + y][x0:x1])

    return extracted_image
