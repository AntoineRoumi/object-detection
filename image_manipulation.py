import numpy as np

def extract_area_from_image(image: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Extract a rectangle area of an image and puts it in another image.

    image: 3D Numpy array representing a RGB image 
    x0: x coordinate of the top left corner of the rectangle
    y0: y coordinate of the top left corner of the rectangle
    x1: x coordinate of the bottom right corner of the rectangle
    y1: y coordinate of the bottom right corner of the rectangle

    Returns a 3D Numpy array with the extracted area (in RGB)."""

    width = x1 - x0
    height = y1 - y0
    extracted_image = np.empty((height,width,3), dtype=image.dtype)
    for y in range(0, height):
        extracted_image[y] = np.array(image[y0 + y][x0:x1])

    return extracted_image
