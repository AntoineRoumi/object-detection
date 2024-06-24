import numpy as np

def extract_area_from_image(image: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    width = x1 - x0
    height = y1 - y0
    extracted_image = np.empty((height,width,3), dtype=image.dtype)
    for y in range(0, height):
        extracted_image[y] = np.array(image[y0 + y][x0:x1])

    return extracted_image
