from dataclasses import dataclass

@dataclass
class BoundingBox:
    x0: int
    y0: int
    x1: int
    y1: int
