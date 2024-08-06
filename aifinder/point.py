from dataclasses import dataclass

@dataclass
class Point:
    x: float = 0.
    y: float = 0.
    z: float = 0.

def point_from_tuple(coords: tuple):
    return Point(coords[0], coords[1], coords[2])
