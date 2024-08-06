import numpy as np
from .point import Point

class CoordinatesConverter:
    def __init__(self, origin: Point, x: Point, y: Point, z: Point, scale: float = 0.1):
        self.origin = origin
        self.coords_mat = np.array([[ x.x - origin.x, y.x - origin.x, z.x - origin.x ],
                               [ x.y - origin.y, y.y - origin.y, z.y - origin.y ],
                               [ x.z - origin.z, y.z - origin.z, z.z - origin.z ]]) / scale
        print(self.coords_mat)
        self.inverted_coords_mat = np.linalg.inv(self.coords_mat)

    def to_coords(self, coords: Point) -> Point:
        coords_vector = np.array([[coords.x - self.origin.x], [coords.y - self.origin.y], [coords.z - self.origin.z]])
        print(coords_vector)
        new_coords = np.matmul(self.inverted_coords_mat, coords_vector)
        return Point(new_coords[0,0], new_coords[1,0], new_coords[2,0])

    def from_coords(self, coords: Point) -> Point:
        coords_vector = np.array([[coords.x], [coords.y], [coords.z]])
        new_coords = np.matmul(self.coords_mat, coords_vector)
        return Point(new_coords[0,0] + self.origin.x, new_coords[1,0] + self.origin.y, new_coords[2,0] + self.origin.z)

cv = CoordinatesConverter(Point(-100, -190, 2080), Point(-5, -230, 2100), Point(-130, -210, 2170), Point(-126, -282, 2050))
print(cv.to_coords(Point(-50, -210, 2090)))
print(cv.from_coords(Point(0.1, 0., 0.)))
