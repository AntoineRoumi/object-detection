import numpy as np
from .point import Point3D

class CoordinatesConverter:
    def __init__(self, origin: Point3D, origin_offset: Point3D, x: Point3D, y: Point3D, z: Point3D, scale: float = 0.1):
        self.origin = origin
        self.origin_offset = origin_offset
        self.scale = scale
        self.coords_mat = np.array([[ x.x - origin.x, y.x - origin.x, z.x - origin.x ],
                               [ x.y - origin.y, y.y - origin.y, z.y - origin.y ],
                               [ x.z - origin.z, y.z - origin.z, z.z - origin.z ]]) / scale
        self.inverted_coords_mat = np.linalg.inv(self.coords_mat)

    def to_coords(self, coords: Point3D) -> Point3D:
        coords_vector = np.array([[coords.x - self.origin.x], [coords.y - self.origin.y], [coords.z - self.origin.z]])
        new_coords = np.matmul(self.inverted_coords_mat, coords_vector)
        return Point3D(new_coords[0,0] + self.origin_offset.x,
                     new_coords[1,0] + self.origin_offset.y, 
                     new_coords[2,0] + self.origin_offset.z)

    def from_coords(self, coords: Point3D) -> Point3D:
        coords_vector = np.array([[coords.x - self.origin_offset.x], [coords.y - self.origin_offset.y], [coords.z - self.origin_offset.z]])
        new_coords = np.matmul(self.coords_mat, coords_vector)
        return Point3D(new_coords[0,0] + self.origin.x, new_coords[1,0] + self.origin.y, new_coords[2,0] + self.origin.z)
