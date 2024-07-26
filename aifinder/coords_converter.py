import math
from dataclasses import dataclass

@dataclass
class Point:
    x: float = 0.
    y: float = 0.
    z: float = 0.

def calculate_theta(p1: Point, p2: Point):
    # p1 and p2 are 2 points on the same line parallel to the x axis of the robot space, in the camera coordinates system
    if p1.x == p2.x:
        raise Exception("Math error: cannot divide by zero in calculate_theta")
    if (p1.x > p2.x and p1.z < p2.z) or (p1.x < p2.x and p1.z > p2.z): 
        # if the angle is negative
        return -math.atan(abs(p1.x - p2.x)/abs(p1.z - p2.z))
    else:
        return math.atan(abs(p1.x - p2.x)/abs(p1.z - p2.z))

def convert_coords(point: Point, origin: Point, theta: float) -> Point:
    # Converts the point coordinates from the camera coordinates system to the robot coordinates system
    # The robot coordinates system is designated by the coordinates of its origin in the camera space,
    # and by the horizontal angle (theta) between the two coordinates system
    new_point = point
    new_point.x += origin.x
    new_point.z += origin.z

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    
    p_x = new_point.x * cos_theta + new_point.y * sin_theta
    p_z = -new_point.x * sin_theta + new_point.y * cos_theta
    p_y = new_point.y + origin.y 

    return Point(p_x, p_y, p_z)
