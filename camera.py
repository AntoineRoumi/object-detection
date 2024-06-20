from typing import TypeAlias
import pyrealsense2 as rs
import numpy as np
from model import BoundingBox

Coords3D: TypeAlias = tuple[float, float, float]

class DepthCamera:
    def __init__(self, width: int, height: int, fps: int) -> None:
        self.pipeline = rs.pipeline() # pyright: ignore
        self.config = rs.config() # pyright: ignore
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps) # pyright: ignore
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps) # pyright: ignore
        self.cfg = self.pipeline.start(self.config) # pyright: ignore
        self.profile = self.cfg.get_stream(rs.stream.depth) # pyright: ignore
        self.intrinsics = self.profile.as_video_stream_profile().get_intrinsics() # pyright: ignore
        self.frame = None
        self.color_frame = None
        self.depth_frame = None

    def update_frame(self) -> None:
        self.frame = self.pipeline.wait_for_frames()
        self.color_frame = self.frame.get_color_frame()
        self.depth_frame = self.frame.get_depth_frame()

    def get_color_frame(self):
        return self.color_frame

    def get_color_frame_as_ndarray(self) -> np.ndarray | None:
        return np.asanyarray(self.color_frame.get_data()) if (self.color_frame is not None) else None

    def get_depth_frame(self):
        return self.depth_frame

    # Returns distance in mm, None if distance is 0 (impossible due to camera constraints)
    def get_distance(self, x: int, y: int) -> float | None:
        distance = self.depth_frame.get_distance(x, y) * 1000 if (self.depth_frame is not None) else None
        return distance if distance != 0 else None

    # Returns the coordinates AND the distance of a pixel
    def get_coords_of_pixel(self, x: int, y: int) -> tuple[Coords3D, float] | tuple[None, None]:
        distance = self.get_distance(x, y)
        return (rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], distance), distance) if (distance is not None) else (None, None) # pyright: ignore

    # Returns the coordinates AND the distance of an object, according to its bounding box
    def get_coords_of_object(self, x0: int, y0: int, x1: int, y1: int) -> tuple[Coords3D, float] | tuple[None, None]:
        center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
        return self.get_coords_of_pixel(center_x, center_y)

    # Returns the coordinates AND the distance of an object, according to its bounding box
    def get_coords_of_object_xyxy(self, box: BoundingBox) -> tuple[Coords3D, float] | tuple[None, None]:
        return self.get_coords_of_object(x0=box[0], y0=box[1], x1=box[2], y1=box[3])

    def terminate(self):
        self.pipeline.stop()
