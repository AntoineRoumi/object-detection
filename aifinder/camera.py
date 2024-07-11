from typing import TypeAlias
import pyrealsense2 as rs
import numpy as np
from .model import BoundingBox

Coords3D: TypeAlias = tuple[float, float, float]
"""Represents coordinates in a 3D space as a tuple of floats."""


class DepthCamera:
    """Class that facilitates the usage of an Intel Realsense D4XX camera."""

    def __init__(self, width: int, height: int, fps: int) -> None:
        self.pipeline = rs.pipeline()  # pyright: ignore
        self.config = rs.config()  # pyright: ignore
        self.config.enable_stream(rs.stream.color, width, height,
                                  rs.format.rgb8, fps)  # pyright: ignore
        self.config.enable_stream(rs.stream.depth, width, height,
                                  rs.format.z16, fps)  # pyright: ignore
        self.cfg = self.pipeline.start(self.config)  # pyright: ignore
        self.profile = self.cfg.get_stream(rs.stream.depth)  # pyright: ignore
        self.intrinsics = self.profile.as_video_stream_profile(
        ).get_intrinsics()  # pyright: ignore
        self.frame = None
        self.color_frame = None
        self.depth_frame = None
        self.pc = rs.pointcloud()  # pyright: ignore
        self.points = None

    def update_frame(self) -> None:
        """Tells the camera to wait for new frames, and store them for later use.
        
        Also updates the pointcloud of the depthframe."""
        self.frame = self.pipeline.wait_for_frames()
        self.color_frame = self.frame.get_color_frame()
        self.depth_frame = self.frame.get_depth_frame()
        self.points = self.pc.calculate(self.depth_frame)
        self.pc.map_to(self.color_frame)

    def get_point_cloud_vertices(self):
        """Returns the vertices of the pointcloud calculated from the depth frame."""
        return self.points.get_vertices()  # pyright: ignore

    def get_point_cloud_texcoords(self):
        """Returns the texture coordinates of the point cloud calculated from the depth frame."""
        return self.points.get_texture_coordinates()  # pyright: ignore

    def get_color_frame(self) -> rs.frame | None:  # pyright: ignore
        """Returns the color frame sent by the camera, None if no frame is currently stored."""
        return self.color_frame

    def get_color_frame_as_ndarray(self) -> np.ndarray | None:
        """Returns the color frame sent by the camera, formatted as a Numpy 3 dimensional array.
        
        The outer array represents the rows of the image pixels, 
        the middle arrays represent the columns of the image pixels
        and the inner array represents the RGB values of the pixels."""
        return np.asanyarray(self.color_frame.get_data()) if (
            self.color_frame is not None) else None

    def get_depth_frame(self) -> rs.frame | None:  # pyright: ignore
        """Returns the depth frame sent by the camera, None if no frame is currently stored."""
        return self.depth_frame

    # Returns distance in mm, None if distance is 0 (impossible due to camera constraints)
    def get_distance(self, x: int, y: int) -> float | None:
        """Calculates the distance between the camera and the pixel (x,y) on the color frame.

        x: x coordinate of the pixel
        y: y coordinate of the pixel

        Returns the distance in millimeters, or None if distance cannot be calculated."""

        distance = self.depth_frame.get_distance(x, y) * 1000 if (
            self.depth_frame is not None) else None
        return distance if distance != 0 else None

    # Returns the coordinates AND the distance of a pixel
    def get_coords_of_pixel(
            self, x: int,
            y: int) -> tuple[Coords3D, float] | tuple[None, None]:
        """Calculates the 3D coordinates of the pixel (x,y) relative to the camera.

        x: x coordinate of the pixel
        y: y coordinate of the pixel

        Returns the 3D coordinates and the distance (in mm) of the pixel, or (None, None) if distance cannot be calculated."""

        distance = self.get_distance(x, y)
        return (
            rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], distance),
            distance) if (distance is not None) else (None,
                                                      None)  # pyright: ignore

    def get_coords_of_object(
            self, x0: int, y0: int, x1: int,
            y1: int) -> tuple[Coords3D, float] | tuple[None, None]:
        """Calculates the coordinates of an object in a 3D space relative to the camera, using its bounding box.
        The calculation is made by using the center pixel of the bounding box and calculating its coordinates.
        
        x0: x coordinate of the top left corner of the bounding box 
        y0: y coordinate of the top left corner of the bounding box
        x1: x coordinate of the bottom right corner of the bounding box
        y1: y coordinate of the bottom right corner of the bounding box

        Returns the 3D coordinates and the distance (in mm) of the pixel, or (None, None) if distance cannot be calculated."""

        center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
        return self.get_coords_of_pixel(center_x, center_y)

    # Returns the coordinates AND the distance of an object, according to its bounding box
    def get_coords_of_object_xyxy(
            self,
            box: BoundingBox) -> tuple[Coords3D, float] | tuple[None, None]:
        """Same as DepthCamera.get_coords_of_object, but takes in parameters the BoundingBox type.

        box: the bounding box of the object in the (x0, y0, x1, y1) format

        Returns the 3D coordinates and the distance (in mm) of the pixel, or (None, None) if distance cannot be calculated."""

        return self.get_coords_of_object(x0=box[0],
                                         y0=box[1],
                                         x1=box[2],
                                         y1=box[3])

    def terminate(self) -> None:
        """Used to stop gracefully the camera."""

        self.pipeline.stop()
