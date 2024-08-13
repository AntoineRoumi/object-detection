from enum import Enum
from threading import Semaphore
import pyrealsense2 as rs
import numpy as np
import math
from .bounding_box import BoundingBox
from .point import Point3D

class CenterMode(Enum):
    MODE_2D = 0
    MODE_3D = 1

class DepthCamera:
    """Class that facilitates the usage of an Intel Realsense D4XX camera."""

    def __init__(self, width: int, height: int, fps: int) -> None:
        self.pipeline = rs.pipeline()  # pyright: ignore
        self.config = rs.config()  # pyright: ignore
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)  # pyright: ignore
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)  # pyright: ignore
        self.cfg = self.pipeline.start(self.config)  # pyright: ignore
        self.profile = self.cfg.get_stream(rs.stream.depth)  # pyright: ignore
        self.intrinsics = self.profile.as_video_stream_profile().get_intrinsics()  # pyright: ignore
        self.frame = None
        self.color_frame = None
        self.depth_frame = None
        self.pc = rs.pointcloud()  # pyright: ignore
        self.points = None
        self.updating_frames = Semaphore()
        self.align = rs.align(rs.stream.color) # pyright: ignore
        
    def update_frame(self) -> None:
        """Tells the camera to wait for new frames, and store them for later use.
        
        Also updates the pointcloud of the depthframe."""
        try:
            self.frame = self.align.process(self.pipeline.wait_for_frames())
        except RuntimeError as e:
            print(f"Pipeline not started or stopped: {e}")
            return
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

        if self.color_frame is None: 
            return None

        image = np.asanyarray(self.color_frame.get_data())

        # Uncomment following lines if OpenCV is compiled with CUDA
        
        # gpu_image = cv2.cuda.GpuMat()
        # gpu_image.upload(image)
        # image = cv2.cuda.fastNlMeansDenoisingColored(gpu_image, 10, 10)
        # image = np.asanyarray(image.download())

        return image

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

    def coords_and_distance_to_point(self, x: int, y: int, distance: float) -> Point3D:
        coords = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], distance) # pyright: ignore
        return Point3D(coords[0], coords[1], coords[2])

    # Returns the coordinates AND the distance of a pixel
    def get_coords_of_pixel(self, x: int, y: int) -> tuple[Point3D, float] | tuple[None, None]:
        """Calculates the 3D coordinates of the pixel (x,y) relative to the camera.

        x: x coordinate of the pixel
        y: y coordinate of the pixel

        Returns the 3D coordinates and the distance (in mm) of the pixel, or (None, None) if distance cannot be calculated."""

        distance = self.get_distance(x, y)
        if distance is None:
            return (None, None)
        coords = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], distance) # pyright: ignore
        return (Point3D(coords[0], coords[1], coords[2]), distance)

    def get_size_of_object(self, x0: int, y0: int, x1: int, y1: int) -> tuple[float, float] | None:
        center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2 
        center_distance = self.get_distance(center_x, center_y)

        if center_distance is None:
            return None
        
        coords_left = self.coords_and_distance_to_point(x0, center_y, center_distance)
        coords_right = self.coords_and_distance_to_point(x1, center_y, center_distance)
        width = math.sqrt((coords_right.x - coords_left.x)**2 +
                          (coords_right.y - coords_left.y)**2 +
                          (coords_right.z - coords_left.z)**2)
        coords_top = self.coords_and_distance_to_point(center_x, y0, center_distance)
        coords_bottom = self.coords_and_distance_to_point(center_x, y1, center_distance)
        height = math.sqrt((coords_bottom.x - coords_top.x)**2 +
                           (coords_bottom.y - coords_top.y)**2 + 
                           (coords_bottom.z - coords_top.z)**2)

        return (width, height)

    def get_size_of_object_xyxy(self, bbox: BoundingBox) -> tuple[float, float] | None:
        """Calculates the width and height of the given 2D bounding box in the camera 3D space.

        bbox: the bounding box of the object in the (x0, y0, x1, y1) format

        Returns the size as a tuple (width, height), or returns None if the size cannot be calculated."""

        return self.get_size_of_object(x0=bbox.x0, y0=bbox.y0, x1=bbox.x1, y1=bbox.y1)
        
    def get_coords_of_object(self, x0: int, y0: int, x1: int, y1: int, center_mode: CenterMode = CenterMode.MODE_2D) -> tuple[Point3D, float] | tuple[None, None]:
        """Calculates the coordinates of an object in a 3D space relative to the camera, using its bounding box.
        The calculation is made by using the center pixel of the bounding box and calculating its coordinates.
        
        x0: x coordinate of the top left corner of the bounding box 
        y0: y coordinate of the top left corner of the bounding box
        x1: x coordinate of the bottom right corner of the bounding box
        y1: y coordinate of the bottom right corner of the bounding box
        center_mode: specified how the center of the object is calculated: 
            if equal to CenterMode.MODE_2D:
                the center is the center of the 2D bounding box of the object
            if equal to CenterMode.MODE_3D:
                the center is placed on the line between the camera center and the bounding box center, 
                at a distance of half the width of the bounding box from the bbox center.
                This mode should only be used if the object's width and depth are nearly the same, otherwise the calculation will be wrong.
            default value is CenterMode.MODE_2D

        Returns the 3D coordinates and the distance (in mm) of the pixel, or (None, None) if distance cannot be calculated."""

        center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
        size = self.get_size_of_object(x0, y0, x1, y1)
        if size is not None and center_mode == CenterMode.MODE_3D:
            center_3d_x = center_x + int(size[0]) // 2 
            return self.get_coords_of_pixel(center_3d_x, center_y)
        else:
            return self.get_coords_of_pixel(center_x, center_y)

    # Returns the coordinates AND the distance of an object, according to its bounding box
    def get_coords_of_object_xyxy(self, bbox: BoundingBox, center_mode: CenterMode = CenterMode.MODE_2D) -> tuple[Point3D, float] | tuple[None, None]:
        """Same as DepthCamera.get_coords_of_object, but takes in parameters the BoundingBox type.

        bbox: the bounding box of the object in the (x0, y0, x1, y1) format

        Returns the 3D coordinates and the distance (in mm) of the pixel, or (None, None) if distance cannot be calculated."""

        return self.get_coords_of_object(x0=bbox.x0,
                                         y0=bbox.y0,
                                         x1=bbox.x1,
                                         y1=bbox.y1,
                                         center_mode=center_mode)

    def terminate(self) -> None:
        """Used to stop gracefully the camera."""

        self.pipeline.stop()
