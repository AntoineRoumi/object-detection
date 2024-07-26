"""Main interface for detection of objects and their coordinates."""

import json
from .camera import CenterMode, DepthCamera, Coords3D
from .model import BoundingBox, YoloModel
from . import color_recognition as cr
from . import image_manipulation as imanip
from . import edge_detection as ed
from dataclasses import dataclass 

TRAINING_DATA_DIR = './training_dataset'
TRAINING_DATA_FILE = './training.data'


@dataclass
class ResultObject:
    coords: Coords3D | None = None
    distance: float | None = None
    bbox: BoundingBox = (0,0,0,0)
    class_name: str = ""
    color: str = ""
    conf: float = 0.0

class DepthFinder:
    """Class used to query the frames of an Intel Realsense 4XX camera, and find objects on them using the YOLO algorithm."""

    def __init__(self, width: int, height: int, fps: int,
                 weights: str) -> None:
        """width: width of the camera frames
        height: height of the camera frames
        weights: weights file used by the Yolo algorithm"""

        self.camera = DepthCamera(width, height, fps)
        self.model = YoloModel(weights)
        self.results = None
        self.frame = None
        self.visible_objects: list[ResultObject] | None = None
        self.center_mode: CenterMode = CenterMode.MODE_2D
        cr.training(TRAINING_DATA_DIR, TRAINING_DATA_FILE)
        self.color_classifier = cr.KnnClassifier(TRAINING_DATA_FILE)
        
    def set_center_mode_from_dim(self, dim: int):
        if dim == 2:
            self.center_mode = CenterMode.MODE_2D
        elif dim == 3:
            self.center_mode = CenterMode.MODE_3D

    def update(self, **kwargs) -> None:
        """Updates the frames of the camera, and process an object detection on the updated color frame.

        *kwargs: any available paramater for the ultralytics predict function
            https://docs.ultralytics.com/modes/predict/#inference-arguments"""

        self.camera.update_frame()
        self.frame = self.camera.get_color_frame_as_ndarray()
        self.results = self.model.predict_frame(self.frame, **kwargs) if self.frame is not None else None
        self.update_visible_objects()

    def get_classes_names(self) -> list[str]:
        """Returns the names of the classes detectable by the Yolo algorithm."""

        return list(self.model.classes_names.values())

    def get_id_from_class_name(self, class_name: str) -> int | None:
        """Returns the id of a detectable class by its name.
        Returns None if that class name doesn't exist."""

        return self.model.classes_ids.get(class_name)

    def find_object_by_name_and_color(self, class_name: str,
                                      color_name: str) -> Coords3D | None:
        """Returns the coordinates of an object given its class name and color.
        
        class_name: the name of the class to be detected
        color_name: the name of the color of the object

        Returns a Coords3D object if the object is detected and is in range, None otherwise."""

        index = self.get_id_from_class_name(class_name)

        if index is None:
            return None

        return self.find_object_by_id_and_color(index, color_name)

    def find_object_by_id_and_color(self, class_id: int,
                                    color_name: str) -> Coords3D | None:
        """Returns the coordinates of an object given its class id and color.
        
        class_id: the id of the class to be detected
        color_name: the name of the color of the object

        Returns a Coords3D object if the object is detected and is in range, None otherwise."""

        if self.results is None or self.frame is None:
            return None

        max_conf = 0.0
        max_coords = None
        for i in range(self.results.results_count()):
            if self.results.get_class_id(i) != class_id:
                continue
            if self.results.get_conf(i) < max_conf:
                continue
            bbox = self.results.get_box_coords(i)
            test_histogram = cr.color_histogram_of_image(imanip.extract_area_from_image(self.frame, bbox))
            if self.color_classifier.predict(test_histogram) != color_name:
                continue
            coords, _ = self.camera.get_coords_of_object_xyxy(
                self.results.get_box_coords(i))
            if coords is None:
                continue
            max_coords = coords

        return max_coords

    def get_color_of_box(self, bbox: BoundingBox) -> str:
        """Returns the color of the area inside the specified bounding box.

        bbox: bounding box for which we want to get the inner color

        Returns the color name as a str."""
        if self.frame is None or self.results is None:
            return ""

        test_histogram = cr.color_histogram_of_image(
            imanip.extract_area_from_image(self.frame, bbox)
        )

        return self.color_classifier.predict(test_histogram)

    def find_object_by_name(self, class_name: str) -> Coords3D | None:
        """Returns the coordinates of an object given its class name.

        class_name: the name of the class to be detected

        Returns a Coords3D object if the object is detected and is in range, None otherwise."""

        index = self.model.classes_ids.get(class_name.strip())

        return self.find_object_by_id(index) if index is not None else None

    def find_object_by_id(self, class_id: int) -> Coords3D | None:
        """Returns the coordinates of an object given its class id.

        class_id: the id of the class to be detected

        Returns a Coords3D object if the object is detected and is in range, None otherwise."""

        if self.results is None:
            return None

        max_conf = 0.0
        max_coords = None
        for i in range(self.results.results_count()):
            if self.results.get_class_id(i) != class_id:
                continue
            if self.results.get_conf(i) < max_conf:
                continue
            coords, _ = self.camera.get_coords_of_object_xyxy(self.results.get_box_coords(i), self.center_mode)
            if coords is None:
                continue
            max_coords = coords

        return max_coords

    def update_visible_objects(self) -> None:
        """Updates the list of the objects detected by the camera."""

        print("update")

        if self.results is None:
            return None

        self.visible_objects = []

        for i in range(self.results.results_count()):
            bbox = self.results.get_box_coords(i)
            coords, distance = self.camera.get_coords_of_object_xyxy(bbox, self.center_mode)
            class_name = self.results.get_class_name(i)
            color_name = self.get_color_of_box(bbox)
            conf = self.results.get_conf(i)
            self.visible_objects.append(ResultObject(coords, distance, bbox, class_name, color_name, conf))

    def get_edges_of_object(self, index: int) -> list[list[int]] | None:
        if self.results is None or index >= self.results.results_count() or index < 0 or self.frame is None:
            return None

        bbox = self.results.get_box_coords(index)
        edges = ed.edge_detection_rectangle_on_frame(self.frame, bbox, canny_low=100, canny_high=200)

        edges = edges.tolist()

        return edges


    def get_size_of_object(self, index: int) -> tuple[float, float] | None:
        """Returns the size (width and height in mm) of the index-th object."""

        if self.results is None or index >= self.results.results_count() or index < 0 or self.frame is None:
            return None

        bbox = self.results.get_box_coords(index)
        size = self.camera.get_size_of_object_xyxy(bbox)
    
        return size
    
    def to_json(self) -> str:
        if self.visible_objects is None:
            return ""

        json_list = []

        for obj in self.visible_objects:
            if obj.distance is None or obj.coords is None:
                json_list.append({
                    'conf': obj.conf,
                    'class_name': obj.class_name,
                    'color_name': obj.color,
                    'bbox': obj.bbox,
                })
            else:
                json_list.append({
                    'conf': obj.conf,
                    'class_name': obj.class_name,
                    'color_name': obj.color,
                    'bbox': obj.bbox,
                    'distance': obj.distance,
                    'coords': obj.coords,
                })

        return json.dumps(json_list)

    def terminate(self) -> None:
        """Stops gracefully the camera. 
        Must be called at the end of the program for proper exiting."""

        self.camera.terminate()


