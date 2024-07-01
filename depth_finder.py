"""Main interface for detection of objects and their coordinates."""

from camera import DepthCamera, Coords3D
from model import YoloModel
import color_recognition
import image_manipulation as imanip

TRAINING_DATA_DIR = './training_dataset'
TRAINING_DATA_FILE = './training.data'


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
        color_recognition.training(TRAINING_DATA_DIR, TRAINING_DATA_FILE)
        self.color_classifier = color_recognition.KnnClassifier(
            TRAINING_DATA_FILE)

    def update(self, *kwargs) -> None:
        """Updates the frames of the camera, and process an object detection on the updated color frame.

        *kwargs: any available paramater for the ultralytics predict function
            https://docs.ultralytics.com/modes/predict/#inference-arguments"""

        self.camera.update_frame()
        self.frame = self.camera.get_color_frame_as_ndarray()
        self.results = self.model.predict_frame(
            self.frame, *kwargs) if self.frame is not None else None

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

        if self.results is None:
            return None

        if self.frame is None:
            return None

        max_conf = 0.0
        max_coords = None
        for i in range(self.results.results_count()):
            if self.results.get_class_id(i) != class_id:
                continue
            if self.results.get_conf(i) < max_conf:
                continue
            bb_box = self.results.get_box_coords(i)
            test_histogram = color_recognition.color_histogram_of_image(
                imanip.extract_area_from_image(self.frame, bb_box[0],
                                               bb_box[1], bb_box[2],
                                               bb_box[3]))
            if self.color_classifier.predict(test_histogram) != color_name:
                continue
            coords, _ = self.camera.get_coords_of_object_xyxy(
                self.results.get_box_coords(i))
            if coords is None:
                continue
            max_coords = coords

        return max_coords

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
            coords, _ = self.camera.get_coords_of_object_xyxy(
                self.results.get_box_coords(i))
            if coords is None:
                continue
            max_coords = coords

        return max_coords

    def terminate(self) -> None:
        """Stops gracefully the camera. 
        Must be called at the end of the program for proper exiting."""

        self.camera.terminate()
