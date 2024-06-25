from camera import DepthCamera, Coords3D
from model import YoloModel
import color_recognition
import image_manipulation as imanip

TRAINING_DATA_DIR = './training_dataset'
TRAINING_DATA_FILE = './training.data'

class DepthFinder:
    def __init__(self, width: int, height: int, fps: int, weights: str) -> None:
        self.camera = DepthCamera(width, height, fps)
        self.model = YoloModel(weights)
        self.results = None
        self.frame = None
        color_recognition.training(TRAINING_DATA_DIR, TRAINING_DATA_FILE)
        self.color_classifier = color_recognition.KnnClassifier(TRAINING_DATA_FILE)

    def update(self) -> None:
        self.camera.update_frame()
        self.frame = self.camera.get_color_frame_as_ndarray()
        self.results = self.model.predict_frame(self.frame) if self.frame is not None else None

    def get_classes_names(self) -> list[str]:
        return list(self.model.classes_names.values())

    def get_id_from_class_name(self, class_name: str) -> int | None:
        return self.model.classes_ids.get(class_name)

    def find_object_by_name_and_color(self, class_name: str, color_name: str) -> Coords3D | None:
        index = self.get_id_from_class_name(class_name)

        if index is None:
            return None

        return self.find_object_by_id_and_color(index, color_name)

    def find_object_by_id_and_color(self, class_id: int, color_name: str) -> Coords3D | None:
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
            test_histogram = color_recognition.color_histogram_of_test_image(imanip.extract_area_from_image(self.frame, bb_box[0], bb_box[1], bb_box[2], bb_box[3]))
            if self.color_classifier.predict(test_histogram) != color_name:
                continue
            coords, _ = self.camera.get_coords_of_object_xyxy(self.results.get_box_coords(i))
            if coords is None:
                continue
            max_coords = coords

        return max_coords

    def find_object_by_name(self, class_name: str) -> Coords3D | None:
        index = self.model.classes_ids.get(class_name.strip())

        return self.find_object_by_id(index) if index is not None else None

    def find_object_by_id(self, class_id: int) -> Coords3D | None:
        if self.results is None:
            return None

        max_conf = 0.0
        max_coords = None
        for i in range(self.results.results_count()):
            if self.results.get_class_id(i) != class_id:
                continue
            if self.results.get_conf(i) < max_conf:
                continue
            coords, _ = self.camera.get_coords_of_object_xyxy(self.results.get_box_coords(i))
            if coords is None:
                continue
            max_coords = coords

        return max_coords

    def terminate(self) -> None:
        self.camera.terminate()
