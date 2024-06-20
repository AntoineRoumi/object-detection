from camera import DepthCamera, Coords3D
from model import YoloModel

class DepthFinder:
    def __init__(self, width: int, height: int, fps: int, weights: str) -> None:
        self.camera = DepthCamera(width, height, fps)
        self.model = YoloModel(weights)
        self.results = None

    def update(self) -> None:
        self.camera.update_frame()
        self.results = self.model.predict_frame(self.camera.get_color_frame())

    def get_classes_names(self) -> list[str]:
        return list(self.model.classes_names.values())

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
