from typing import TypeAlias
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np

BoundingBox: TypeAlias = tuple[int, int, int, int]
"""Type used to represent a 2D bounding box in the format (x0,y0,x1,y1)"""


class PredictResults:
    """Makes it easier to use the results of detection prediction."""

    def __init__(self, results: Results, classes_names: dict[int,
                                                             str]) -> None:
        """results: the raw Results returned by the ultralytics predict function
        classes_names: a dictionary associating a class id to its name"""

        self.results: Results = results
        self.classes_names: dict[int, str] = classes_names
        self.confidences: np.ndarray = self.results.boxes.conf.cpu().numpy(
        )  # pyright: ignore
        self.classes: np.ndarray = self.results.boxes.cls.cpu().int().numpy(
        )  # pyright: ignore
        self.boxes_coords: np.ndarray = self.results.boxes.xyxy.cpu().int(
        ).numpy()  # pyright: ignore

    def results_count(self) -> int:
        """Returns the number of objects detected by the prediction."""

        return len(self.results.boxes)  # pyright: ignore

    def get_conf(self, index: int) -> float:
        """Returns the confidence of the index-th prediction."""

        return self.confidences[index]

    def get_class_id(self, index: int) -> int:
        """Returns the class id of the index-th prediction."""

        return self.classes[index]

    def get_class_name(self, index: int) -> str:
        """Returns the class name of the index-th prediction."""

        return self.classes_names[self.get_class_id(index)]

    def get_box_coords(self, index: int) -> BoundingBox:
        """Returns the bounding box coordinates of the index-th prediction."""

        return self.boxes_coords[index]

    def render(self) -> np.ndarray:
        """Returns the RGB image on which the prediction was run, with the predicted bounding boxes displayed.
        Returns it as a 3D Numpy array."""

        return np.squeeze(self.results.plot())


class YoloModel:
    """Helps using the Ultralytics Yolo model"""

    def __init__(self, weights: str) -> None:
        """weights: the name of the weights file used by the Yolo algorithm to detect objects.
            Can be either a local file trained with the Ultralytics library, or any pretrained model (e.g. 'yolov8s.pt') as specified on Ultralytics website"""

        self.load_model(weights)
        self.results = None

    def load_model(self, weights: str) -> None:
        """Loads an AI object detection model from a weights file.

        weights: the name of the weights file used by the Yolo algorithm to detect objects.
            Can be either a local file trained with the Ultralytics library, or any pretrained model (e.g. 'yolov8s.pt') as specified on Ultralytics website"""

        self.model: YOLO = YOLO(weights).to(
            'cuda' if torch.cuda.is_available() else 'cpu')
        if type(isinstance(self.model.names, dict)):
            self.classes_names: dict[int,
                                     str] = self.model.names  # pyright: ignore
            self.classes_ids: dict[str, int] = {
                name: id
                for id, name in self.model.names.items()
            }  # pyright: ignore
        else:
            self.classes_names: dict[int, str] = {
                id: name
                for id, name in enumerate(self.model.names)
            }
            self.classes_ids: dict[str, int] = {
                name: id
                for id, name in enumerate(self.model.names)
            }

    def predict_frame(self, frame: np.ndarray, **kwargs) -> PredictResults:
        """Detects objects on a given frame.

        frame: 3D Numpy array representing a RGB image
        *kwargs: any available paramater for the ultralytics predict function
            https://docs.ultralytics.com/modes/predict/#inference-arguments 

        Returns a PredictResults object containing the results of the prediction on the image."""

        results = self.model.predict(frame, **kwargs)[0]
        self.results = PredictResults(results, self.classes_names)
        return self.results
