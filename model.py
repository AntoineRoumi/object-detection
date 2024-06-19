import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np

class PredictResults:
    def __init__(self, results: Results, classes_names: list[str]) -> None:
        self.results: Results = results
        self.classes_names: list[str] = classes_names
        self.confidences: np.ndarray = self.results.boxes.conf.cpu().numpy() # pyright: ignore
        self.classes: np.ndarray = self.results.boxes.cls.cpu().int().numpy() # pyright: ignore
        self.boxes_coords: np.ndarray = self.results.boxes.xyxy.cpu().int().numpy() # pyright: ignore

    def results_count(self) -> int:
        return len(self.results.boxes) # pyright: ignore

    def get_conf(self, index: int) -> float:
        return self.confidences[index]

    def get_class_id(self, index: int) -> int:
        return self.classes[index]

    def get_class_name(self, index: int) -> str:
        return self.classes_names[self.get_class_id(index)]
    
    def get_box_coords(self, index: int) -> tuple[int, int, int, int]:
        return self.boxes_coords[index]

    def render(self) -> np.ndarray:
        return np.squeeze(self.results.plot())


class YoloModel:
    def __init__(self, weights: str) -> None:
        self.load_model(weights)
        self.results = None

    def load_model(self, weights: str) -> None:
        self.model: YOLO = YOLO(weights).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes: list[str] = self.model.names

    def predict_frame(self, frame, **kwargs) -> PredictResults:
        image = np.asanyarray(frame.get_data())
        results = self.model.predict(image, **kwargs)[0]
        self.results = PredictResults(results, self.classes)
        return self.results

