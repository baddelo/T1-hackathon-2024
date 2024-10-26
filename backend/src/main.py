from typing import List, Dict, Any

import numpy as np
from PIL.Image import Image
import bentoml

from src.dto import OutputDTO


@bentoml.service(
    resources={"gpu": 1},
    traffic={"timeout": 10},
)
class TextDetector:
    def __init__(self) -> None:
        from ultralytics import YOLO
        self.model = YOLO('trained_models/yolov8_weights_v2.pt')

    def convert_output_to_dto(self, predictions) -> List[OutputDTO]:
        output_dtos = []

        pred = predictions[0]
        # Извлечение координат (x_min, y_min, x_max, y_max)
        for box, signature in zip(pred.boxes.xyxy, pred.boxes.cls):
            x_min, y_min, x_max, y_max = box.tolist()
            coordinates = ((x_min, y_min), (x_max, y_max))
            # Извлечение других данных
            signature = True if pred.names[signature.item()] == 'signature' else False
            # Создание экземпляра OutputDTO
            output_dto = OutputDTO(coordinates=coordinates, signature=signature)
            output_dtos.append(output_dto)
        return output_dtos


    @bentoml.api
    def detect(self, image: Image) -> List[OutputDTO]:
        result = self.model(image)
        return self.convert_output_to_dto(result)
