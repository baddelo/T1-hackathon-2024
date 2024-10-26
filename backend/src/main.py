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
        self.model = YOLO('trained_models/yolov8_weights.pt')
        # lambda x: [OutputDTO(coordinates=((0, 0), (0, 0)), content="xui", lang='eng', signature=False)]

    def convert_output_to_dto(self, predictions) -> List[OutputDTO]:
        output_dtos = []
        for pred in predictions:
            # Извлечение координат (x_min, y_min, x_max, y_max)
            x_min, y_min, x_max, y_max = pred.xyxy[0].tolist()  # Извлекаем координаты из предсказания
            coordinates = ((x_min, y_min), (x_max, y_max))
            # Извлечение других данных
            signature = bool(pred.signature)  # Предполагаем, что есть поле signature в предсказаниях
            # Создание экземпляра OutputDTO
            output_dto = OutputDTO(coordinates=coordinates, signature=signature)
            output_dtos.append(output_dto)

        return output_dtos


    @bentoml.api
    def detect(self, image: Image) -> List[OutputDTO]:
        result = self.model(image)
        return self.convert_output_to_dto(result)
