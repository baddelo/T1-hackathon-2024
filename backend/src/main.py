from typing import List, Dict, Any

import numpy as np
import torch
from PIL.Image import Image
import bentoml

from src.dto import OutputDTO
from src.model.ocr_transformer import (
    TransformerModel,
    ALPHABET,
    HIDDEN,
    ENC_LAYERS,
    DEC_LAYERS,
    N_HEADS,
    DROPOUT,
    DEVICE
)


@bentoml.service(
    resources={"gpu": 1},
    traffic={"timeout": 10},
)
class TextDetector:
    def __init__(self) -> None:
        from ultralytics import YOLO
        self.model_detection = YOLO('trained_models/yolov8_weights_v2.pt')
        self.model_recognition = TransformerModel(
            len(ALPHABET),
            hidden=HIDDEN,
            enc_layers=ENC_LAYERS,
            dec_layers=DEC_LAYERS,
            nhead=N_HEADS,
            dropout=DROPOUT
        ).to(DEVICE)
        # self.model_recognition.load_state_dict(torch.load(''))

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
        result = self.model_detection(image)
        return self.convert_output_to_dto(result)
