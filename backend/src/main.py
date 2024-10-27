import os
import uuid
from typing import List

from PIL.Image import Image
import bentoml

from src.dto import OutputDTO
with bentoml.importing():
    import torch
    from src.model.ocr_transformer import (
        TransformerModel,
        ALPHABET,
        HIDDEN,
        ENC_LAYERS,
        DEC_LAYERS,
        N_HEADS,
        DROPOUT,
        DEVICE,
        prediction
    )
    from torch import tensor
    from ultralytics.engine.results import Results
    from ultralytics.utils.tal import TaskAlignedAssigner


@bentoml.service(
    resources={"gpu": 1},
    traffic={"timeout": 10},
    http={
        "cors": {
            "enabled": True,
            "access_control_allow_origins": ['*', "http://192.168.43.188"],
            "access_control_allow_methods": ["GET", "OPTIONS", "POST", "HEAD", "PUT"],
            "access_control_allow_credentials": True,
            "access_control_allow_headers": ["*"],
            "access_control_max_age": 1200,
            "access_control_expose_headers": ["Content-Length"]
        }
    }
)
class TextDetector:
    def __init__(self) -> None:
        from ultralytics import YOLO
        os.mkdir('/home/bentoml/crops')
        self.model_detection = YOLO('trained_models/yolov8_weights_v2.pt')
        self.model_recognition = TransformerModel(
            len(ALPHABET),
            hidden=HIDDEN,
            enc_layers=ENC_LAYERS,
            dec_layers=DEC_LAYERS,
            nhead=N_HEADS,
            dropout=DROPOUT
        ).to(DEVICE)
        self.model_recognition.load_state_dict(
            torch.load('trained_models/checkpoint_132.pt', map_location=DEVICE)
        )

    def convert_output_to_dto(self, predictions, valid_boxes: list[int]) -> List[OutputDTO]:
        output_dtos = []
        pred = predictions[0]

        # Извлечение координат (x_min, y_min, x_max, y_max)
        for i, (box, signature) in enumerate(zip(pred.boxes.xyxy, pred.boxes.cls)):
            if i not in valid_boxes:
                continue
            x_min, y_min, x_max, y_max = box.tolist()
            coordinates = ((x_min, y_min), (x_max, y_max))
            # Извлечение других данных
            signature = True if pred.names[signature.item()] == 'signature' else False
            # Создание экземпляра OutputDTO
            output_dto = OutputDTO(coordinates=coordinates, signature=signature)
            output_dtos.append(output_dto)
        return output_dtos

    @bentoml.api
    def detect(self, image: Image):
        aligned = TaskAlignedAssigner(topk=13, num_classes=2, alpha=1.0, beta=6.0, eps=1e-09)

        detection_result = self.model_detection(image)
        pred: Results = detection_result[0]
        valid_boxes = []
        for index1, (box1, conf1) in enumerate(zip(pred.boxes.xyxy, pred.boxes.conf)):
            for index2, (box2, conf2) in enumerate(zip(pred.boxes.xyxy, pred.boxes.conf)):
                if index1 == index2:
                    continue
                iou = aligned.iou_calculation(box1, box2)
                print(f'{iou =}')
                if iou > tensor([0.5]):
                    print(f"{conf1 = } >= {conf2 = }")
                    if conf1 >= conf2:
                        valid_boxes.append(index1)
                    else:
                        valid_boxes.append(index2)

        dto_list = self.convert_output_to_dto(detection_result)
        text_dto_list = [dto for dto in dto_list if not dto.signature]
        cropped_images = []
        for dto in text_dto_list:
            cropped_image = image.crop(
                (dto.coordinates[0][0], dto.coordinates[0][1], dto.coordinates[1][0], dto.coordinates[1][1])
            )
            cropped_image.convert('RGB').save(f'/home/bentoml/crops/{uuid.uuid4().hex}.jpg')
            cropped_images.append(cropped_image)
        with torch.no_grad():
            result = prediction(self.model_recognition, cropped_images, ALPHABET)
        for dto, content in zip(text_dto_list, result):
            dto.content = content
        return dto_list
