from typing import List, Dict, Any

import numpy as np
from PIL.Image import Image
import bentoml

from src.dto import OutputDTO


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class TextDetector:
    def __init__(self) -> None:
        from ultralytics import YOLO
        self.model = YOLO('trained_models/yolov8_weights.pt')
        # lambda x: [OutputDTO(coordinates=((0, 0), (0, 0)), content="xui", lang='eng', signature=False)]

    def convert_output_to_dto(self, output) -> List[OutputDTO]:
        result = []
        for entry in output:
            boxes = entry['boxes']
            labels = entry['labels']

            for i in range(len(boxes)):
                box = boxes[i]
                label = labels[i].item()  # Convert tensor item to a scalar

                # Format coordinates as pairs of [x, y]
                coordinates = [[box[0].item(), box[1].item()], [box[2].item(), box[3].item()]]

                # Append formatted dictionary to results
                result.append(
                    OutputDTO.model_validate(
                        {
                            "coordinates": coordinates,
                            "content": "",
                            "language": "",
                            "signature": True if label == 1 else False
                        }
                    )
                )
        return result

    @bentoml.api
    def detect(self, image: Image) -> Any:
        result = self.model(image)
        return result
        # return self.convert_output_to_dto(result)
