from typing import List

from PIL.Image import Image
import bentoml

from src.dto import OutputDTO


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class TextDetector:
    def __init__(self) -> None:
        self.model = self.get_model(2)

    def get_model(self, num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model


    @bentoml.api
    def detect(self, image: Image) -> List[OutputDTO]:
        result = self.model(image)
        return result
