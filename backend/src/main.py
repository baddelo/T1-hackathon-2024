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
        if not bentoml.models.list():
            self.get_model(2)
        self.model = bentoml.pytorch.load_model("test_detection")
        # lambda x: [OutputDTO(coordinates=((0, 0), (0, 0)), content="xui", lang='eng', signature=False)]

    def get_model(self, num_classes):
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        bentoml.pytorch.save_model("test_detection", model)
        return model

    @bentoml.api
    def detect(self, image: Image) -> List[OutputDTO]:
        result = self.model(image)
        return result
