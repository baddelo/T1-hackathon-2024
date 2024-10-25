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
        self.model = lambda x: [OutputDTO(coordinates=((0, 0), (0, 0)), content="xui", lang='eng', signature=False)]

    @bentoml.api
    def detect(self, image: Image) -> List[OutputDTO]:
        result = self.model(image)
        return result
