from enum import Enum
from typing import Tuple

from pydantic import BaseModel, ConfigDict


class LanguageEnum(str, Enum):
    RUSSIAN = 'rus'
    ENGLISH = 'eng'


class OutputDTO(BaseModel):
    coordinates: Tuple[Tuple[int, int], Tuple[int, int]]
    content: str
    lang: LanguageEnum
    signature: bool

    model_config = ConfigDict(
        use_enum_values=True,
    )
