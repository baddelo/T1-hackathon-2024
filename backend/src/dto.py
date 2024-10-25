from enum import Enum
from typing import Tuple

from pydantic import BaseModel, ConfigDict


class LanguageEnum(str, Enum):
    RUSSIAN = 'rus'
    ENGLISH = 'eng'
    UNKNOWN = ''


class OutputDTO(BaseModel):
    coordinates: Tuple[Tuple[int, int], Tuple[int, int]]
    content: str = ''
    lang: LanguageEnum = LanguageEnum.UNKNOWN
    signature: bool = False

    model_config = ConfigDict(
        use_enum_values=True,
    )
