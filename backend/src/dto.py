from enum import Enum
from typing import Tuple

from pydantic import BaseModel, ConfigDict


class LanguageEnum(str, Enum):
    RUSSIAN = 'rus'
    ENGLISH = 'eng'
    UNKNOWN = ''


class OutputDTO(BaseModel):
    coordinates: Tuple[Tuple[float, float], Tuple[float, float]]
    content: str = ''
    signature: bool = False
