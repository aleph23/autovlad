from pydantic.dataclasses import dataclass
from lbm.config import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    input_key: str = "image"
