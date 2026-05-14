from abc import ABC, abstractmethod
from typing import Dict


class BaseTool(ABC):
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, input_data: str = "") -> str:
        ...

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "description": self.description}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
