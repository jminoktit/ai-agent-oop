from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from django.db import models

from .base_tool import BaseTool


class BaseAgent(ABC):
    def __init__(self, name: str, system_prompt: str = ""):
        self.name = name
        self.system_prompt = system_prompt
        self.tools: Dict[str, BaseTool] = {}
        self._is_running = False
        self._conversation_id: Optional[int] = None

    def register_tool(self, tool: BaseTool) -> None:
        self.tools[tool.name] = tool

    def register_tools(self, *tools: BaseTool) -> None:
        for tool in tools:
            self.register_tool(tool)

    def unregister_tool(self, tool_name: str) -> None:
        self.tools.pop(tool_name, None)

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self.tools.get(name)

    @abstractmethod
    def run(self, input_data: str) -> str:
        ...

    def reset(self) -> None:
        self._clear_memory()
        self._is_running = False

    def _clear_memory(self) -> None:
        from ..models import Message
        if self._conversation_id:
            Message.objects.filter(conversation_id=self._conversation_id).delete()

    def start(self) -> None:
        self._is_running = True

    def stop(self) -> None:
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def conversation_id(self) -> Optional[int]:
        return self._conversation_id

    @conversation_id.setter
    def conversation_id(self, value: Optional[int]) -> None:
        self._conversation_id = value

    def get_memory(self, recent: int = 20) -> List[Dict[str, str]]:
        from ..models import Message
        if not self._conversation_id:
            return []
        qs = Message.objects.filter(conversation_id=self._conversation_id).order_by("-created_at")[:recent]
        return [{"role": m.role, "content": m.content} for m in reversed(qs)]

    def add_memory(self, role: str, content: str) -> None:
        from ..models import Conversation, Message
        if not self._conversation_id:
            conv = Conversation.objects.create(agent_name=self.name)
            self._conversation_id = conv.id
        Message.objects.create(conversation_id=self._conversation_id, role=role, content=content)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "tools": self.list_tools(),
            "running": self._is_running,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, tools={self.list_tools()})"
