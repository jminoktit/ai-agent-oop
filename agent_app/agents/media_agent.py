from typing import Optional

from ..core import BaseAgent, OllamaLLM
from ..tools import ImageGenTool, VideoGenTool, ImageEditTool, VideoEditTool, CalculatorTool, BlenderTool


class MediaAgent(BaseAgent):
    def __init__(
        self,
        name: str = "MediaBot",
        system_prompt: str = (
            "You are a creative media generation assistant. "
            "You specialize in generating images, videos, and 3D animations from text descriptions. "
            "Use blender_render for 3D animations, video_gen for AI-generated videos, "
            "and image_gen for images. "
            "Be helpful, creative, and provide clear responses about the media you've generated."
        ),
        llm: Optional[OllamaLLM] = None,
    ):
        super().__init__(name, system_prompt)
        self.llm = llm or OllamaLLM()
        self.register_tools(
            ImageGenTool(),
            VideoGenTool(),
            BlenderTool(),
            ImageEditTool(),
            VideoEditTool(),
            CalculatorTool(),
        )

    def run(self, input_data: str) -> str:
        self.add_memory("user", input_data)

        tool_result = self._check_tools(input_data)
        if tool_result is not None:
            self.add_memory("tool", tool_result)
            response = tool_result
            if "error" not in response.lower() and "request received" not in response.lower():
                response = self._ask_llm()
        else:
            response = self._ask_llm()

        self.add_memory("assistant", response)
        return response

    def _check_tools(self, input_data: str) -> Optional[str]:
        lower = input_data.lower()
        for tool in self.tools.values():
            if tool.name in lower or input_data.startswith(f"{tool.name}:"):
                return tool.execute(input_data)
        return None

    def _ask_llm(self) -> str:
        history = self.get_memory(recent=20)
        return self.llm.generate(history, self.system_prompt)

    def reset(self) -> None:
        self._clear_memory()
        self.stop()