from typing import Optional

from ..core import BaseAgent, OllamaLLM
from ..tools import WebSearchTool


class ResearchAgent(BaseAgent):
    def __init__(
        self,
        name: str = "ResearchBot",
        system_prompt: str = "You are a research assistant. Search the web and summarize findings.",
        llm: Optional[OllamaLLM] = None,
    ):
        super().__init__(name, system_prompt)
        self.llm = llm or OllamaLLM()
        self.register_tool(WebSearchTool())

    def run(self, input_data: str) -> str:
        self.add_memory("user", input_data)

        if "web_search:" in input_data or "search:" in input_data:
            tool = self.get_tool("web_search")
            if tool:
                search_result = tool.execute(input_data)
                self.add_memory("tool", search_result)
                response = self.llm.generate(self.get_memory(recent=10), self.system_prompt)
                self.add_memory("assistant", response)
                return response

        response = self.llm.generate(self.get_memory(recent=10), self.system_prompt)
        self.add_memory("assistant", response)
        return response

    def summarize(self, topic: str) -> str:
        self.add_memory("user", f"Research and summarize: {topic}")
        tool = self.get_tool("web_search")
        if tool:
            search_result = tool.execute(topic)
            self.add_memory("tool", search_result)
        response = self.llm.generate(self.get_memory(recent=5), self.system_prompt)
        self.add_memory("assistant", response)
        return response

    def reset(self) -> None:
        self._clear_memory()
        self.stop()
