from typing import Optional

from ..core import BaseAgent, OllamaLLM
from ..tools import DataAnalysisTool


class DataAgent(BaseAgent):
    def __init__(
        self,
        name: str = "DataBot",
        system_prompt: str = "You are a data analysis assistant.",
        llm: Optional[OllamaLLM] = None,
    ):
        super().__init__(name, system_prompt)
        self.llm = llm or OllamaLLM()
        self.register_tool(DataAnalysisTool())

    def run(self, input_data: str) -> str:
        self.add_memory("user", input_data)

        if "data_analysis:" in input_data or "data:" in input_data:
            tool = self.get_tool("data_analysis")
            if tool:
                result = tool.execute(input_data)
                self.add_memory("tool", result)
                return result

        if "load " in input_data and any(ext in input_data for ext in [".csv", ".json"]):
            tool = self.get_tool("data_analysis")
            if tool:
                path = input_data.split("load ", 1)[1].strip().split()[0]
                result = tool.execute(f"load {path}")
                self.add_memory("tool", result)
                response = self.llm.generate(self.get_memory(recent=10), self.system_prompt)
                self.add_memory("assistant", response)
                return response

        response = self.llm.generate(self.get_memory(recent=10), self.system_prompt)
        self.add_memory("assistant", response)
        return response

    def reset(self) -> None:
        self._clear_memory()
        self.stop()
