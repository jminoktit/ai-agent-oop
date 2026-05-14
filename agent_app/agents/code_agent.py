from typing import Optional

from ..core import BaseAgent, OllamaLLM
from ..tools import CodeExecTool, FileOpsTool


class CodeAgent(BaseAgent):
    def __init__(
        self,
        name: str = "CodeBot",
        system_prompt: str = "You are an expert programming assistant.",
        llm: Optional[OllamaLLM] = None,
    ):
        super().__init__(name, system_prompt)
        self.llm = llm or OllamaLLM()
        self.register_tool(CodeExecTool())
        self.register_tool(FileOpsTool())

    def run(self, input_data: str) -> str:
        self.add_memory("user", input_data)

        for tool_name in ["code_exec", "file_ops"]:
            if tool_name in input_data.lower():
                tool = self.get_tool(tool_name)
                if tool:
                    result = tool.execute(input_data)
                    self.add_memory("tool", result)
                    return result

        response = self.llm.generate(self.get_memory(recent=10), self.system_prompt)
        self.add_memory("assistant", response)
        return response

    def reset(self) -> None:
        self._clear_memory()
        self.stop()
