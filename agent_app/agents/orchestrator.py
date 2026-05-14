from typing import Dict, List, Optional

from ..core import BaseAgent, OllamaLLM
from .chat_agent import ChatAgent
from .code_agent import CodeAgent
from .data_agent import DataAgent
from .research_agent import ResearchAgent
from .planner_agent import PlannerAgent
from ..tools import CalculatorTool, WebSearchTool, KeyboardConverter


class AgentOrchestrator:
    def __init__(self, llm: Optional[OllamaLLM] = None):
        self.llm = llm or OllamaLLM()
        self.agents: Dict[str, BaseAgent] = {}
        self._active_agent: Optional[str] = None

    def create_default_agents(self) -> None:
        chat = ChatAgent(name="ChatBot", llm=self.llm)
        chat.register_tools(CalculatorTool(), WebSearchTool(), KeyboardConverter())
        self.agents["chat"] = chat

        code = CodeAgent(name="CodeBot", llm=self.llm)
        code.register_tool(CalculatorTool())
        self.agents["code"] = code

        data = DataAgent(name="DataBot", llm=self.llm)
        data.register_tool(CalculatorTool())
        self.agents["data"] = data

        research = ResearchAgent(name="ResearchBot", llm=self.llm)
        research.register_tool(CalculatorTool())
        self.agents["research"] = research

        planner = PlannerAgent(name="PlannerBot", llm=self.llm, orchestrator=self)
        self.agents["planner"] = planner

        self._active_agent = "chat"

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        return self.agents.get(name)

    def list_agents(self) -> List[str]:
        return list(self.agents.keys())

    def run(self, agent_name: str, input_data: str, conversation_id: Optional[int] = None) -> str:
        agent = self.get_agent(agent_name)
        if not agent:
            return f"Agent '{agent_name}' not found."
        if conversation_id:
            agent.conversation_id = conversation_id
        self._active_agent = agent_name
        agent.start()
        return agent.run(input_data)

    def auto_route(self, input_data: str, conversation_id: Optional[int] = None) -> str:
        lower = input_data.lower()
        complex_keywords = ["plan:", "multi:", "then ", "first ", "step ", "compare", "combine"]
        if any(kw in lower for kw in complex_keywords) or lower.count(" and ") > 1:
            target = "planner"
        elif any(kw in lower for kw in ["code_exec:", "exec:", "file_ops:", "read ", "write "]):
            target = "code"
        elif any(kw in lower for kw in ["data_analysis:", "data:", ".csv", ".json", "load "]):
            target = "data"
        elif any(kw in lower for kw in ["web_search:", "search:", "research", "summarize"]):
            target = "research"
        else:
            target = "chat"
        self._active_agent = target
        return self.run(target, input_data, conversation_id)

    @property
    def active_agent(self) -> Optional[str]:
        return self._active_agent

    @active_agent.setter
    def active_agent(self, name: str) -> None:
        if name in self.agents:
            self._active_agent = name

    def to_dict(self) -> Dict:
        return {
            "active_agent": self._active_agent,
            "agents": {n: a.to_dict() for n, a in self.agents.items()},
        }
