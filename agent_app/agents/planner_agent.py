import json
import re
from typing import Dict, List, Optional

from ..core import BaseAgent, OllamaLLM


class PlannerAgent(BaseAgent):
    def __init__(
        self,
        name: str = "PlannerBot",
        system_prompt: str = (
            "You are a planning agent. Break complex user requests into "
            "a sequence of steps. For each step, assign a sub-agent: "
            "chat, code, data, or research. "
            "Output JSON: {\"steps\": [{\"agent\": \"...\", \"task\": \"...\"}]}"
        ),
        llm: Optional[OllamaLLM] = None,
        orchestrator: Optional["AgentOrchestrator"] = None,
    ):
        super().__init__(name, system_prompt)
        self.llm = llm or OllamaLLM()
        self.orchestrator = orchestrator
        self._plan: List[Dict[str, str]] = []
        self._results: List[Dict[str, str]] = []

    def set_orchestrator(self, orchestrator: "AgentOrchestrator") -> None:
        self.orchestrator = orchestrator

    def run(self, input_data: str) -> str:
        self.add_memory("user", input_data)
        self._plan = self._create_plan(input_data)
        self.add_memory("system", f"Plan: {json.dumps(self._plan, ensure_ascii=False)}")

        self._results = []
        for step in self._plan:
            agent_name = step.get("agent", "chat")
            task = step.get("task", "")
            if self.orchestrator and agent_name in self.orchestrator.list_agents():
                step_result = self.orchestrator.run(agent_name, task, None)
            else:
                step_result = f"No orchestrator or unknown agent '{agent_name}'"
            self._results.append({"agent": agent_name, "task": task, "result": step_result})
            self.add_memory("tool", f"[{agent_name}] {step_result}")

        final = self._synthesize()
        self.add_memory("assistant", final)
        return final

    def _create_plan(self, input_data: str) -> List[Dict[str, str]]:
        prompt = (
            f"Given the user request, create a step-by-step plan. "
            f"Available agents: chat (general), code (programming), "
            f"data (data analysis), research (web search).\n"
            f"User: {input_data}\n"
            f"Respond with JSON only: {{\"steps\": [{{\"agent\": \"...\", \"task\": \"...\"}}]}}"
        )
        response = self.llm.generate([{"role": "user", "content": prompt}], self.system_prompt)
        match = re.search(r"\{.*?\}", response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return data.get("steps", [])
            except json.JSONDecodeError:
                pass
        return [{"agent": "chat", "task": input_data}]

    def _synthesize(self) -> str:
        parts = [f"**Plan executed in {len(self._results)} steps:**\n"]
        for i, r in enumerate(self._results, 1):
            parts.append(f"**Step {i}** ({r['agent']}): {r['task']}")
            parts.append(f"```\n{r['result'][:300]}\n```")
        combined = "\n".join(parts)

        prompt = (
            f"Summarize these step results into a final cohesive answer:\n{combined}"
        )
        summary = self.llm.generate(
            [{"role": "user", "content": prompt}],
            "You synthesize multi-step results into a clear final answer.",
        )
        return f"{summary}\n\n---\n{combined}"

    def reset(self) -> None:
        self._clear_memory()
        self._plan = []
        self._results = []
        self.stop()
