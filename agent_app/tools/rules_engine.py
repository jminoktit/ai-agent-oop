import re
from typing import List, Optional, Tuple

from ..core import BaseTool
from ..models import Rule


class RuleEngine(BaseTool):
    def __init__(self):
        super().__init__(
            name="rules",
            description="Learn and apply rules. Usage: learn: when I say X reply with Y | rules: list | rules: delete <id>",
        )

    def execute(self, input_data: str = "") -> str:
        cmd = self._clean(input_data)
        if not cmd:
            return self._usage()

        if cmd.startswith("learn:") or cmd.startswith("learn "):
            return self._learn(cmd)
        elif "rules: list" in cmd.lower():
            return self._list_rules()
        elif cmd.startswith("rules: delete ") or cmd.startswith("rules delete "):
            return self._delete_rule(cmd)
        else:
            return self._usage()

    def check(self, user_input: str, agent_key: Optional[str] = None) -> Optional[str]:
        qs = Rule.objects.filter(is_active=True)
        if agent_key:
            qs = qs.filter(agent_key__in=["", agent_key, None])

        matched_input = user_input
        for rule in qs:
            if rule.trigger.lower() in user_input.lower():
                if rule.rule_type == "respond":
                    return rule.action
                elif rule.rule_type == "transform":
                    matched_input = re.sub(
                        re.escape(rule.trigger), rule.action, user_input, flags=re.IGNORECASE
                    )
                    return matched_input
        return None

    def _learn(self, cmd: str) -> str:
        text = cmd
        for p in ["learn:", "learn"]:
            if text.lower().startswith(p):
                text = text[len(p):]
        text = text.strip()

        trigger, action = self._parse_rule(text)
        if not trigger:
            return self._usage()

        rule_type = "respond"
        agent_key = None

        if " using " in action.lower():
            parts = re.split(r"\s+using\s+", action, maxsplit=1, flags=re.IGNORECASE)
            action = parts[0].strip()
            agent_spec = parts[1].strip().lower()
            possible_agents = ["chat", "code", "data", "research", "planner"]
            for a in possible_agents:
                if a in agent_spec:
                    agent_key = a
                    break

        if any(kw in text.lower() for kw in ["transform", "replace"]):
            rule_type = "transform"

        Rule.objects.create(
            trigger=trigger,
            action=action,
            rule_type=rule_type,
            agent_key=agent_key,
        )
        parts = []
        parts.append(f"[SAVED] when user says \"{trigger}\"")
        if agent_key:
            parts.append(f"using agent [{agent_key}]")
        parts.append(f"=> {action[:100]}")
        return " ".join(parts)

    def _parse_rule(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        patterns = [
            r'when\s+(?:I\s+)?say\s+["\']?(.+?)["\']?\s+(?:reply|respond|answer|do|say|run)\s+["\']?(.+?)["\']?$',
            r'when\s+["\']?(.+?)["\']?\s*(?:then|->|→)\s*(.+)',
            r'"(.+)"\s*[:=]\s*"(.+)"',
            r'if\s+(.+?)\s+(?:then|->|→)\s+(.+)',
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip(), m.group(2).strip()

        if "reply with" in text.lower():
            parts = re.split(r"reply\s+with", text, flags=re.IGNORECASE)
            if len(parts) == 2:
                t = re.sub(r"when\s+(?:I\s+)?say\s+", "", parts[0], flags=re.IGNORECASE).strip().strip('"')
                return t, parts[1].strip().strip('"')

        return None, None

    def _list_rules(self) -> str:
        qs = Rule.objects.filter(is_active=True)
        if not qs:
            return "No rules saved yet."
        lines = ["Saved rules:\n"]
        for r in qs:
            agent = f" [{r.agent_key}]" if r.agent_key else ""
            lines.append(f"  #{r.id} {r.rule_type}: \"{r.trigger}\" => \"{r.action[:50]}\"{agent}")
        return "\n".join(lines)

    def _delete_rule(self, cmd: str) -> str:
        parts = cmd.split()
        try:
            rid = int(parts[-1])
            rule = Rule.objects.get(id=rid)
            rule.delete()
            return f"[DELETED] Rule #{rid}"
        except (ValueError, IndexError):
            return "Usage: rules: delete <id>"
        except Rule.DoesNotExist:
            return f"Rule #{rid} not found"

    def _usage(self) -> str:
        return (
            "Usage:\n"
            '  learn: when I say "hello" reply with "مرحبا"\n'
            '  learn: when I say "code" using code agent\n'
            '  rules: list\n'
            '  rules: delete 1'
        )

    def _clean(self, cmd: str) -> str:
        return cmd.strip()
