import sys
import traceback
from io import StringIO

from ..core import BaseTool


class CodeExecTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="code_exec",
            description="Execute Python code safely. Usage: code_exec: print('hello')",
        )

    def execute(self, input_data: str = "") -> str:
        c = self._clean(input_data)
        if not c:
            return "Provide code, e.g. code_exec: print(sum(range(10)))"

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            compiled = compile(c, "<agent_code>", "exec", flags=0)
            local_vars = {}
            exec(compiled, {"__builtins__": __builtins__}, local_vars)
            output = sys.stdout.getvalue()
            errors = sys.stderr.getvalue()
            result = output if output else "(no output)"
            if errors:
                result += f"\n[stderr]\n{errors}"
            return result.strip()
        except Exception:
            return traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _clean(self, code: str) -> str:
        for prefix in ["code_exec:", "exec:", "code:"]:
            if code.lower().startswith(prefix):
                code = code[len(prefix):]
        return code.strip()
