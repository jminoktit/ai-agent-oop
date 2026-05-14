import math
import operator

from ..core import BaseTool


class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Evaluate math expressions. Usage: calculator: 2 + 3 * 4",
        )

    def execute(self, input_data: str = "") -> str:
        expr = self._clean(input_data)
        if not expr:
            return "Provide an expression, e.g. calculator: 2 + 3 * 4"

        safe_ops = {
            "+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv,
            "//": operator.floordiv, "%": operator.mod, "**": operator.pow,
        }
        safe_funcs = {
            "abs": abs, "round": round, "int": int, "float": float,
            "min": min, "max": max, "sum": sum, "pow": pow,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "pi": math.pi, "e": math.e,
        }

        try:
            result = eval(expr, {"__builtins__": {}}, {**safe_ops, **safe_funcs})
            return str(result)
        except Exception as e:
            return f"Calculation error: {e}"

    def _clean(self, expr: str) -> str:
        for prefix in ["calculator:", "calc:"]:
            if expr.lower().startswith(prefix):
                expr = expr[len(prefix):]
        return expr.strip()
