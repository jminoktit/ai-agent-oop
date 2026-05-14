import csv
import json

from ..core import BaseTool


class DataAnalysisTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="data_analysis",
            description="Analyze CSV/JSON data. Usage: data_analysis: load path/to/file.csv",
        )

    def execute(self, input_data: str = "") -> str:
        cmd = self._clean(input_data)
        if not cmd:
            return "Usage: data_analysis: load <path>, or analyze <path> | <query>"
        try:
            if cmd.startswith("load "):
                return self._load(cmd[5:])
            elif cmd.startswith("analyze "):
                return self._analyze(cmd[8:])
            else:
                return f"Unknown: {cmd}. Use load or analyze."
        except Exception as e:
            return f"Data error: {e}"

    def _load(self, path: str) -> str:
        path = path.strip()
        if path.endswith(".csv"):
            return self._load_csv(path)
        elif path.endswith(".json"):
            return self._load_json(path)
        else:
            return f"Unsupported format: {path}"

    def _load_csv(self, path: str) -> str:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return "Empty CSV file."
        cols = list(rows[0].keys())
        summary = f"File: {path}\nRows: {len(rows)}\nColumns: {len(cols)}\n"
        summary += f"Columns: {', '.join(cols)}\n"
        for col in cols:
            vals = [r[col] for r in rows if r[col]]
            numeric = [float(v) for v in vals if self._is_number(v)]
            if numeric:
                summary += f"  {col}: min={min(numeric):.2f}, max={max(numeric):.2f}, avg={sum(numeric)/len(numeric):.2f}\n"
            else:
                summary += f"  {col}: {len(vals)} values, {len(set(vals))} unique\n"
        return summary

    def _load_json(self, path: str) -> str:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            summary = f"File: {path}\nItems: {len(data)}\n"
            if data and isinstance(data[0], dict):
                summary += f"Keys: {', '.join(data[0].keys())}"
            return summary
        elif isinstance(data, dict):
            return f"File: {path}\nKeys: {', '.join(data.keys())}"
        return f"File: {path}\nContent: {str(data)[:500]}"

    def _analyze(self, arg: str) -> str:
        parts = arg.split("|", 1)
        if len(parts) != 2:
            return "Format: analyze <path> | <Python expression>"
        path, expr = parts[0].strip(), parts[1].strip()
        import statistics

        if path.endswith(".csv"):
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        elif path.endswith(".json"):
            with open(path, encoding="utf-8") as f:
                rows = json.load(f)
        else:
            return "Unsupported format"

        try:
            result = eval(expr, {"__builtins__": {}}, {"data": rows, "statistics": statistics, "len": len, "sum": sum, "min": min, "max": max, "float": float, "int": int, "str": str, "list": list, "dict": dict, "set": set})
            return str(result)
        except Exception as e:
            return f"Analysis error: {e}"

    def _is_number(self, val: str) -> bool:
        try:
            float(val)
            return True
        except ValueError:
            return False

    def _clean(self, cmd: str) -> str:
        for prefix in ["data_analysis:", "data:"]:
            if cmd.lower().startswith(prefix):
                cmd = cmd[len(prefix):]
        return cmd.strip()
