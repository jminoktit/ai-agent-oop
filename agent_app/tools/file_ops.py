from pathlib import Path

from ..core import BaseTool


class FileOpsTool(BaseTool):
    def __init__(self, work_dir: str = "."):
        super().__init__(
            name="file_ops",
            description="Read/write/list files. Usage: file_ops: read path/to/file",
        )
        self.work_dir = Path(work_dir).resolve()

    def execute(self, input_data: str = "") -> str:
        cmd = self._clean(input_data)
        if not cmd:
            return "Usage: file_ops: read <path>, write <path> | <content>, or list <dir>"
        try:
            if cmd.startswith("read "):
                return self._read(cmd[5:])
            elif cmd.startswith("write "):
                return self._write(cmd[6:])
            elif cmd.startswith("list "):
                return self._list(cmd[5:])
            else:
                return f"Unknown command: {cmd}. Use read, write, or list."
        except Exception as e:
            return f"File error: {e}"

    def _read(self, path: str) -> str:
        full = self._resolve(path)
        if not full.exists():
            return f"File not found: {full}"
        if not full.is_file():
            return f"Not a file: {full}"
        return full.read_text(encoding="utf-8")

    def _write(self, arg: str) -> str:
        parts = arg.split("|", 1)
        if len(parts) != 2:
            return "Format: write <path> | <content>"
        full = self._resolve(parts[0].strip())
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(parts[1].strip(), encoding="utf-8")
        return f"Written to {full}"

    def _list(self, path: str) -> str:
        full = self._resolve(path) if path else self.work_dir
        if not full.exists():
            return f"Directory not found: {full}"
        if not full.is_dir():
            return f"Not a directory: {full}"
        items = []
        for entry in sorted(full.iterdir()):
            items.append(f"{entry.name}/" if entry.is_dir() else entry.name)
        return "\n".join(items) if items else "(empty)"

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.work_dir / p).resolve()

    def _clean(self, cmd: str) -> str:
        for prefix in ["file_ops:", "file:"]:
            if cmd.lower().startswith(prefix):
                cmd = cmd[len(prefix):]
        return cmd.strip()
