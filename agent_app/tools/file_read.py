import json
from typing import Optional


class FileReadTool:
    def __init__(self):
        self.name = "file_read"
        self.description = "Read content from user's uploaded files. Use to answer questions about uploaded documents, code, data, etc."

    def execute(self, input_data: str = "") -> str:
        try:
            data = json.loads(input_data) if input_data.startswith("{") else {"query": input_data}
        except json.JSONDecodeError:
            data = {"query": input_data}

        query = data.get("query", input_data).lower()
        return json.dumps({
            "type": "info",
            "content": "To read your uploaded files, please use the file browser in the chat interface to select a file, or upload new files using the upload button. I can help analyze text files, code, CSV data, and more.",
            "files_accessible": "Use the file manager on the left side to browse and select uploaded files."
        })

    def can_handle(self, input_data: str) -> bool:
        keywords = ["file", "upload", "document", "read", "pdf", "txt", "csv", "code", "data"]
        return any(kw in input_data.lower() for kw in keywords)