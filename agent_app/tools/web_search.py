import json
import urllib.parse
import urllib.request

from ..core import BaseTool


class WebSearchTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web. Usage: web_search: your query",
        )

    def execute(self, input_data: str = "") -> str:
        q = self._clean(input_data)
        if not q:
            return "Provide a query, e.g. web_search: latest AI news"

        params = urllib.parse.urlencode({
            "q": q, "format": "json", "no_html": "1", "skip_disambig": "1",
        })
        url = f"https://api.duckduckgo.com/?{params}"

        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            results = []
            for topic in data.get("RelatedTopics", []):
                if "Text" in topic:
                    results.append(topic["Text"])
                if "Topics" in topic:
                    for sub in topic["Topics"]:
                        if "Text" in sub:
                            results.append(sub["Text"])
            if not results and "AbstractText" in data and data["AbstractText"]:
                results.append(data["AbstractText"])
            return "\n\n".join(results[:5]) if results else "No results found."
        except Exception as e:
            return f"Search error: {e}"

    def _clean(self, query: str) -> str:
        for prefix in ["web_search:", "search:"]:
            if query.lower().startswith(prefix):
                query = query[len(prefix):]
        return query.strip()
