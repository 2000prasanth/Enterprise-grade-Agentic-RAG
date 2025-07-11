import requests
#from crewai_tools import BaseTool
from crewai.tools import BaseTool
from typing import Optional

class SearxngTool(BaseTool):
    name: str = "SearXNG Local Search"
    description: str = "Search the web using locally hosted SearXNG on http://localhost:8080"

    def _run(self, query: str) -> str:
        try:
            params = {
                "q": query,
                "format": "json",
                "language": "en",
                "engines": "google,bing,duckduckgo"
            }
            response = requests.get("http://localhost:8080/search", params=params)
            results = response.json().get("results", [])
            if not results:
                return "No search results found."
            return "\n\n".join(f"{r['title']} - {r['url']}\n{r.get('content', '')}" for r in results[:3])
        except Exception as e:
            return f"Error fetching results: {e}"