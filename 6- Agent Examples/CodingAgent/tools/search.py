import os
from google import genai
from google.genai import types
import logging

logger = logging.getLogger("CodebaseAgent")

def web_research(query: str) -> str:
    """Performs a live Google Search to research technical documentation."""
    print(f"--> Calling web_research (query: {query})...")
    logger.info(f"Search: Researching '{query}'...")
    
    temp_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        response = temp_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Research the following technical topic and provide a detailed summary: {query}",
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        return response.text
    except Exception as e:
        return f"Web search failed: {e}"