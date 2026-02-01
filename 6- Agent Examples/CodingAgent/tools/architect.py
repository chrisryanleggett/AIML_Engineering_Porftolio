import os
from google import genai
import logging

logger = logging.getLogger("CodebaseAgent")

def explain_architecture(tree_data: str) -> str:
    """Explains high-level architecture based on provided tree data."""
    print("--> Calling explain_architecture...")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = f"Analyze this file tree and describe the overall software architecture pattern used:\n{tree_data}"
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text