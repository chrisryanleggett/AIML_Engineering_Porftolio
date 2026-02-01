# This file provides functionality to summarize the content of files and directories.
import os
from pathlib import Path
from google import genai
import logging

logger = logging.getLogger("CodebaseAgent")

def summarize_file_content(filepath: str) -> str:
    """Summarizes a single file's logic and purpose."""
    print(f"--> Calling summarize_file_content (path: {filepath})...")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        content = Path(filepath).read_text(errors="ignore")
        prompt = f"Summarize the logic of this file concisely:\n\n{content}"
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return response.text
    except Exception as e:
        return f"Analysis Error: {e}"

def summarize_directory_behavior(directory: str) -> str:
    """Analyzes a directory to explain what that specific module/folder does."""
    print(f"--> Calling summarize_directory_behavior (directory: {directory})...")
    from .explorer import get_file_tree
    tree = get_file_tree(directory)
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = f"Look at this folder structure and explain the behavioral purpose of this directory:\n{tree}"
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text