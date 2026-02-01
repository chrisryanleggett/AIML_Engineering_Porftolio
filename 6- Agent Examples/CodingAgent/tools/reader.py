# This file provides functionality for reading raw file content.
from pathlib import Path
import logging

logger = logging.getLogger("CodebaseAgent")

def read_raw_file(filepath: str) -> str:
    """Reads the exact raw content of a file."""
    print(f"--> Calling read_raw_file (path: {filepath})...")
    logger.info(f"Reader: Loading {filepath}")
    try:
        content = Path(filepath).read_text(errors="ignore")
        return content[:30000] if len(content) > 30000 else content
    except Exception as e:
        return f"Error: {e}"