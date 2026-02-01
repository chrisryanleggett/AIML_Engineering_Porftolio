# This file provides linting functionality for various file types.
import subprocess
import os
import sys  # <--- Added this import
import logging
from pathlib import Path

logger = logging.getLogger("CodebaseAgent")

def run_linter(filepath: str) -> str:
    """Runs a language-specific linter on the given file."""
    print(f"--> Calling run_linter (path: {filepath})...")
    path = Path(filepath)
    if not path.exists():
        return f"Error: File {filepath} not found."

    ext = path.suffix.lower()
    try:
        if ext == ".py":
            # UPDATED: Use sys.executable -m to ensure we use the local flake8 installation
            cmd = [sys.executable, "-m", "flake8", "--select=E,F,W,C", str(path)]
        elif ext in [".js", ".ts", ".tsx"]:
            cmd = ["npx", "eslint", str(path), "--no-error-on-unmatched-pattern"]
        else:
            return f"Linter support for {ext} is not yet implemented."

        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Flake8 returns exit code 1 if issues are found, result.stdout will contain them
        if result.stdout:
            return f"Linter Output for {filepath}:\n{result.stdout}"
        
        # If there is no stdout but an error in stderr, report that
        if result.stderr:
            return f"Linter Error:\n{result.stderr}"
        
        return f"No syntax or linting issues found in {filepath}."
    except Exception as e:
        return f"Unexpected error during linting: {str(e)}"