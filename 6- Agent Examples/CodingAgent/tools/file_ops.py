from pathlib import Path
import logging

logger = logging.getLogger("CodebaseAgent")

def list_files(directory: str = ".") -> str:
    """Returns a full file tree and prints it to terminal immediately."""
    logger.info(f"Tool Action: Scanning {directory}")
    output = []
    skip = {".git", "__pycache__", "node_modules", ".venv", "venv", ".env", "dist", "build"}
    
    def walk(path, indent=""):
        try:
            for item in sorted(Path(path).iterdir()):
                if item.name in skip or item.name.startswith("."): continue
                line = f"{indent}|-- {item.name}" if item.is_file() else f"{indent}[DIR] {item.name}"
                output.append(line)
                if item.is_dir():
                    walk(item, indent + "|   ")
        except Exception as e:
            output.append(f"{indent}[Error: {e}]")
            
    tree_text = "\n".join(output) if output else "Directory is empty."
    
    # --- FORCED TERMINAL OUTPUT ---
    print("\n" + "="*20 + " SYSTEM: RAW FILE TREE " + "="*20)
    print(tree_text)
    print("="*55 + "\n")
    
    return tree_text

def read_file(filepath: str) -> str:
    """Reads content from a file (limit 30k chars)."""
    logger.info(f"Tool Action: Reading {filepath}")
    try:
        content = Path(filepath).read_text(errors="ignore")
        return content[:30000] if len(content) > 30000 else content
    except Exception as e:
        return f"Error: {e}"