from pathlib import Path
import logging
import os

logger = logging.getLogger("CodebaseAgent")

def get_file_tree(directory: str = ".") -> str:
    """Solely generates a recursive file tree. Defaults to CWD."""
    print(f"--> Calling get_file_tree (directory: {directory})...")
    target_path = Path(directory).resolve()
    
    logger.info(f"Explorer: Mapping {target_path}")
    output = []
    skip = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.env', 'dist', 'build'}
    
    def walk(path, indent=""):
        try:
            for item in sorted(path.iterdir()):
                if item.name in skip or item.name.startswith("."):
                    continue
                
                if item.is_dir():
                    output.append(f"{indent}[DIR] {item.name}")
                    walk(item, indent + "|   ")
                else:
                    output.append(f"{indent}|-- {item.name}")
        except Exception as e:
            output.append(f"{indent}[Error accessing {path.name}: {e}]")
            
    walk(target_path)
    tree = "\n".join(output)
    return tree if tree else "The directory appears to be empty or inaccessible."