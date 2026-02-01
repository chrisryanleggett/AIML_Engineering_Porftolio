import os
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger("CodebaseAgent")

def write_to_file(filepath: str, content: str) -> str:
    """Overwrites a file with atomic replacement, deep debugging, and verification."""
    
    # --- UNCLOBBERED FIREWALL ---
    print(f"\n[!] TOOL LEVEL BLOCK: {filepath}")
    preview = content[:50].replace('\n', ' ')
    print(f"[DEBUG 1] Content Length: {len(content)} characters")
    print(f"[DEBUG 2] Content Preview: {preview}...") 
    
    confirm = input(f"CONFIRM WRITE? (y/n): ").strip().lower()
    if confirm != 'y':
        return "ERROR: User blocked write."
    # ----------------------------

    path = Path(filepath).resolve()
    print(f"[DEBUG 3] Resolved Absolute Path: {path}")

    try:
        # Check permissions before writing
        if path.exists() and not os.access(path, os.W_OK):
            return f"ERROR: No write permission for {path}"

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # --- ATOMIC WRITE LOGIC ---
        # 1. Create a temp file in the same directory as the target
        # 2. Write content and force sync to disk
        # 3. Rename/Replace the target file (Triggers IDE File Watcher)
        print(f"[DEBUG 4] Attempting atomic write via temp file...")
        
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, encoding='utf-8') as tf:
            tf.write(content)
            tf.flush()
            os.fsync(tf.fileno())
            temp_path = tf.name

        # Swap the files
        os.replace(temp_path, path)
        
        # --- VERIFICATION STEP ---
        if path.exists():
            actual_size = path.stat().st_size
            print(f"[DEBUG 5] Write complete. File size on disk: {actual_size} bytes")
            
            # Read back immediately to verify
            verification_read = path.read_text(encoding="utf-8")
            
            # FINAL PROOF: Read the first line back from the physical disk
            first_line = verification_read.splitlines()[0] if verification_read else "[EMPTY]"
            print(f"[DEBUG 6] Actual First Line on Disk: {first_line}")

            if verification_read.strip() == content.strip():
                return f"Successfully updated {filepath} (Verified)."
            else:
                return f"Error: Write mismatch. Disk data differs from memory data."
        else:
            return f"Error: File {filepath} disappeared after write attempt."
            
    except Exception as e:
        # Cleanup temp file if an error occurred during write
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"[DEBUG ERROR] Exception during write: {str(e)}")
        return f"Error writing to file: {str(e)}"