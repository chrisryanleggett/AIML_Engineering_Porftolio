"""
GeoTools to import dataset
"""

import os
from pathlib import Path
from typing import Optional, Generator
from urllib.parse import urlparse
import asyncio

# Exact constants 
DEFAULT_CHUNK_SIZE = 8 << 10  # 8192 bytes

class InvalidURLException(Exception):
    def __init__(self, url):
        self.url = url
        self.message = f"'{self.url}' is not a valid URL."
        super().__init__(self.message)

def _is_url_valid(url: str) -> bool:
    try:
        result = urlparse(url)
        # Assume it's a valid URL if a URL scheme and netloc are successfully parsed
        return all([result.scheme, result.netloc])
    except Exception:
        # If urlparse chokes on something, assume it's an invalid URL
        return False

def _is_jupyterlite() -> bool:
    """Check if running in JupyterLite environment"""
    try:
        import js
        return True
    except ImportError:
        return False

async def _get_chunks(url: str, chunk_size: int) -> Generator[bytes, None, None]:
    """
    Generator that yields consecutive chunks of bytes from URL 'url'
    :param url: The URL containing the data file to be read
    :param chunk_size: The size of each chunk (in no. of bytes).
    :returns: Generator yielding chunks of bytes from file at URL until done.
    :raise InvalidURLException: When URL is invalid.
    :raise Exception: When Exception encountered when reading from URL.
    """
    if not _is_url_valid(url):
        raise InvalidURLException(url)
    
    desc = f"Downloading {Path(urlparse(url).path).name}"
    
    if _is_jupyterlite():
        # JupyterLite environment (browser-based)
        from js import fetch  # pyright: ignore
        from pyodide.ffi import JsException  # pyright: ignore
        
        try:
            response = await fetch(url)
            reader = response.body.getReader()
            
            # Try to import tqdm, fallback if not available
            try:
                from tqdm import tqdm
                pbar = tqdm(
                    mininterval=1,
                    desc=desc,
                    total=int(response.headers.get("content-length", 0)),
                )
            except ImportError:
                pbar = None
                print(f"{desc}...")
            
            while True:
                res = (await reader.read()).to_py()
                value, done = res["value"], res["done"]
                if done:
                    break
                value = value.tobytes()
                yield value
                if pbar:
                    pbar.update(len(value))
            
            if pbar:
                pbar.close()
                
        except JsException:
            raise Exception(f"Failed to read dataset at '{url}'.") from None
    else:
        # Standard Python environment
        import requests
        from requests.exceptions import ConnectionError
        
        try:
            with requests.get(url, stream=True) as response:
                # If requests.get fails, it will return readable error
                if response.status_code >= 400:
                    raise Exception(
                        f"received status code {response.status_code} from '{url}'."
                    )
                
                # Try to import tqdm, fallback if not available
                try:
                    from tqdm import tqdm
                    pbar = tqdm(
                        miniters=1,
                        desc=desc,
                        total=int(response.headers.get("content-length", 0)),
                    )
                except ImportError:
                    pbar = None
                    print(f"{desc}...")
                
                for chunk in response.iter_content(chunk_size=chunk_size):
                    yield chunk
                    if pbar:
                        pbar.update(len(chunk))
                
                if pbar:
                    pbar.close()
                    
        except ConnectionError:
            raise Exception(f"Failed to read dataset at '{url}'.") from None

async def download(
    url: str,
    path: Optional[str] = None,
    verbose: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """
    Downloads file located at URL to path.
    """
    filename = Path(urlparse(url).path).name
    if path is None:
        path = Path.cwd() / filename
    else:
        path = Path(path)
        if path.is_dir():
            path /= filename
    
    with open(path, "wb") as f:  # Will raise FileNotFoundError if invalid path
        async for chunk in _get_chunks(url, chunk_size):
            f.write(chunk)
    
    if verbose:
        print(f"Saved as '{os.path.relpath(path.resolve())}'")

async def download_dataset(
    url: str,
    path: Optional[str] = None,
    verbose: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    return await download(url, path, verbose, chunk_size)

async def read(url: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> bytes:
    """
    Reads file at URL into bytes
    """
    return b"".join([chunk async for chunk in _get_chunks(url, chunk_size)])

# ============================================================================
# STEP-BY-STEP IMPLEMENTATION GUIDE
# ============================================================================

"""
STEP 1: Save this code as 'geotools.py' in your notebook directory

STEP 2: In your main notebook, use the import
"""