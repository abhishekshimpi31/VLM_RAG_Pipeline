import hashlib
import re
import pymupdf


# ==========================================
# 2. PDF & IMAGE RENDERING UTILITIES
# ==========================================

def page_has_visuals(page: pymupdf.Page) -> bool:
    """Detects if a page contains raster images or vector graphics/drawings."""
    has_raster = len(page.get_images(full=True)) > 0
    has_vector = len(page.get_drawings()) > 0
    return has_raster or has_vector


def render_page_snapshot(page: pymupdf.Page, zoom: float = 2.0) -> bytes:
    """Renders a high-resolution screenshot of a page and returns raw PNG bytes."""
    matrix = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    return pix.tobytes("png")

def compute_image_hash(image_bytes: bytes, length: int = 12) -> str:
    """Generates a deterministic SHA-256 fingerprint from image byte data."""
    return hashlib.sha256(image_bytes).hexdigest()[:length]

def get_chapter_details(raw_header: str) -> tuple[str, str]:
    """
    Parses a markdown header to return a readable folder name and a unique hash.
    Example: "## Chapter 4: Future Climate" -> ("Chapter_4_Future_Climate", "8f3a9b21")
    """
    # Remove the '## ' from the string
    raw_title = re.sub(r"^##\s+", "", raw_header).strip()
    
    # 1. Generate the hash to use as an internal unique ID
    chap_hash = hashlib.md5(raw_title.encode('utf-8')).hexdigest()[:8]
    
    # 2. Extract the number and name for the folder
    match = re.match(r"^(?:Chapter\s+)?(\d+)?[\s:\-\.]*(.+)", raw_title, re.IGNORECASE)
    if match:
        chap_num = match.group(1)
        chap_title = match.group(2).strip()
        
        # Strip illegal characters for Windows/Mac/Linux folders
        safe_title = re.sub(r'[\\/*?:"<>|]', "", chap_title)
        safe_title = re.sub(r'\s+', "_", safe_title)
        
        folder_name = f"Chapter_{chap_num}_{safe_title}" if chap_num else safe_title
    else:
        # Fallback if the header is weird
        safe_title = re.sub(r'[\\/*?:"<>|]', "", raw_title)
        folder_name = re.sub(r'\s+', "_", safe_title.strip())

    print(folder_name, chap_num, chap_hash)
        
    return folder_name, chap_hash