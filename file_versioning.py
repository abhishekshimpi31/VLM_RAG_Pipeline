import hashlib
import os
import glob
import time
from datetime import datetime
from typing import Optional


# ==========================================
# 1. VERSIONING & RETENTION (TTL) UTILITIES
# ==========================================

def get_latest_file(directory: str, prefix: str, extension: str) -> Optional[str]:
    """Finds the most recently modified file matching a prefix and extension."""
    search_pattern = os.path.join(directory, f"{prefix}*{extension}")
    files = glob.glob(search_pattern)
    if not files:
        print("No files found")
        return None
    return max(files, key=os.path.getmtime)

def load_file_content(filepath: str) -> str:
    """Loads raw text from a file safely."""
    if not filepath or not os.path.exists(filepath):
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def get_timestamped_filename(prefix: str, extension: str) -> str:
    """Generates a filename with the current UTC timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{extension}"

def get_safe_chapter_id(raw_title: str) -> str:
    """Creates a clean, guaranteed OS-safe folder name using a short hash."""
    title_hash = hashlib.md5(raw_title.encode('utf-8')).hexdigest()[:8]
    return f"chap_{title_hash}"

def save_if_changed(directory: str, prefix: str, extension: str, new_content: str) -> Optional[str]:
    """
    Saves content to a new timestamped file ONLY if it differs from the latest version.
    Returns the path to the current/new file.
    """
    latest_file = get_latest_file(directory, prefix, extension)
    latest_content = load_file_content(latest_file) if latest_file else ""
    
    if new_content != latest_content:
        new_filename = get_timestamped_filename(prefix, extension)
        new_filepath = os.path.join(directory, new_filename)
        with open(new_filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"  [+] Changes detected. Created new version: {new_filename}")
        return new_filepath
    
    print(f"  [-] No changes for {prefix}. Kept latest version.")
    return latest_file

def get_chapter_dirs(base_dir: str, chapter_name: str) -> dict:
    """Creates and returns the strictly isolated subfolders for a specific chapter."""
    dirs = {
        "md": os.path.join(base_dir, chapter_name, "md"),
        "manifest": os.path.join(base_dir, chapter_name, "manifest"),
        "images": os.path.join(base_dir, chapter_name, "images"),
        "registry": os.path.join(base_dir, chapter_name, "registry"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def enforce_retention_policy(directory: str, days: int = 30) -> int:
    """Deletes files in a directory older than the specified retention period."""
    if not os.path.exists(directory):
        return 0
        
    deleted_count = 0
    cutoff_time = time.time() - (days * 86400) # 86400 seconds = 1 day
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            if os.path.getmtime(filepath) < cutoff_time:
                print(f"  [!] Retention Policy: Deleting expired file {filename}")
                os.remove(filepath)
                deleted_count += 1
                
    return deleted_count