import os
from typing import Dict, List, Optional, Set, Tuple
import pymupdf
import warnings
import re

from pdf_image_rendering import compute_image_hash, page_has_visuals, render_page_snapshot


# ==========================================
# 3. CONTEXT & MANIFEST HELPERS
# ==========================================

def flush_pending_images_to_manifest(
    manifest_list: List[dict],
    pending_images: List[dict],
    chapter_name: str,
    section_context: str
) -> None:
    """Attaches accumulated section context to queued images and pushes them to manifest."""
    for img in pending_images:
        manifest_list.append({
            "image_id": img["id"],
            "image_path": img["path"],
            "chapter": chapter_name,
            "surrounding_context": section_context
        })


def check_and_queue_visual(
    page: pymupdf.Page, chapter_name: str, images_dir: str,
    registry: Dict[str, dict], seen_ids: Set[str], pending_images: List[dict],
    found_captions: list[dict] # <-- Now expects a list of dictionaries
) -> Optional[str]:
    
    if not found_captions:
        return None

    pix = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0))
    png_bytes = pix.tobytes("png")
    img_hash = compute_image_hash(png_bytes)
    unique_img_id = f"IMG_{chapter_name}_{img_hash}"
    
    seen_ids.add(unique_img_id)

    if unique_img_id in registry:
        return f"\n<!-- VLM_SUMMARY:{unique_img_id} -->\n"

    img_path = os.path.join(images_dir, f"{unique_img_id}.png")
    with open(img_path, "wb") as f:
        f.write(png_bytes)

    # Attach the structured metadata!
    pending_images.append({
        "id": unique_img_id, 
        "path": img_path,
        "figures_metadata": found_captions # Contains [{"figure_id": "...", "caption_text": "..."}]
    })
    
    return f"\n<!-- VLM_SUMMARY:{unique_img_id} -->\n"

# GARBAGE COLLECTION / ORPHAN PRUNING

def prune_orphaned_images(
    registry: Dict[str, dict],
    processed_chapters: Set[str],
    seen_ids: Set[str],
    images_dir: str
) -> int:
    """
    Removes visual records and physical PNG files for images that were deleted or modified
    in the chapters processed during this run.
    """
    orphans_removed = 0

    for existing_id in list(registry.keys()):
        # Check if the existing ID belongs to any chapter processed in this execution
        is_in_processed_scope = any(
            existing_id.startswith(f"IMG_{chap}_") for chap in processed_chapters
        )

        if is_in_processed_scope and existing_id not in seen_ids:
            # 1. Delete registry entry
            del registry[existing_id]

            # 2. Clean up disk file
            file_to_remove = os.path.join(images_dir, f"{existing_id}.png")
            if os.path.exists(file_to_remove):
                os.remove(file_to_remove)

            orphans_removed += 1

    return orphans_removed


def sanitize_pdf_text(text: str, chapter_name: str, page_index: int) -> str:
    """
    Detects and removes corrupted Unicode characters (e.g., XXXX or ).
    Raises a warning if detected, indicating lost PDF font mapping data.

    """
    # Regex targets literal strings like "\u00B2" or the Unicode replacement char ""
    corruption_pattern = re.compile(r'\\u[0-9a-fA-F]{4}|\ufffd')
    
    if corruption_pattern.search(text):
        warnings.warn(
            f"\n[!] DATA LOSS WARNING: Unmapped Unicode characters detected in {chapter_name} (Page {page_index}). "
            f"This indicates missing font mappings in the PDF. The corrupted artifacts have been stripped.\n",
            UserWarning
        )
        # Strip the corrupted characters out of the text
        text = corruption_pattern.sub('', text)
        
    return text