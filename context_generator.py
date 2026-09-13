import os
from typing import Dict, List, Optional, Set, Tuple
import pymupdf
import warnings
import re

from pdf_image_rendering import compute_image_hash


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
    registry: dict, seen_ids: set, pending_images: list,
    found_captions: list[dict]
) -> str:
    
    if not found_captions:
        return None

    # 1. Snapshot and Hash
    pix = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0))
    png_bytes = pix.tobytes("png")
    img_hash = compute_image_hash(png_bytes)
    unique_img_id = f"IMG_{chapter_name}_{img_hash}"
    
    # Track the Image ID to prevent orphan deletion
    seen_ids.add(unique_img_id)
    placeholders = ""

    # 2. CACHE HIT: Image exists in the registry
    if unique_img_id in registry:
        # Loop through the nested figures in the registry
        for fig_id in registry[unique_img_id].keys():
            # Dual-ID Placeholder!
            placeholders += f"\n<!-- VLM_SUMMARY:{unique_img_id}|{fig_id} -->\n"
        return placeholders

    # 3. CACHE MISS: Save the physical image once
    img_path = os.path.join(images_dir, f"{unique_img_id}.png")
    if not os.path.exists(img_path):
        with open(img_path, "wb") as f:
            f.write(png_bytes)

    # 4. Queue distinct tasks for the VLM manifest & build placeholders
    for fig in found_captions:
        fig_id = fig["figure_id"] # e.g., "Figure 3.3"
        
        pending_images.append({
            "image_id": unique_img_id,
            "image_path": img_path,
            "figure_id": fig_id,
            "caption_text": fig["caption_text"]
        })
        
        # Dual-ID Placeholder!
        placeholders += f"\n<!-- VLM_SUMMARY:{unique_img_id}|{fig_id} -->\n"
        
    return placeholders

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