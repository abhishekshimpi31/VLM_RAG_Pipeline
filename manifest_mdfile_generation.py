import hashlib
import json
import os
import re
import pymupdf
import pymupdf4llm

from context_generator import check_and_queue_visual, sanitize_pdf_text
from file_versioning import enforce_retention_policy, get_chapter_dirs, get_latest_file, get_safe_chapter_id, get_timestamped_filename, load_file_content, save_if_changed
from pdf_image_rendering import get_chapter_details


def extract_document_context(pdf_path: str, base_output_dir: str = "pipeline_data"):
    doc = pymupdf.open(pdf_path)
    page_chunks = pymupdf4llm.to_markdown(doc, page_chunks=True)

    # We just grab the whole line now; the helper function does the hard work
    chapter_regex = re.compile(r"^(##\s+.*)", re.MULTILINE)
    section_regex = re.compile(r"^###\s+(.*)", re.MULTILINE)

    # --- ACTIVE CHAPTER STATE ---
    active_folder_name = "Frontmatter"
    active_chap_hash = "frontmatter_hash"
    
    active_dirs = get_chapter_dirs(base_output_dir, active_folder_name)
    latest_reg = get_latest_file(active_dirs["registry"], "registry", ".json")
    active_registry = json.loads(load_file_content(latest_reg)) if latest_reg else {}
    
    active_seen_ids = set()
    active_md = ""
    active_section_text = ""
    active_manifest = []
    pending_section_images = []

    def save_and_close_active_chapter():
        """Flushes the current chapter to disk before moving to the next one."""
        nonlocal active_manifest, active_md, active_section_text, pending_section_images
        
        for img in pending_section_images:
            active_manifest.append({
                "image_id": img["id"], "image_path": img["path"],
                "chapter": active_folder_name, "surrounding_context": active_section_text
            })
            
        for existing_id in list(active_registry.keys()):
            if existing_id not in active_seen_ids:
                del active_registry[existing_id]
                img_path = os.path.join(active_dirs["images"], f"{existing_id}.png")
                if os.path.exists(img_path): os.remove(img_path)

        save_if_changed(active_dirs["md"], active_folder_name, ".md", active_md)
        save_if_changed(active_dirs["registry"], "registry", ".json", json.dumps(active_registry, indent=4))
        
        if active_manifest:
            manifest_content = "\n".join(json.dumps(e) for e in active_manifest) + "\n"
            manifest_filename = get_timestamped_filename("manifest", ".jsonl")
            with open(os.path.join(active_dirs["manifest"], manifest_filename), "w", encoding="utf-8") as f:
                f.write(manifest_content)

        enforce_retention_policy(active_dirs["md"])
        enforce_retention_policy(active_dirs["manifest"])
        enforce_retention_policy(active_dirs["registry"])

    print("Parsing AST and extracting visuals...")

    for page_num, chunk in enumerate(page_chunks):
        page_text = sanitize_pdf_text(chunk["text"], active_folder_name, page_num)
        
        # --- CHAPTER TRANSITION ---
        chapter_match = chapter_regex.search(page_text)
        if chapter_match:
            raw_header = chapter_match.group(1) # e.g., "## Chapter 4: Future Global Climate"
            new_folder_name, new_chap_hash = get_chapter_details(raw_header)
            
            # Use the hash to check if we are ACTUALLY transitioning to a new chapter
            if new_chap_hash != active_chap_hash:
                save_and_close_active_chapter()
                
                # Update State
                active_folder_name = new_folder_name
                active_chap_hash = new_chap_hash
                
                # Initialize new folders and load its specific registry
                active_dirs = get_chapter_dirs(base_output_dir, active_folder_name)
                latest_reg = get_latest_file(active_dirs["registry"], "registry", ".json")
                active_registry = json.loads(load_file_content(latest_reg)) if latest_reg else {}
                
                active_seen_ids = set()
                active_md = ""
                active_section_text = ""
                active_manifest = []
                pending_section_images = []

        # --- SECTION TRANSITION ---
        if section_regex.search(page_text):
            for img in pending_section_images:
                active_manifest.append({
                    "image_id": img["id"], "image_path": img["path"],
                    "chapter": active_folder_name, "surrounding_context": active_section_text
                })
            active_section_text = ""
            pending_section_images = []

        # --- ACCUMULATE TEXT & VISUALS ---
        active_md += page_text + "\n\n"
        active_section_text += page_text + "\n\n"

        # Note: We pass active_chap_hash here so your Image IDs stay short (e.g., IMG_8f3a9b21_...)
        # instead of the massively long human-readable folder name!
        placeholder = check_and_queue_visual(
            doc[page_num], active_chap_hash, active_dirs["images"],
            active_registry, active_seen_ids, pending_section_images
        )
        if placeholder:
            active_md += placeholder
            active_section_text += placeholder

    save_and_close_active_chapter()
    print("\n[✔] Enterprise Pipeline Execution Complete.")

if __name__ == "__main__":
    PDF_FILE = "data/pdf_files/IPCC_AR6_WGI_Chapter03.pdf"
    output_dir = "data/extracted_data/"
    extract_document_context(pdf_path=PDF_FILE, base_output_dir=output_dir)