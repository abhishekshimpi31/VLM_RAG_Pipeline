import os
import re
import json
import glob

from file_versioning import get_latest_file

def hydrate_markdown_files(base_dir: str = "pipeline_data"):
    for chapter_folder in os.listdir(base_dir):
        chapter_path = os.path.join(base_dir, chapter_folder)
        if not os.path.isdir(chapter_path): continue
            
        registry_file = get_latest_file(os.path.join(chapter_path, "registry"), "registry", ".json")
        if not registry_file: continue
            
        with open(registry_file, "r", encoding="utf-8") as f:
            registry = json.load(f)

        md_dir = os.path.join(chapter_path, "md")
        print(md_dir)
            
        md_files = [f for f in glob.glob(os.path.join(md_dir, "*.md")) if not f.endswith("_hydrated.md")]
        
        for md_file in md_files:
            with open(md_file, "r", encoding="utf-8") as f:
                text = f.read()
            
            # 1. Strip out the PyMuPDF4LLM garbage tags 
            text = re.sub(r"<!-- Start of picture text -->.*?<!-- End of picture text -->\n?", "", text, flags=re.DOTALL)
            
            for img_id, figures in registry.items():
                for fig_id, data in figures.items():
                    
                    # 2. Delete the wandering placeholder entirely
                    placeholder = f"<!-- VLM_SUMMARY:{img_id}|{fig_id} -->"
                    text = text.replace(placeholder, "")
                    
                    # 3. Build the Blockquote
                    injection = (
                        f"\n> **[{fig_id}] Visual Data Extraction**\n"
                        f"> *Original Caption:* {data.get('caption', '')}\n>\n"
                        f"> {data.get('vlm_summary', '')}\n"
                    )
                    
                    # 4. Swap the exact paragraph
                    paragraphs = text.split("\n\n")
                    for i, p in enumerate(paragraphs):
                        # Clean the string just for the check (removes ** and spaces)
                        clean_p = p.strip().replace("**", "")
                        
                        if clean_p.startswith(fig_id):
                            # Replace the entire original caption paragraph with our VLM data
                            paragraphs[i] = injection
                            break 
                            
                    text = "\n\n".join(paragraphs)
            
            output_md = md_file.replace(".md", "_hydrated.md")
            with open(output_md, "w", encoding="utf-8") as f:
                f.write(text)
                
            print(f"[✔] Hydrated: {os.path.basename(output_md)}")

if __name__ == "__main__":
    base_directory = "data/extracted_data/"
    hydrate_markdown_files(base_directory)