import torch
import warnings
import logging
import transformers
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import json
import os
import glob
import datetime
import gc
import shutil

import config
from file_versioning import get_latest_file

warnings.filterwarnings("ignore")
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()


# Define global variables to store the model and processor
_GLOBAL_MODEL = None
_GLOBAL_PROCESSOR = None

def initialize_model():
    global _GLOBAL_MODEL, _GLOBAL_PROCESSOR

    # 2. Check if they are already loaded; if yes, skip and return them
    if _GLOBAL_MODEL is not None and _GLOBAL_PROCESSOR is not None:
        print("Model and processor are already initialized. Skipping load.")
        return _GLOBAL_MODEL, _GLOBAL_PROCESSOR

    print("Loading Qwen2-VL-7B in 8-bit for A100 GPU...")

    # Define the 8-bit quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    # Load the model and assign it to the global variable
    _GLOBAL_MODEL = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation="sdpa"
    )

    # Load the processor and assign it to the global variable
    _GLOBAL_PROCESSOR = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    print("Model loaded successfully in 8-bit precision!")

    return _GLOBAL_MODEL, _GLOBAL_PROCESSOR


def process_image_batch(model, processor, batch_tasks):
    batch_messages = []

    for task in batch_tasks:
        raw_path = task["image_path"].replace("\\", "/")
        clean_path = os.path.join("/content/drive/MyDrive/Rag_Pipeline/", raw_path)
        print(f"Opening: {clean_path} for {task['figure_id']}")

        pil_img = Image.open(clean_path).convert("RGB")

        formatted_prompt = config.user_prompt.format(
            captions_text=task["caption_text"],
            image_id=task["figure_id"]
        )

        message = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": formatted_prompt}
            ]}
        ]
        batch_messages.append(message)

    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    image_inputs, _ = process_vision_info(batch_messages)

    inputs = processor(text=texts, images=image_inputs, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    outputs = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # --- BUILD THE NESTED JSON STRUCTURE ---
    results = {}
    for task, summary in zip(batch_tasks, outputs):
        img_id = task["image_id"]
        fig_id = task["figure_id"]

        if img_id not in results:
            results[img_id] = {}

        results[img_id][fig_id] = {
            "caption": task["caption_text"],
            "vlm_summary": summary.strip()
        }

    # Aggressive cleanup
    del inputs, generated_ids, generated_ids_trimmed, image_inputs
    return results


def run_vlm_router(model, processor, base_dir: str = "pipeline_data", batch_size: int = 8):
    for chapter_folder in os.listdir(base_dir):
        chapter_path = os.path.join(base_dir, chapter_folder)
        if not os.path.isdir(chapter_path):
            continue

        manifest_dir = os.path.join(chapter_path, "manifest")
        registry_dir = os.path.join(chapter_path, "registry")
        archive_dir = os.path.join(chapter_path, "archived_manifests")
        os.makedirs(archive_dir, exist_ok=True)

        manifest_files = glob.glob(os.path.join(manifest_dir, "*.jsonl"))
        if not manifest_files:
            continue

        print(f"\n[{chapter_folder}] Found {len(manifest_files)} pending manifest(s). Booting VLM...")

        latest_reg_file = get_latest_file(registry_dir, "registry", ".json")
        if latest_reg_file:
            with open(latest_reg_file, "r", encoding="utf-8") as f:
                chapter_registry = json.load(f)
        else:
            chapter_registry = {}

        for manifest_path in manifest_files:
            tasks = []
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    tasks.append(json.loads(line))

            print(f"  -> Processing {len(tasks)} images in batches of {batch_size}...")

            for i in range(0, len(tasks), batch_size):
                batch = tasks[i : i + batch_size]
                batch_results = process_image_batch(model, processor, batch)

                # --- DEEP MERGE LOGIC ---
                # Safely adds the specific figure without overwriting the whole image dict
                for img_id, figures_dict in batch_results.items():
                    if img_id not in chapter_registry:
                        chapter_registry[img_id] = {}
                    chapter_registry[img_id].update(figures_dict)
                # ------------------------

                gc.collect()
                torch.cuda.empty_cache()

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_reg_path = os.path.join(registry_dir, f"registry_{timestamp}.json")
            with open(new_reg_path, "w", encoding="utf-8") as f:
                json.dump(chapter_registry, f, indent=4)

            shutil.move(manifest_path, os.path.join(archive_dir, os.path.basename(manifest_path)))
            print(f"  [✔] Registry updated. Archived manifest.")

if __name__ == "__main__":
    # BASE_DIR = "data/extracted_data/"
    batch_size = 8
    # _GLOBAL_MODEL, _GLOBAL_PROCESSOR = initialize_model()
    # run_vlm_router(base_dir=BASE_DIR, batch_size=8, model=_GLOBAL_MODEL, processor=_GLOBAL_PROCESSOR)