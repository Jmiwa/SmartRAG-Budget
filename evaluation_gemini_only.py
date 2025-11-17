"""
Run answer generation with the Gemini API alone (without RAG) and evaluate the results.

[Workflow]
1. Load user input JSON files from the validation folder (userinputs_test).
2. Use `master_prompt_llm.txt` as the base prompt.
3. Embed each input file's contents into the master prompt to build the final prompt.
4. Send the request to the Gemini API (model specified by GEMINI_MODEL_NAME) and get the answer.
5. Save the generated answer as a JSON file to the specified results folder (results_gemini_only).

Use this script to measure baseline performance without RAG (Retrieval-Augmented Generation).
"""

import time
import os
import glob
import json
from pathlib import Path
from dotenv import load_dotenv  # type: ignore
import google.genai as genai  # type: ignore
from datetime import datetime

# ── 1. Constants and paths ──
total_start = time.time()

# --- Load environment variables from .env ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Model configuration ---
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# --- Folder configuration ---
BASE_DIR = Path(__file__).resolve().parent

# Validation input folder (20 items)
USER_INPUTS_DIR = BASE_DIR / "userinputs_test"

# Validation output folder
RESULTS_DIR = BASE_DIR / "results_gemini_only_test"

MASTER_PROMPT_PATH = BASE_DIR / "master_prompt_llm.txt"

# --- Logging ---
log_path = "gemini_only_test_run_log.txt"

# --- Limit for test runs (set None to process all) ---
# Process every file in the folder (20 files)
PROCESSING_LIMIT = None
# PROCESSING_LIMIT = 3


# ── 2. Logging helper and Gemini client setup ──


def log(message):
    """Write logs to both file and console."""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    text = f"{timestamp} {message}"
    print(text)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


log("=== Batch start (LLM Only Test) ===")

if not GEMINI_API_KEY:
    log("Error: GEMINI_API_KEY not found in token.env.")
    exit()

client = genai.Client(api_key=GEMINI_API_KEY)
log(f"Using model: {GEMINI_MODEL_NAME}")


# ── 3. Load master prompt ──
log(f"Loading master prompt from {MASTER_PROMPT_PATH}...")
try:
    with open(MASTER_PROMPT_PATH, "r", encoding="utf-8") as f:
        master_prompt_text = f.read()
    log("Master prompt loaded.")
except FileNotFoundError:
    log(f"Error: master prompt file not found: {MASTER_PROMPT_PATH}")
    exit()

# ── 5. File processing loop ──

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
input_files = sorted(glob.glob(str(USER_INPUTS_DIR / "*_input.json")))

if PROCESSING_LIMIT:
    log(f"\n--- Processing first {PROCESSING_LIMIT} files (Test Mode) ---")
    input_files = input_files[:PROCESSING_LIMIT]
else:
    # Process every file in 'userinputs_test'
    log(f"\n--- Processing all {len(input_files)} files (Test Mode) ---")


for input_path_str in input_files:
    input_path = Path(input_path_str)

    # Output file name
    out_name = input_path.name.replace("_input.json", "_result_only.json")
    out_path = RESULTS_DIR / out_name

    if out_path.exists():
        log(f"Skip {out_name} (already exists)")
        continue

    # Read the input JSON
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            user_input_data = json.load(f)
            # Pass financial_data as a JSON string to the prompt
            user_input_data["financial_data"] = json.dumps(
                user_input_data.get("financial_data", {}), ensure_ascii=False
            )

    except Exception as e:
        log(f"Error reading {input_path.name}: {e}. Skipping.")
        continue

    log(f"\nProcessing {input_path.name}...")
    start = time.time()

    # --- Call Gemini API ---
    try:
        final_prompt = master_prompt_text.format(**user_input_data)

        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[final_prompt],
        )

        generated_advice_str = response.text.strip()

        # --- Build output JSON ---
        result_data = {
            "input_file": input_path.name,
            "retrieved_db_id": "N/A (LLM Only test)",  # Changed from "N/A (Gemini test)"
            "retrieved_source": "N/A (LLM Only test)",  # Changed from "N/A (Gemini test)"
            "generated_advice": generated_advice_str,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start
        elapsed_int = int(elapsed)
        h, rem = divmod(elapsed_int, 3600)
        m, s = divmod(rem, 60)
        log(f"-> {out_name}: {h}h {m}m {s}s (saved)")

        # --- Sleep to avoid rate limits ---
        time.sleep(1.5)

    except Exception as e:
        msg = f"❌ Error processing {input_path.name}: {e}"
        log(msg)
        s = str(e)
        if (
            "RESOURCE_EXHAUSTED" in s
            and "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in s
        ):
            log("⛳ Daily free-tier quota hit. Stopping safely. ")
            break

# ── 6. Finish ──
total_elapsed = time.time() - total_start
total_elapsed_int = int(total_elapsed)
th, trem = divmod(total_elapsed_int, 3600)
tm, ts = divmod(trem, 60)
log(f"\n=== All files processed ===")
log(f"Total elapsed: {th}h {tm}m {ts}s")
log("=== Batch end ===\n")
