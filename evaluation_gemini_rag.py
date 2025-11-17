"""
Run answer generation by combining RAG (Retrieval-Augmented Generation) with the Gemini API
and evaluate the results.

[Workflow]
1. Load user input JSON files from the validation folder (userinputs_test).
2. Use the provided consultation_text as a query to search the Pinecone vector store
   (INDEX_NAME) for retrieval.
3. Use `master_prompt_rag.txt` as the base prompt.
4. Inject the retrieved context (e.g., numeric data from DB) and user input (consultation
   text, numeric data) into the master prompt.
5. Send the final prompt to the Gemini API (GEMINI_MODEL_NAME) and obtain the answer
   (RAG-Generation).
6. Save the generated answer and retrieval source IDs as JSON files in the specified
   results folder (results_gemini_rag_test).

Use this script to compare against `evaluation_gemini_only.py` and measure improvements
from RAG.
"""

import time
import os
import glob
import json
from pathlib import Path
from dotenv import load_dotenv  # type: ignore
from datetime import datetime
import google.genai as genai  # type: ignore

# Reuse parts of LangChain components for RAG (retrieval)
from langchain_openai import OpenAIEmbeddings  # type: ignore
from langchain_pinecone import PineconeVectorStore  # type: ignore

# ── 1. Constants and paths ──
total_start = time.time()

# --- Load environment variables from .env ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# --- Model and DB settings ---
GEMINI_MODEL_NAME = "gemini-2.5-flash"
INDEX_NAME = "dbjsonthewebv3"  # Updated index name

# --- Folder settings ---
BASE_DIR = Path(__file__).resolve().parent

# Validation input folder (20 items)
USER_INPUTS_DIR = BASE_DIR / "userinputs_test"

# Validation output folder
RESULTS_DIR = BASE_DIR / "results_gemini_rag_test"

# RAG prompt file
MASTER_PROMPT_PATH = BASE_DIR / "master_prompt/master_prompt_rag_jp.txt"

# --- Logging ---
log_path = "gemini_rag_test_run_log.txt"

# --- Limit for test runs (None to process all) ---
PROCESSING_LIMIT = None  # Process all 20
# PROCESSING_LIMIT = 5


# ── 2. Logging helper and clients ──


def log(message):
    """Write logs to both file and console."""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    text = f"{timestamp} {message}"
    print(text)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


log("=== RAG Batch start (Gemini Test) ===")

# --- Check API keys ---
if not GEMINI_API_KEY:
    log("Error: GEMINI_API_KEY not found in token.env.")
    exit()
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    log("Error: OPENAI_API_KEY or PINECONE_API_KEY for RAG not found.")
    exit()

# --- Initialize clients ---
client = genai.Client(api_key=GEMINI_API_KEY)
log(f"Using model: {GEMINI_MODEL_NAME}")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
try:
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    log(f"Pinecone index '{INDEX_NAME}' connected.")
except Exception as e:
    log(f"Error: Failed to connect to Pinecone: {e}")
    exit()


# ── 3. Load master prompt ──
log(f"Loading master prompt from {MASTER_PROMPT_PATH}...")
try:
    with open(MASTER_PROMPT_PATH, "r", encoding="utf-8") as f:
        master_prompt_text = f.read()
    log("Master RAG prompt loaded.")
except FileNotFoundError:
    log(f"Error: master prompt file not found: {MASTER_PROMPT_PATH}")
    exit()


# ── 4. Helper functions (RAG) ──


def prepare_db_context(retrieved_docs):
    """
    Parse documents fetched from Pinecone into a dict.
    (expert_advice_text is intentionally excluded to focus on numeric data)
    """
    if not retrieved_docs:
        return {
            "id": "N/A",
            "source": "N/A",
            "financial_data": {},
            "expert_reduction": {},
        }

    doc = retrieved_docs[0]
    metadata = doc.metadata

    try:
        financial_data = json.loads(metadata.get("financial_data", "{}"))
        expert_reduction = json.loads(metadata.get("expert_reduction", "{}"))
    except json.JSONDecodeError as e:
        log(f"Error decoding metadata JSON: {e}")
        financial_data = {"error": "Failed to parse financial_data"}
        expert_reduction = {"error": "Failed to parse expert_reduction"}

    # Return only the data needed for the RAG prompt
    db_context = {
        "id": metadata.get("id"),
        "source": metadata.get("source"),
        "financial_data": financial_data,
        "expert_reduction": expert_reduction,  # Used as 'db_reduction'
    }
    return db_context


def format_prompt_data(user_input: dict, db_context: dict) -> dict:
    """
    Build a dict for .format() when sending the prompt to Gemini.
    """

    def safe_json_dumps(data):
        return json.dumps(data, ensure_ascii=False)

    prompt_data = {
        # User input
        "user_consultation": user_input.get("consultation_text", ""),
        "user_financials": safe_json_dumps(user_input.get("financial_data", {})),
        # DB context
        "db_id": db_context.get("id", "N/A"),
        "db_financials": safe_json_dumps(db_context.get("financial_data", {})),
        "db_reduction": safe_json_dumps(
            db_context.get("expert_reduction", {})
        ),  # Map 'expert_reduction' to 'db_reduction'
    }
    return prompt_data


# ── 5. File processing loop ──

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
input_files = sorted(glob.glob(str(USER_INPUTS_DIR / "*_input.json")))

if PROCESSING_LIMIT:
    log(f"\n--- Processing first {PROCESSING_LIMIT} files (Test Mode) ---")
    input_files = input_files[:PROCESSING_LIMIT]
else:
    log(f"\n--- Processing all {len(input_files)} files (Production Mode) ---")


for input_path_str in input_files:
    input_path = Path(input_path_str)

    # Output file name
    out_name = input_path.name.replace("_input.json", "_result_rag.json")
    out_path = RESULTS_DIR / out_name

    if out_path.exists():
        log(f"Skip {out_name} (already exists)")
        continue

    # 1. Read the input JSON file
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            user_input_data = json.load(f)
    except Exception as e:
        log(f"Error reading {input_path.name}: {e}. Skipping.")
        continue

    log(f"\nProcessing {input_path.name}...")
    start = time.time()

    try:
        # 2. RAG execute (Pinecone search)
        query = user_input_data.get("consultation_text", "")
        if not query:
            log(
                f"Warning: No 'consultation_text' in {input_path.name}. Skipping retrieval."
            )
            retrieved_docs = []
        else:
            retrieved_docs = retriever.invoke(query)

        # 3. Format the retrieval results
        db_context = prepare_db_context(retrieved_docs)
        log(
            f"Retrieved DB ID: {db_context.get('id')} (Source: {db_context.get('source')})"
        )

        # 4. Build prompt data
        prompt_data = format_prompt_data(user_input_data, db_context)
        final_prompt = master_prompt_text.format(**prompt_data)

        # 4.5. Count tokens and log
        try:
            token_count_response = client.models.count_tokens(contents=[final_prompt])
            log(
                f"--- [DEBUG] Total Prompt Tokens: {token_count_response.total_tokens} ---"
            )
        except Exception as e:
            log(f"Warning: Token count failed. {e}")

        # 5. Call Gemini API
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[final_prompt],
        )

        generated_advice_str = response.text.strip()

        # 6. Build output JSON
        result_data = {
            "input_file": input_path.name,
            "retrieved_db_id": db_context.get("id"),
            "retrieved_source": db_context.get("source"),
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
            break  # Stop loop

# ── 6. Finish ──
total_elapsed = time.time() - total_start
total_elapsed_int = int(total_elapsed)
th, trem = divmod(total_elapsed_int, 3600)
tm, ts = divmod(trem, 60)
log(f"\n=== All files processed ===")
log(f"Total elapsed: {th}h {tm}m {ts}s")
log("=== Batch end ===\n")
