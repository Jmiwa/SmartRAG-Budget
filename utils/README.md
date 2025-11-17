# Data Preparation Scripts (Reference Only)

The Python scripts in this directory (`imagetojson.py`, `calculator_gemini.py`, `dbjson.py`) are the utility scripts that were used to process the original source data (private CSVs) and generate the final data used by the main evaluation scripts.

**⚠️ Important: These scripts are for reference and transparency only.**

You **do not need to run** these scripts to reproduce the paper's results (which are compared against the `../expected_outputs/` directory).

These scripts were designed to run on a large, private dataset (`merged_csv_data.csv`) that is not included in this repository. Running them on the included `sample_source_data.csv` will not reproduce the full dataset.

The main evaluation scripts in the root directory (e.g., `evaluation_gemini_rag.py`) run using the pre-generated sample data located in the `../dbjson/` and `../userinputs_test/` folders.

---

## Data Pipeline Overview

The data was generated in the following sequence:

1.  **`imagetojson.py`**
    * **Purpose:** Reads household budget images (from URLs in the CSV) and uses `gemini-2.5-flash` to perform OCR, outputting structured financial data (income, expenses, etc.) into the `imagejson/` directory.
    * **Example Output:** See the `financial_data` key within the sample JSON files in the `../dbjson/` directory.

2.  **`calculator_gemini.py`**
    * **Purpose:** Reads both the article text (`text_data`) and the images from the CSV. It uses `gemini-2.5-flash` to analyze the expert's advice within the text and calculates the specific reduction proposals (e.g., "reduce insurance by 5,000 JPY"), outputting them to the `outputs/` directory.
    * **Example Output:** See the `expert_reduction` key within the sample JSON files in the `../dbjson/` directory.

3.  **`dbjson.py`**
    * **Purpose:** The final assembly script. It combines the CSV data (like `consultation_text`), the OCR results from `imagejson/`, and the reduction proposals from `outputs/` to create the final, clean JSON files in the `../dbjson/` directory. This output is used by `../pinecone_loader.py`.

## Included Sample Data

* `sample_source_data.csv`
    * This is a 3-row sample from our original `merged_csv_data.csv`. Its purpose is **only to show the data schema** (column structure) that these utility scripts expected.

## Note on Prompts

The prompts (e.g., `PROMPT = ...`) within `imagetojson.py` and `calculator_gemini.py` are **intentionally left in the original Japanese**. They are the research artifacts used to generate the dataset and demonstrate the exact logic given to the model.