
# SmartRAG-Budget

This repository provides the code and data samples necessary to reproduce the results for the paper (title TBD).

The evaluation is split into two main parts:
1.  **Baseline (LLM-Only):** Generates advice using only the Gemini model. (`evaluation_gemini_only.py`)
2.  **RAG (Retrieval-Augmented):** Generates advice by first retrieving a similar case from a Pinecone vector database. (`evaluation_gemini_rag.py`)

---

## 1. ⚙️ Setup (Configuration)

Follow these steps to set up the environment.

### Step 1: Install Requirements
Install all required Python libraries.
```bash
pip install -r requirements.txt
```

### Step 2: Set API Keys

Copy the example environment file and add your secret keys.

```bash
cp .env.example .env
```

You must then edit the `.env` file to add your API keys for:

  * `OPENAI_API_KEY` (used for embeddings)
  * `GEMINI_API_KEY` (used for generation)
  * `PINECONE_API_KEY` (used for the vector database)

-----

## 2\. 🚀 How to Run (Evaluation)

Follow these steps to run the evaluation scripts.

### Step 1: Populate Vector Database (for RAG)

Before running the RAG evaluation, you must populate your Pinecone index with the sample data.

1.  **Configure Index:** Open `pinecone_loader.py` and set `INDEX_NAME` to the Pinecone index you want to use (e.g., `INDEX_NAME = "my-test-index"`). Use the same name in Step 3 below.
2.  **Run Loader:** Execute the script. It reads the sample data from the `dbjson/` folder and upserts it to Pinecone.

<!-- end list -->

```bash
python pinecone_loader.py
```

### Step 2: Run Baseline (LLM-Only) Evaluation

This script runs the evaluation *without* RAG.

1.  **Run Script:**
    ```bash
    python evaluation_gemini_only.py
    ```
2.  **Output:** This will read inputs from `userinputs_test/` and create a **new directory** named `results_gemini_only/`. You can inspect the JSON files in this new directory to see the model's output.

### Step 3: Run RAG Evaluation

This script runs the evaluation *with* RAG.

1.  **Configure Index:** Open `evaluation_gemini_rag.py` and ensure the `INDEX_NAME` variable **matches the exact same name** you used in Step 1.
2.  **Run Script:**
    ```bash
    python evaluation_gemini_rag.py
    ```
3.  **Output:** This will read inputs from `userinputs_test/`, query Pinecone, and create a **new directory** named `results_gemini_rag_test/`. Inspect the JSON files there to see the RAG model's output.

-----

## 3\. 🗂️ Repository Structure

  * `.`
      * `evaluation_gemini_only.py`: Main script for LLM-Only baseline evaluation.
      * `evaluation_gemini_rag.py`: Main script for RAG evaluation.
      * `pinecone_loader.py`: Script to populate the vector DB from `dbjson/`.
      * `requirements.txt`: Python dependencies.
      * `.env.example`: Template for API keys.
  * `dbjson/`: Sample data (2 files) to be loaded into Pinecone.
  * `userinputs_test/`: Sample inputs (2 files) used for both evaluations.
  * `expected_outputs/`: Contains sample JSON outputs for quick reference.
  * `master_prompt/`: Japanese (original) and English (reference) prompt templates.
  * `utils/`: Data preparation scripts (reference only; see `utils/README.md`).

-----

## 4\. 💬 Note on Prompt Files

> **Prompt Files**
>
> The master prompts used for this research are provided in Japanese (`_jp.txt`), which were used to generate the results in our paper.
>
> For clarity and review purposes, reference translations are provided as `_en.txt` files.
