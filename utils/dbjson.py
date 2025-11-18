"""
Data assembly script for the utils folder (reference).

[Workflow]
1. Load the `okjson/` folder (manually curated list of valid IDs).
2. From `merged_csv_...` (CSV), fetch text data (consultation, expert advice) keyed by `ID`.
3. From `imagejson/`, fetch OCR results (household data) keyed by `ID`.
4. From `outputs/`, fetch reduction proposal data keyed by `ID`.
5. Combine and clean all data, then generate final JSON files (e.g., 0001.json) in `dbjson/`
   for Pinecone ingestion.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# --- 1. Constants / paths ---
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "merged_csv_with_id_with_sections.csv"
IMAGE_JSON_DIR = BASE_DIR / "imagejson"
OUTPUTS_DIR = BASE_DIR / "outputs"
OK_JSON_DIR = BASE_DIR / "okjson"
OUTPUT_DIR = BASE_DIR / "dbjson"

# --- Optional: limit processing for tests (None processes all) ---
PROCESSING_LIMIT = None  # Set an int to cap processing for tests


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file from a given path."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def clean_financial_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean imagejson (OCR data).
    Keep only 'totals' and 'expenses' under 'structured'.
    """
    structured = data.get("structured", {})
    return {
        "totals": structured.get("totals", {}),
        "expenses": structured.get("expenses", []),
    }


def clean_expert_reduction(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean outputs (reduction data).
    Keep only 'items', and rename 'evidence_best' to 'justification'.
    """
    cleaned_items: List[Dict[str, Any]] = []

    # Only keep keys needed for RAG
    allowed_keys = {
        "category",
        "current_amount_yen",
        "reduction_amount_yen",
        "reduction_ratio",
    }

    for item in data.get("items", []):
        new_item = {}
        for key, value in item.items():
            if key in allowed_keys:
                new_item[key] = value
            # Rename 'evidence_best' to 'justification'
            elif key == "evidence_best":
                new_item["justification"] = value

        if new_item:  # Do not append empty items
            cleaned_items.append(new_item)

    return {"items": cleaned_items}


def main() -> None:
    """
    Aggregate data from CSV and JSON folders and generate final dbjson files.
    """
    if not CSV_PATH.exists():
        print(f"Error: CSV file not found: {CSV_PATH}", file=sys.stderr)
        return

    if not OK_JSON_DIR.exists():
        print(f"Error: okjson folder not found: {OK_JSON_DIR}", file=sys.stderr)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records_created_count = 0

    print(f"Loading CSV from {CSV_PATH}...")

    okjson_candidates = sorted(OK_JSON_DIR.glob("*.json"), key=lambda path: path.name)

    if not okjson_candidates:
        print(f"Warning: No JSON files found in {OK_JSON_DIR}.", file=sys.stderr)
        return

    csv_rows_by_id = {}
    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            case_id_str = row.get("ID")
            if not case_id_str:
                print("Warning: Skipped a row without ID column.", file=sys.stderr)
                continue

            try:
                case_id = int(case_id_str)
            except ValueError:
                print(
                    f"Warning: Skipped non-numeric ID '{case_id_str}'.",
                    file=sys.stderr,
                )
                continue

            if case_id in csv_rows_by_id:
                print(
                    f"Warning: Duplicate ID {case_id} in CSV. Overwriting with later row.",
                    file=sys.stderr,
                )

            csv_rows_by_id[case_id] = row

    for okjson_path in okjson_candidates:

        if PROCESSING_LIMIT is not None and records_created_count >= PROCESSING_LIMIT:
            print(f"\nReached test limit of {PROCESSING_LIMIT} items. Stopping.")
            break

        file_stub = okjson_path.stem

        try:
            case_id = int(file_stub)
        except ValueError:
            print(
                f"Warning: Could not parse ID from {okjson_path.name}. Skipping.",
                file=sys.stderr,
            )
            continue

        print(f"--- Processing okjson {okjson_path.name} (ID: {case_id}) ---")

        row = csv_rows_by_id.get(case_id)
        if not row:
            print(
                f"Warning: ID {case_id} not found in CSV. Skipping.",
                file=sys.stderr,
            )
            continue

        if not row.get("consultation_text"):
            print(
                f"Warning: ID {case_id} has no consultation_text in CSV. Skipping.",
                file=sys.stderr,
            )
            continue

        if not row.get("expert_advice_text"):
            print(
                f"Warning: ID {case_id} has no expert_advice_text in CSV. Skipping.",
                file=sys.stderr,
            )
            continue

        image_json_path = IMAGE_JSON_DIR / f"{file_stub}.json"
        if not image_json_path.exists():
            print(
                f"Warning: {image_json_path} not found. Skipping.",
                file=sys.stderr,
            )
            continue

        reduction_json_path = OUTPUTS_DIR / f"{file_stub}.json"
        if not reduction_json_path.exists():
            print(
                f"Warning: {reduction_json_path} not found. Skipping.",
                file=sys.stderr,
            )
            continue

        try:
            image_data = load_json(image_json_path)
        except Exception as e:
            print(
                f"Warning: Error reading {image_json_path}: {e}",
                file=sys.stderr,
            )
            continue

        try:
            reduction_data = load_json(reduction_json_path)
        except Exception as e:
            print(
                f"Warning: Error reading {reduction_json_path}: {e}",
                file=sys.stderr,
            )
            continue

        final_record = {
            "id": case_id,
            "consultation_text": row["consultation_text"],
            "financial_data": clean_financial_data(image_data),
            "expert_reduction": clean_expert_reduction(reduction_data),
            "expert_advice_text": row["expert_advice_text"],
        }

        output_path = OUTPUT_DIR / f"{file_stub}.json"
        output_path.write_text(
            json.dumps(final_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"-> Generated {output_path}")
        records_created_count += 1

    print(f"\nDone: Generated {records_created_count} JSON files in {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
