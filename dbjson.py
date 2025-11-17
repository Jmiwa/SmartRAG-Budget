import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# --- 1. 定数・パス設定 ---
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "merged_csv_with_id_with_sections.csv"
IMAGE_JSON_DIR = BASE_DIR / "imagejson"
OUTPUTS_DIR = BASE_DIR / "outputs"
OK_JSON_DIR = BASE_DIR / "okjson"
OUTPUT_DIR = BASE_DIR / "dbjson"

# --- ここを追加 ---
PROCESSING_LIMIT = None  # テスト用に処理する最大件数（Noneで全件処理）


def load_json(path: Path) -> Dict[str, Any]:
    """指定されたパスからJSONファイルを読み込む"""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def clean_financial_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    imagejson（OCRデータ）を整形する。
    'structured' 配下の 'totals' と 'expenses' のみを取得する。
    """
    structured = data.get("structured", {})
    return {
        "totals": structured.get("totals", {}),
        "expenses": structured.get("expenses", []),
    }


def clean_expert_reduction(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    outputs（削減割合データ）を整形する。
    'items' のみを取得し、'evidence_best' を 'justification' にリネームする。
    """
    cleaned_items: List[Dict[str, Any]] = []

    # RAGで参照するキーのみを許可する
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
            # 'evidence_best' は 'justification' にリネームして採用
            elif key == "evidence_best":
                new_item["justification"] = value

        if new_item:  # 空のアイテムは追加しない
            cleaned_items.append(new_item)

    return {"items": cleaned_items}


def main() -> None:
    """
    CSVと各JSONフォルダからデータを集約し、
    最終的なdbjsonファイルを生成する。
    """
    if not CSV_PATH.exists():
        print(f"エラー: CSVファイルが見つかりません: {CSV_PATH}", file=sys.stderr)
        return

    if not OK_JSON_DIR.exists():
        print(f"エラー: okjsonフォルダが見つかりません: {OK_JSON_DIR}", file=sys.stderr)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records_created_count = 0

    print(f"{CSV_PATH} を読み込んでいます...")

    okjson_candidates = sorted(
        OK_JSON_DIR.glob("*.json"), key=lambda path: path.name
    )

    if not okjson_candidates:
        print(f"警告: {OK_JSON_DIR} に処理対象のJSONがありません。", file=sys.stderr)
        return

    csv_rows_by_id = {}
    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            case_id_str = row.get("ID")
            if not case_id_str:
                print("警告: ID列がない行をスキップしました。", file=sys.stderr)
                continue

            try:
                case_id = int(case_id_str)
            except ValueError:
                print(
                    f"警告: 数値に変換できないID '{case_id_str}' をスキップしました。",
                    file=sys.stderr,
                )
                continue

            if case_id in csv_rows_by_id:
                print(
                    f"警告: ID {case_id} がCSV内で重複しています。後に出現した行で上書きします。",
                    file=sys.stderr,
                )

            csv_rows_by_id[case_id] = row

    for okjson_path in okjson_candidates:

        if PROCESSING_LIMIT is not None and records_created_count >= PROCESSING_LIMIT:
            print(
                f"\nテスト上限の {PROCESSING_LIMIT} 件に達したため、処理を終了します。"
            )
            break

        file_stub = okjson_path.stem

        try:
            case_id = int(file_stub)
        except ValueError:
            print(
                f"警告: okjson内のファイル名 {okjson_path.name} からIDを取得できません。スキップします。",
                file=sys.stderr,
            )
            continue

        print(f"--- okjson {okjson_path.name} (ID: {case_id}) を処理中 ---")

        row = csv_rows_by_id.get(case_id)
        if not row:
            print(
                f"警告: ID {case_id} がCSVに存在しません。スキップします。",
                file=sys.stderr,
            )
            continue

        if not row.get("consultation_text"):
            print(
                f"警告: ID {case_id} の consultation_text がCSVにありません。スキップします。",
                file=sys.stderr,
            )
            continue

        if not row.get("expert_advice_text"):
            print(
                f"警告: ID {case_id} の expert_advice_text がCSVにありません。スキップします。",
                file=sys.stderr,
            )
            continue

        image_json_path = IMAGE_JSON_DIR / f"{file_stub}.json"
        if not image_json_path.exists():
            print(
                f"警告: {image_json_path} が見つかりません。スキップします。",
                file=sys.stderr,
            )
            continue

        reduction_json_path = OUTPUTS_DIR / f"{file_stub}.json"
        if not reduction_json_path.exists():
            print(
                f"警告: {reduction_json_path} が見つかりません。スキップします。",
                file=sys.stderr,
            )
            continue

        try:
            image_data = load_json(image_json_path)
        except Exception as e:
            print(
                f"警告: {image_json_path} の読み込み中にエラーが発生しました: {e}",
                file=sys.stderr,
            )
            continue

        try:
            reduction_data = load_json(reduction_json_path)
        except Exception as e:
            print(
                f"警告: {reduction_json_path} の読み込み中にエラーが発生しました: {e}",
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

        print(f"-> {output_path} を生成しました。")
        records_created_count += 1

    print(
        f"\n完了: {OUTPUT_DIR} に {records_created_count} 件のJSONファイルを生成しました。"
    )


if __name__ == "__main__":
    main()
