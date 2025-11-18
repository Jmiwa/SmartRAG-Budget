"""
imagetojson.py
Fetch household ledger images from the Image column of every row in merged_csv_data.csv,
and generate structured (Rich) JSON with Gemini 2.5 Flash.
Use the CSV 'ID' column as both the filename and the internal JSON 'case_id'.
Existing outputs are skipped, safety stops on quota errors, and logs are written to run_log_imagejson.txt.
"""

from google import genai
from google.genai import types  # type: ignore
from dotenv import load_dotenv  # type: ignore
import os, requests, json, time, sys  # type: ignore
import pandas as pd  # type: ignore
from datetime import datetime

# ======== Config ========
CSV_PATH = "merged_csv_data.csv"
OUT_DIR = "imagejson"
LOG_PATH = "run_log_imagejson.txt"
MODEL = "gemini-2.5-flash"  # Target model
SLEEP_SEC = 1.5  # Sleep between requests (seconds)
MAX_ITEMS = 1000000  # Optional cutoff (default: process all)

# ======== Init ========
t0_all = time.perf_counter()
os.makedirs(OUT_DIR, exist_ok=True)
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def log(msg: str):
    ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


log("=== Batch start (CSV→Rich JSON using Gemini 2.5 Flash) ===")

# ======== Load CSV ========
try:
    # Read 'ID' as string to be safe
    df = pd.read_csv(CSV_PATH, dtype={"ID": str})
except Exception as e:
    log(f"❌ Failed to read CSV: {e}")
    sys.exit(1)

log(f"Total rows in CSV: {len(df)}")

# ======== Gemini prompt ========
PROMPT = """
あなたは家計表画像の構造化変換アシスタントです。
与えられた画像（1枚）から、添付のJSONスキーマ「だけ」を満たすオブジェクトを出力してください。
出力は application/json のみ。説明文・補足・マークダウンは禁止。

【厳守ルール】
- 金額はすべて円の整数（半角）。「万円」は10000倍して円に換算。カンマは除去。
- 「収入」「支出」「貯蓄」「投資」「毎月の貯蓄額/積立額」「各合計」を読み取り、
  スキーマに沿って structured.* に格納。
- 「合計」は明細の合算で計算し、画像内の合計表示と ±1% を超えて乖離する場合は qc.checks.* を false にし、qc.notes に理由を書く。
- 画像に存在しない項目はその配列から省略（nullや空文字を入れない）。
- ラベルは画像の見出しに合わせる（例：「電気・ガス・水道料金」）。
- 小数表記が出た場合は四捨_入して整数円にする。
- 出力はスキーマのキー順を守る。

【出力スキーマ】
{
  "case_id": 1,
  "source": { "url": "string or null", "image_url": "string", "title": "string" },
  "structured": {
    "income": [{"label": "string", "amount_yen": 0}],
    "expenses": [{"label": "string", "amount_yen": 0}],
    "savings": [{"label": "string", "amount_yen": 0}],
    "investments": [{"label": "string", "amount_yen": 0}],
    "totals": {
      "income_monthly_yen": 0,
      "expense_monthly_yen": 0,
      "savings_total_yen": 0,
      "investments_total_yen": 0,
      "savings_monthly_yen": 0,
      "investments_monthly_yen": 0
    }
  },
  "normalization": {
    "unit_rules": "『万円』×10000, 『円』はそのまま, 全角→半角, カンマ除去",
    "currency": "JPY"
  },
  "qc": {
    "confidence": 0.0,
    "checks": {
      "sum_expenses_matches_panel": true,
      "required_fields_present": true
    },
    "notes": []
  }
}
"""

# ======== Counters ========
count_done = count_skip = count_err = 0
jsonl_path = os.path.join(OUT_DIR, "household.jsonl")

# ======== Main loop ========
for idx, row in df.iterrows():
    if count_done >= MAX_ITEMS:
        log(f"⛳ Hit MAX_ITEMS={MAX_ITEMS}. Stopping safely.")
        break

    # Grab 'ID' column from CSV and convert to int
    try:
        case_id_str = str(row.get("ID", "")).strip()
        if not case_id_str or case_id_str.lower() == "nan":
            log(f"[Row {idx+1}] ⚠️ ID column is missing or empty. Skip.")
            count_skip += 1
            continue

        case_id = int(case_id_str)

    except ValueError:
        log(f"[Row {idx+1}] ⚠️ ID '{case_id_str}' is not a valid integer. Skip.")
        count_skip += 1
        continue

    # Build output path using CSV 'ID'
    out_path = os.path.join(OUT_DIR, f"{case_id:04d}.json")

    # Skip if already exists
    if os.path.exists(out_path):
        log(f"⏭️ Skip {out_path} (already exists)")
        count_skip += 1
        continue

    title = str(row.get("Title", ""))[:60]
    image_url = str(row.get("Image", "")).strip()
    article_url = str(row.get("URL", "")).strip()

    if not image_url or image_url.lower() == "nan":
        # Log with CSV ID (case_id)
        log(f"[{case_id:04d}] ⚠️ Image URL missing. Skip.")
        count_skip += 1
        continue

    log(f"[{case_id:04d}] Fetching image & parsing | {title}")

    try:
        # ---- Fetch image ----
        r = requests.get(image_url, timeout=25)
        r.raise_for_status()
        img_bytes = r.content

        # ---- Run Gemini ----
        t1 = time.perf_counter()
        resp = client.models.generate_content(
            model=MODEL,
            contents=[
                PROMPT,
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            ],
            config={"response_mime_type": "application/json"},
        )
        elapsed = time.perf_counter() - t1
        raw = (resp.text or "").strip()
        log(f"[{case_id:04d}] Gemini ok ({elapsed:.2f}s)")

        # ---- Parse JSON ----
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"status": "error", "raw_output": raw}

        # Fill metadata
        if "source" not in data or not isinstance(data["source"], dict):
            data["source"] = {}

        # Write correct case_id from CSV into JSON
        data["case_id"] = case_id

        data["source"]["url"] = article_url or None
        data["source"]["image_url"] = image_url
        data["source"]["title"] = title

        # ---- Save ----
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

        count_done += 1
        log(f"✅ Saved {out_path}")

        # ---- Sleep to manage rate ----
        time.sleep(SLEEP_SEC)

    except Exception as e:
        emsg = str(e)
        # Log with CSV ID (case_id)
        log(f"❌ Error on case {case_id:04d}: {emsg}")
        count_err += 1

        # Stop on quota detection
        if ("RESOURCE_EXHAUSTED" in emsg) and (
            "GenerateRequestsPerDayPerProjectPerModel" in emsg
            or "per-project-per-model" in emsg
            or "RequestsPerMinutePerProject" in emsg
        ):
            log("⛳ Quota/Rate limit hit. Stopping safely. Resume later to continue.")
            break

        continue

# ======== Summary ========
dt = time.perf_counter() - t0_all
m, s = divmod(dt, 60)
log(f"\nSummary: Done={count_done}, Skipped={count_skip}, Error={count_err}")
log(
    f"Elapsed: {int(m)}m{s:05.2f}s for {len(df)} rows (processed {count_done+count_skip+count_err})"
)
log("=== Batch end ===")
