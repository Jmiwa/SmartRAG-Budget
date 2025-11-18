"""
Read `merged_csv_data.csv` and extract expert "expense reduction suggestions".
(Reference script for the utils/ folder)

[Process]
1. Read `Image` (image URL) and `text` (full article) from the CSV.
2. Send the image, article text, and Japanese prompt (`PROMPT`) to `gemini-2.5-flash`.
3. Let Gemini analyze the article to calculate/extract expense-reduction suggestions
   (item, reduction amount, reduction ratio) and return JSON.
4. Save the extracted JSON to the `outputs/` folder.

[ID handling]
Read the CSV `ID` column and use it as the output filename (e.g., 0001.json).
"""

from google import genai
from google.genai import types  # type: ignore
from dotenv import load_dotenv  # type: ignore
import os, requests, time, json  # type: ignore
import pandas as pd  # type: ignore
from datetime import datetime

# ======== Setup ========
t_all_start = time.perf_counter()
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Logging
log_path = "run_log.txt"


def log(message):
    """Write logs to both file and console."""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    text = f"{timestamp} {message}"
    print(text)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


log("=== Batch start ===")

# Read CSV file
csv_path = "utils/sample_source_data.csv"
try:
    # Read 'ID' column as string for safety
    df = pd.read_csv(csv_path, dtype={"ID": str})
except Exception as e:
    log(f"❌ Failed to read CSV: {e}")
    # Uncomment to stop the script on CSV read failure
    # import sys
    # sys.exit(1)
    df = pd.DataFrame()  # Empty DataFrame to skip loop

log(f"Total rows in CSV: {len(df)}")

# Process all rows (or sample by uncommenting)
# df_sample = df.head(5)
df_sample = df

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

# Counters
count_ok, count_no, count_error, count_skip = 0, 0, 0, 0

# ======== Main loop ========
for idx, row in df_sample.iterrows():

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
    out_path = f"outputs/{case_id:04d}.json"

    title = str(row["Title"])[:30]
    image_url = str(row["Image"])
    text_data = str(row["text"])  # Use 'text' column (full article)

    # Skip if output already exists
    if os.path.exists(out_path):
        msg = f"⏭️ Skip {out_path} (already exists)"
        log(msg)
        count_skip += 1
        continue

    log(f"[{case_id:04d}] Processing: {title}")

    try:
        # ---- Fetch image bytes ----
        img_bytes = requests.get(image_url, timeout=20).content

        # ---- Prompt (Japanese kept as-is) ----
        prompt = """
あなたは家計アドバイス抽出アシスタントです。以下の「収支画像」と「本文」だけを根拠に、
支出削減に関する具体的提案を抽出し、各項目の現在額→削減額→削減割合(0〜1, 小数第6位丸め)
を算出して JSON で返してください。推測や一般論は禁止です。

厳守事項:
- 画像の「支出」に記載がある項目の現在額のみ使用すること。
- 本文に明示された削減提案（％, 円, 上限 等）に基づき削減額を決めること。
- 本文に根拠がない項目は出力しない。要約や言い換えは禁止（原文引用）。
- 小数点第6位で四捨_入した reduction_ratio を返すこと。
- 削減額が現状額を上回る場合でも、本文中で削減額が明示されているときはその数値を優先し、status は "ok" とする。
- 削減提案に「見直し」「新規加入を考慮しても」などの表現が含まれる場合でも、実質的に支出が減少している旨が明記されていれば "ok" として扱うこと。

根拠（evidence）に関する規定（重要）:
- 本文全体を走査し、対象項目の根拠文を複数抽出すること。
- 数値（円/万円/%/割/半分/上限 など）を含む文を優先し、上位3件を "evidence_all" に入れること。
- "evidence_best" は最も定量的で削減額を直接確定できる文（必要なら隣接2文まで結合可）。
- 可能なら元テキスト内の文字オフセット start/end も返す（任意）。
- "evidence_best" は、quant_amount > percent > cap > action_only の優先順位で選ぶこと。

出力は次の JSON のみ（説明文は禁止）:
{
"status": "ok|no_advice|partial",
"items": [
    {
    "category": "string",
    "current_amount_yen": number,
    "reduction_amount_yen": number,
    "reduction_ratio": number,
    "evidence_best": "string",
    "evidence_all": [
        { "text": "string", "type": "quant_amount|percent|cap|action_only", "start": 0, "end": 0 }
    ]
    }
],
"notes": "string"
}
"""

        # ---- Start timer ----
        t0 = time.perf_counter()

        # ---- Call Gemini ----
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                prompt,
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                "本文:\n" + text_data,
            ],
            config={"response_mime_type": "application/json"},
        )

        # ---- Stop timer ----
        elapsed = time.perf_counter() - t0

        # ---- Parse JSON ----
        result_text = response.text.strip()
        try:
            data = json.loads(result_text)
        except json.JSONDecodeError:
            data = {"status": "error", "raw_output": result_text}

        # ---- Save file ----
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # ---- Update counters based on status ----
        status = data.get("status", "")
        if status == "ok":
            count_ok += 1
        elif status == "no_advice":
            count_no += 1
        else:
            count_error += 1

        log(f"✅ Saved {out_path} | Status={status} | Time={elapsed:.2f}s")

        # ---- Sleep to avoid rate limits ----
        time.sleep(1.5)

    except Exception as e:
        msg = f"❌ Error on case {case_id:04d}: {e}"
        log(msg)

        # --- Stop safely on daily free-tier quota cap (RPD) ---
        # 429 RESOURCE_EXHAUSTED caused by per-model free tier daily cap
        s = str(e)
        if (
            "RESOURCE_EXHAUSTED" in s
            and "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in s
        ):
            log(
                "⛳ Daily free-tier quota hit. Stopping safely. "
                "Resume after Pacific midnight (≈16:00 JST)."
            )
            break  # Stop loop to avoid further requests

        count_error += 1
        continue

# ======== Overall timer end ========
t_all_end = time.perf_counter()
total_time = t_all_end - t_all_start

# ---- Output summary ----
mins, secs = divmod(total_time, 60)
summary = (
    f"\nResults summary: OK={count_ok}, No advice={count_no}, Error={count_error}, Skipped={count_skip}\n"
    f"=== All {len(df_sample)} items processed in {total_time:.2f} seconds "
    f"({int(mins)}m {int(secs):02d}s) ==="
)

log(summary)
log("=== Batch end ===\n")
