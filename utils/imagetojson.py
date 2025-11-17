"""
imagetojson.py
CSV(merged_csv_data.csv)全行のImage列URLから家計簿画像を取得し、
Gemini 2.5 Flashで構造化(Rich) JSONを生成。
★ 修正: CSVの 'ID' 列を読み取り、ファイル名とJSON内部の 'case_id' として使用する。
既存出力はスキップ、上限エラーで安全停止、ログは run_log_imagejson.txt。
"""

from google import genai
from google.genai import types  # type: ignore
from dotenv import load_dotenv  # type: ignore
import os, requests, json, time, sys  # type: ignore
import pandas as pd  # type: ignore
from datetime import datetime

# ======== 設定 ========
CSV_PATH = "merged_csv_data.csv"
OUT_DIR = "imagejson"
LOG_PATH = "run_log_imagejson.txt"
MODEL = "gemini-2.5-flash"  # ★ 指定モデル
SLEEP_SEC = 1.5  # ★ リクエスト間スリープ（秒）
MAX_ITEMS = 1000000  # 任意の打ち切り数（基本は全件）

# ======== 初期化 ========
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

# ======== CSV読み込み ========
try:
    # CSV読み込み時、'ID' 列を文字列として読み込むよう指定するとより安全
    df = pd.read_csv(CSV_PATH, dtype={"ID": str})
except Exception as e:
    log(f"❌ CSV読み込み失敗: {e}")
    sys.exit(1)

log(f"Total rows in CSV: {len(df)}")

# ======== Geminiプロンプト ========
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

# ======== カウンタ ========
count_done = count_skip = count_err = 0
jsonl_path = os.path.join(OUT_DIR, "household.jsonl")

# ======== メインループ ========
for idx, row in df.iterrows():
    if count_done >= MAX_ITEMS:
        log(f"⛳ Hit MAX_ITEMS={MAX_ITEMS}. Stopping safely.")
        break

    # --- ▼ 修正箇所 ▼ ---
    # CSVから 'ID' カラムを取得し、整数に変換
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

    # CSVの 'ID' を使って出力パスを決定
    out_path = os.path.join(OUT_DIR, f"{case_id:04d}.json")
    # --- ▲ 修正ここまで ▲ ---

    # 既存スキップ
    if os.path.exists(out_path):
        log(f"⏭️ Skip {out_path} (already exists)")
        count_skip += 1
        continue

    title = str(row.get("Title", ""))[:60]
    image_url = str(row.get("Image", "")).strip()
    article_url = str(row.get("URL", "")).strip()

    if not image_url or image_url.lower() == "nan":
        # ログにCSVのID（case_id）を使うように変更
        log(f"[{case_id:04d}] ⚠️ Image URL missing. Skip.")
        count_skip += 1
        continue

    log(f"[{case_id:04d}] Fetching image & parsing | {title}")

    try:
        # ---- 画像取得 ----
        r = requests.get(image_url, timeout=25)
        r.raise_for_status()
        img_bytes = r.content

        # ---- Gemini 実行 ----
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

        # ---- JSON整形 ----
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"status": "error", "raw_output": raw}

        # メタ補完
        if "source" not in data or not isinstance(data["source"], dict):
            data["source"] = {}

        # --- ▼ 修正箇所 ▼ ---
        # JSON内部にもCSVから取得した正しい 'case_id' を書き込む
        data["case_id"] = case_id
        # --- ▲ 修正ここまで ▲ ---

        data["source"]["url"] = article_url or None
        data["source"]["image_url"] = image_url
        data["source"]["title"] = title

        # ---- 保存 ----
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

        count_done += 1
        log(f"✅ Saved {out_path}")

        # ---- スリープでレート制御 ----
        time.sleep(SLEEP_SEC)

    except Exception as e:
        emsg = str(e)
        # ログにCSVのID（case_id）を使うように変更
        log(f"❌ Error on case {case_id:04d}: {emsg}")
        count_err += 1

        # 上限検知で停止
        if ("RESOURCE_EXHAUSTED" in emsg) and (
            "GenerateRequestsPerDayPerProjectPerModel" in emsg
            or "per-project-per-model" in emsg
            or "RequestsPerMinutePerProject" in emsg
        ):
            log("⛳ Quota/Rate limit hit. Stopping safely. Resume later to continue.")
            break

        continue

# ======== サマリ ========
dt = time.perf_counter() - t0_all
m, s = divmod(dt, 60)
log(f"\nSummary: Done={count_done}, Skipped={count_skip}, Error={count_err}")
log(
    f"Elapsed: {int(m)}m{s:05.2f}s for {len(df)} rows (processed {count_done+count_skip+count_err})"
)
log("=== Batch end ===")
