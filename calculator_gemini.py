from google import genai
from google.genai import types # type: ignore
from dotenv import load_dotenv # type: ignore
import os, requests, time, json # type: ignore
import pandas as pd # type: ignore
from datetime import datetime

# ======== 初期設定 ========
t_all_start = time.perf_counter()
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ログ設定
log_path = "run_log.txt"


def log(message):
    """ログをファイルとコンソール両方に出す"""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    text = f"{timestamp} {message}"
    print(text)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


log("=== Batch start ===")

# CSVファイル読み込み
csv_path = "merged_csv_data.csv"  # ファイル名は適宜変更
df = pd.read_csv(csv_path)
log(f"Total rows in CSV: {len(df)}")

# テストとして最初の5件だけ処理
df_sample = df

# 出力ディレクトリ作成
os.makedirs("outputs", exist_ok=True)

# カウンタ初期化
count_ok, count_no, count_error, count_skip = 0, 0, 0, 0

# ======== メインループ ========
for idx, row in df_sample.iterrows():
    title = str(row["Title"])[:30]
    image_url = str(row["Image"])
    text_data = str(row["text"])

    # 出力ファイル名（0001.json形式）
    out_path = f"outputs/{idx+1:04d}.json"

    # すでに出力ファイルがある場合はスキップ
    if os.path.exists(out_path):
        msg = f"⏭️ Skip {out_path} (already exists)"
        print(msg)
        log(msg)
        count_skip += 1
        continue

    log(f"[{idx+1}] Processing: {title}")

    try:
        # ---- 画像のバイト列を取得 ----
        img_bytes = requests.get(image_url, timeout=20).content

        # ---- プロンプト定義 ----
        prompt = """
あなたは家計アドバイス抽出アシスタントです。以下の「収支画像」と「本文」だけを根拠に、
支出削減に関する具体的提案を抽出し、各項目の現在額→削減額→削減割合(0〜1, 小数第6位丸め)
を算出して JSON で返してください。推測や一般論は禁止です。

厳守事項:
- 画像の「支出」に記載がある項目の現在額のみ使用すること。
- 本文に明示された削減提案（％, 円, 上限 等）に基づき削減額を決めること。
- 本文に根拠がない項目は出力しない。要約や言い換えは禁止（原文引用）。
- 小数点第6位で四捨五入した reduction_ratio を返すこと。
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

        # ---- タイマー開始 ----
        t0 = time.perf_counter()

        # ---- Gemini実行 ----
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                prompt,
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                "本文:\n" + text_data,
            ],
            config={"response_mime_type": "application/json"},
        )

        # ---- タイマー終了 ----
        elapsed = time.perf_counter() - t0

        # ---- JSON整形 ----
        result_text = response.text.strip()
        try:
            data = json.loads(result_text)
        except json.JSONDecodeError:
            data = {"status": "error", "raw_output": result_text}

        # ---- ファイル保存 ----
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # ---- ステータスでカウント更新 ----
        status = data.get("status", "")
        if status == "ok":
            count_ok += 1
        elif status == "no_advice":
            count_no += 1
        else:
            count_error += 1

        log(f"✅ Saved {out_path} | Status={status} | Time={elapsed:.2f}s")

        # ---- sleepでアクセス制限回避 ----
        time.sleep(1.5)

    except Exception as e:
        msg = f"❌ Error on index {idx+1}: {e}"
        print(msg)
        log(msg)

        # --- 日次無料枠の「回数上限」(RPD) を検知したら安全終了 ---
        # 429 RESOURCE_EXHAUSTED で、かつ Free Tier の日次上限（per-model）が原因のとき
        s = str(e)
        if (
            "RESOURCE_EXHAUSTED" in s
            and "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in s
        ):
            log(
                "⛳ Daily free-tier quota hit. Stopping safely. "
                "Resume after Pacific midnight (≈16:00 JST)."
            )
            break  # ここでループ終了（以降の無駄なリクエストを止める）

        count_error += 1
        continue

# ======== 全体タイマー終了 ========
t_all_end = time.perf_counter()
total_time = t_all_end - t_all_start

# ---- 結果サマリ出力 ----
mins, secs = divmod(total_time, 60)
summary = (
    f"\nResults summary: OK={count_ok}, No advice={count_no}, Error={count_error}, Skipped={count_skip}\n"
    f"=== All {len(df_sample)} items processed in {total_time:.2f} seconds "
    f"({int(mins)}m {int(secs):02d}s) ==="
)

# print(summary)

log(summary)
log("=== Batch end ===\n")
