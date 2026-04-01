# =============================================================================
# AML EXPLANATION ENGINE
# =============================================================================

# ==============================================================================
# BLOCK 1 — CONFIGURATION
# All secrets and paths are loaded from .env (see .env.example for the full list).
# Copy .env.example → .env and fill in your real values before running.
# ==============================================================================

from dotenv import load_dotenv
import os

load_dotenv()  # Reads .env from the current directory

# Backend selection — set MODEL_BACKEND in .env to "gemini" or "local_llm"
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "gemini")

# Hugging Face
HF_TOKEN        = os.getenv("HF_TOKEN")         # Set in .env
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")  # Set in .env

# Gemini (only used if MODEL_BACKEND == "gemini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")    # Set in .env
# Rate-limit settings (gemini-2.0-flash free tier: 15 RPM / 20 RPD)
SECONDS_BETWEEN_CALLS = int(os.getenv("SECONDS_BETWEEN_CALLS", "5"))
MAX_CUSTOMERS_PER_RUN = int(os.getenv("MAX_CUSTOMERS_PER_RUN", "1"))

# Local LLM (only used if MODEL_BACKEND == "local_llm")
GGUF_LOCAL_PATH = os.getenv("GGUF_LOCAL_PATH", "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
GGUF_HF_REPO    = os.getenv("GGUF_HF_REPO",    "bartowski/Llama-3.2-3B-Instruct-GGUF")
GGUF_FILENAME   = os.getenv("GGUF_FILENAME",    "Llama-3.2-3B-Instruct-Q4_K_M.gguf")

# HF paths for input files (relative paths inside HF_DATASET_REPO)
HF_MODEL_OUTPUT_PATH      = os.getenv("HF_MODEL_OUTPUT_PATH",      "models/model_output.csv")
HF_ISOLATION_RESULTS_PATH = os.getenv("HF_ISOLATION_RESULTS_PATH", "models/isolation_forest_results.csv")
HF_FEATURES_PATH          = os.getenv("HF_FEATURES_PATH",          "features/customer_features_engineered.csv")
HF_AML_LIBRARY_PATH       = os.getenv("HF_AML_LIBRARY_PATH",       "AML_RedFlags_Database.xlsx")

# Output path (relative path inside HF_DATASET_REPO where explanations will be uploaded)
HF_EXPLANATION_OUTPUT_PATH = os.getenv("HF_EXPLANATION_OUTPUT_PATH", "models/model_output_explanations.csv")


# ==============================================================================
# BLOCK 2 — IMPORTS & HF SETUP
# ==============================================================================

import pandas as pd
import time
import re
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

if MODEL_BACKEND == "gemini":
    import google.generativeai as genai
elif MODEL_BACKEND == "local_llm":
    from llama_cpp import Llama


# ==============================================================================
# BLOCK 3 — DOWNLOAD INPUT FILES FROM HF (replaces all GDrive reads)
# ==============================================================================

results_df = pd.read_csv(
    hf_hub_download(repo_id=HF_DATASET_REPO, filename=HF_MODEL_OUTPUT_PATH,
                    repo_type="dataset", token=HF_TOKEN)
)

isolation_df = pd.read_csv(
    hf_hub_download(repo_id=HF_DATASET_REPO, filename=HF_ISOLATION_RESULTS_PATH,
                    repo_type="dataset", token=HF_TOKEN)
)

df = pd.read_csv(
    hf_hub_download(repo_id=HF_DATASET_REPO, filename=HF_FEATURES_PATH,
                    repo_type="dataset", token=HF_TOKEN)
)

aml_library = pd.read_excel(
    hf_hub_download(repo_id=HF_DATASET_REPO, filename=HF_AML_LIBRARY_PATH,
                    repo_type="dataset", token=HF_TOKEN)
)
aml_library = aml_library.dropna(subset=["Red_Flag_Description"])
aml_library = aml_library[aml_library["Indicator_ID"].notna()].iloc[0:63]


# ==============================================================================
# BLOCK 4 — PRE-DOWNLOAD GGUF MODEL (local_llm only)
# Skips download if the file already exists locally.
# ==============================================================================

if MODEL_BACKEND == "local_llm":
    if not Path(GGUF_LOCAL_PATH).exists():
        print("GGUF model not found locally. Downloading from Hugging Face...")
        Path(GGUF_LOCAL_PATH).parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=GGUF_HF_REPO,
            filename=GGUF_FILENAME,
            local_dir=str(Path(GGUF_LOCAL_PATH).parent)
        )
        print(f"Model downloaded to: {GGUF_LOCAL_PATH}")
    else:
        print(f"GGUF model found at {GGUF_LOCAL_PATH}, skipping download.")


# ==============================================================================
# BLOCK 5 — SHARED COMPONENTS (identical for both backends)
# ==============================================================================

# ------------------------------------------------------------------
# AML Indicator Library summary
# ------------------------------------------------------------------
def build_library_summary(library_df):
    lines = []
    for _, row in library_df.iterrows():
        line = (
            f"[{row['Indicator_ID']}] {row['Indicator_Category']} > {row['Sub_Category']}: "
            f"{row['Red_Flag_Description']} "
            f"(Threshold: {row['Quantifiable_Threshold']}; "
            f"Tx Types: {row['Transaction_Type']}; "
            f"Risk Level: {row['Risk_Level']})"
        )
        lines.append(line)
    return "\n".join(lines)

LIBRARY_SUMMARY = build_library_summary(aml_library)

# ------------------------------------------------------------------
# Feature labels
# ------------------------------------------------------------------
FEATURE_LABELS = {
    "total_credit_amount":              "total money received",
    "total_debit_amount":               "total money sent",
    "avg_credit_amount":                "average incoming transaction size",
    "avg_debit_amount":                 "average outgoing transaction size",
    "max_credit_amount":                "largest single incoming transaction",
    "max_debit_amount":                 "largest single outgoing transaction",
    "credit_debit_ratio":               "ratio of money in vs money out",
    "transaction_count":                "total number of transactions",
    "credit_count":                     "number of incoming transactions",
    "debit_count":                      "number of outgoing transactions",
    "transactions_per_day":             "transactions per day",
    "unique_counterparties":            "number of different counterparties",
    "weekend_transaction_ratio":        "proportion of transactions on weekends",
    "night_transaction_ratio":          "proportion of transactions at night",
    "rapid_turnaround_count":           "number of rapid in-then-out fund movements",
    "avg_days_between_transactions":    "average days between transactions",
    "cash_transaction_ratio":           "proportion of cash transactions",
    "atm_transaction_ratio":            "proportion of ATM transactions",
    "international_transaction_ratio":  "proportion of international transactions",
    "wire_transaction_ratio":           "proportion of wire transfers",
    "structuring_indicator":            "structuring risk score",
    "round_amount_ratio":               "proportion of suspiciously round transaction amounts",
    "below_threshold_ratio":            "proportion of transactions just below reporting thresholds",
    "churn_ratio":                      "fund churn ratio (in vs out speed)",
    "income_to_credit_ratio":           "income vs actual credits ratio",
    "account_age_days":                 "how long the account has been open",
}

# ------------------------------------------------------------------
# Feature columns
# ------------------------------------------------------------------
all_cols = df.columns.tolist()

exclude_cols = [
    'customer_id',
    'label',
    'first_transaction_date',
    'last_transaction_date',
    'date',
    'transaction_datetime',
    'birth_date',
    'onboard_date',
    'established_date',
    'customer_type',
    'occupation',
    'industry',
    'day_name',
    'time_of_day',
    'year_month'
]

feature_cols = []
for col in all_cols:
    if col not in exclude_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
        else:
            print(f"Skipping non-numeric column: {col} (type: {df[col].dtype})")

# ------------------------------------------------------------------
# Feature extractor
# ------------------------------------------------------------------
def extract_notable_features(customer_row, top_n=10):
    notes = []
    for feat in feature_cols:
        if feat not in customer_row.index:
            continue
        val = customer_row[feat]
        pop_mean = df[feat].mean()
        pop_std  = df[feat].std()
        if pop_std == 0:
            continue
        z = (val - pop_mean) / pop_std
        label = FEATURE_LABELS.get(feat, feat.replace("_", " "))
        notes.append((abs(z), z, label, val, pop_mean))

    notes.sort(key=lambda x: x[0], reverse=True)

    lines = []
    for _, z, label, val, avg in notes[:top_n]:
        direction = "higher" if z > 0 else "lower"
        lines.append(
            f"- {label.capitalize()}: {val:.2f} "
            f"({'%.2f' % abs(z)}x {direction} than the typical customer average of {avg:.2f})"
        )
    return "\n".join(lines) if lines else "No standout features identified."

# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are an AML (Anti-Money Laundering) compliance expert writing risk
summaries for bank investigators. Your audience understands AML concepts but has NO
knowledge of machine learning or statistics.

You have access to the following AML Red Flag Indicator Library:
{LIBRARY_SUMMARY}

Your task: Given data about a customer and their model risk score, write a SHORT professional summary of no more than 3 short paragraphs (2000 characters):
1. States clearly whether the customer was flagged for AML review or not, and their risk level.
2. Explains, in plain terms, which specific behavioural patterns in their account data
   triggered concern — or why none did. Maps those patterns to the most relevant red flag indicators from the library above
   (cite the Indicator IDs in brackets, e.g. [CMLN-MULE-01]). Explains what these patterns could indicate in an AML context.
3. Ends with a clear recommended next step for the investigator.


Do NOT use headers, bullet points, or bold text in your response. Write in plain prose only.
Only mention terrorist financing if a terrorism-related indicator is explicitly matched.
Do NOT mention machine learning, isolation forest, anomaly scores, z-scores, or any
statistical jargon. Write as if you personally reviewed the account activity."""

def build_user_prompt(customer_id, risk_score, risk_category, prediction, feature_notes):
    flagged_text = "FLAGGED for AML review" if prediction == 1 else "NOT flagged for AML review"
    return f"""Customer ID: {customer_id}
Model Decision: {flagged_text}
Risk Level: {risk_category} (Risk Score: {risk_score:.3f})

Key account behaviours identified (compared to all customers):
{feature_notes}

Please write the AML risk explanation for this customer."""


# ==============================================================================
# BLOCK 6 — BACKEND INITIALISATION
# ==============================================================================

if MODEL_BACKEND == "gemini":
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel("gemini-2.5-flash")

elif MODEL_BACKEND == "local_llm":
    print("Loading model into memory...")
    llm = Llama(
        model_path=GGUF_LOCAL_PATH,
        n_ctx=5012,
        n_gpu_layers=-1,
        verbose=False
    )
    print("Model loaded.")


# ==============================================================================
# BLOCK 7 — UNIFIED call_llm() WRAPPER
# ==============================================================================

def call_llm(system_prompt, user_prompt):
    if MODEL_BACKEND == "gemini":
        explanation_text = "[ERROR: no response generated]"
        for attempt in range(3):
            try:
                response = model_gemini.generate_content(
                    contents=[
                        {"role": "user", "parts": [system_prompt + "\n\n" + user_prompt]}
                    ],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3
                    )
                )
                explanation_text = response.text.strip()
                break

            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    match = re.search(r"retry in ([\d.]+)s", err_str)
                    wait = float(match.group(1)) + 2 if match else 60
                    if attempt == 2:
                        explanation_text = "[ERROR: rate limit after 3 attempts]"
                        break
                    print(f"RATE LIMITED — waiting {wait:.0f}s then retrying (attempt {attempt+1}/3)...")
                    time.sleep(wait)
                else:
                    explanation_text = f"[ERROR: {err_str}]"
                    break
        return explanation_text

    elif MODEL_BACKEND == "local_llm":
        try:
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                max_tokens=512,
                temperature=0.3,
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[ERROR: {str(e)}]"


# ==============================================================================
# BLOCK 8 — MAIN EXPLANATION LOOP
# ==============================================================================

print("=" * 70)
print(f"AML EXPLANATION ENGINE — Powered by {MODEL_BACKEND}")
print("=" * 70)
print(f"\nAML Indicator Library loaded: {len(aml_library)} indicators")
print(f"Customers to explain: up to {MAX_CUSTOMERS_PER_RUN}")

# Gemini-only rate limit info
if MODEL_BACKEND == "gemini":
    print(f"Rate limit pause: {SECONDS_BETWEEN_CALLS}s between calls\n")

isolation_df = isolation_df.set_index("customer_id")

# Option 1 (local_llm) sorts on predicted_label; Option 2 (gemini) sorts on prediction.
# Both columns represent the same thing — confirm column name matches your results_df.
sort_col = "predicted_label" if MODEL_BACKEND == "local_llm" else "prediction"

results_to_explain = (
    results_df
    .sort_values([sort_col, "risk_score"], ascending=[False, False])
    .head(MAX_CUSTOMERS_PER_RUN)
    .copy()
)

print(f"Explaining {len(results_to_explain)} customers "
      f"({results_to_explain[sort_col].sum()} flagged, "
      f"{(results_to_explain[sort_col]==0).sum()} not flagged)\n")

explanations = []
errors       = []

# Local checkpoint path (saved after every customer)
LOCAL_CHECKPOINT_PATH = Path("./model_output_explanations_checkpoint.csv")

for i, (_, row) in enumerate(results_to_explain.iterrows(), 1):
    customer_id   = row["customer_id"]
    risk_score    = row["risk_score"]
    prediction    = row[sort_col]

    customer_features = df[df["customer_id"] == customer_id]

    if MODEL_BACKEND == "local_llm":
        risk_category = isolation_df.loc[customer_id, "risk_category"] if customer_id in isolation_df.index else "Unknown"
    else:
        risk_category = row["risk_category"]

    if customer_features.empty:
        feature_notes = "Detailed feature breakdown unavailable for this customer."
    else:
        feature_notes = extract_notable_features(customer_features.iloc[0])

    user_prompt = build_user_prompt(
        customer_id, risk_score, risk_category, prediction, feature_notes
    )

    print(f"[{i}/{len(results_to_explain)}] Customer {customer_id} | "
          f"{'FLAGGED' if prediction==1 else 'CLEAN':7s} | {risk_category} risk", end=" ... ")

    explanation_text = call_llm(SYSTEM_PROMPT, user_prompt)

    if explanation_text.startswith("[ERROR"):
        errors.append({"customer_id": customer_id, "error": explanation_text})
        print(f"FAILED — {explanation_text}")
    else:
        print("OK")

    explanations.append({
        "customer_id":   customer_id,
        "risk_score":    round(risk_score, 4),
        "risk_category": risk_category,
        "flagged":       bool(prediction),
        "explanation":   explanation_text
    })

    # Save checkpoint to local disk after every customer
    pd.DataFrame(explanations).to_csv(LOCAL_CHECKPOINT_PATH, index=False)

    # Gemini-only rate limit pause
    if MODEL_BACKEND == "gemini" and i < len(results_to_explain):
        time.sleep(SECONDS_BETWEEN_CALLS)


# ==============================================================================
# BLOCK 9 — UPLOAD OUTPUT TO HF (replaces GDrive save)
# ==============================================================================

explanations_df = pd.DataFrame(explanations)

# Save final CSV locally first, then upload to HF
LOCAL_OUTPUT_PATH = Path("./model_output_explanations.csv")
explanations_df.to_csv(LOCAL_OUTPUT_PATH, index=False)

api = HfApi()
api.upload_file(
    path_or_fileobj=str(LOCAL_OUTPUT_PATH),
    path_in_repo=HF_EXPLANATION_OUTPUT_PATH,
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
)

print(f"\n{'='*70}")
print(f"EXPLANATIONS COMPLETE")
print(f"{'='*70}")
print(f"  Total explained : {len(explanations_df)}")
if not explanations_df.empty:
    print(f"  Flagged         : {explanations_df['flagged'].sum()}")
    print(f"  Not flagged     : {(~explanations_df['flagged']).sum()}")
else:
    print(f"  (No explanations generated — check errors above)")
print(f"  Errors          : {len(errors)}")
print(f"  Uploaded to HF  : {HF_DATASET_REPO}/{HF_EXPLANATION_OUTPUT_PATH}")

if errors:
    print(f"\n  Failed customers: {[e['customer_id'] for e in errors]}")


# ==============================================================================
# BLOCK 10 — PREVIEW (Top 3 Explanations)
# ==============================================================================

print(f"\n{'='*70}")
print("SAMPLE EXPLANATIONS (Top 3 by Risk Score)")
print(f"{'='*70}")

for _, row in explanations_df.head(3).iterrows():
    print(f"\nCustomer: {row['customer_id']}  |  Risk: {row['risk_category']}  "
          f"|  Flagged: {row['flagged']}")
    print("-" * 60)
    print(row["explanation"])
    print()