"""
04_explanation_model.py
=======================
Generate AML analyst explanations from high-risk evidence records stored in a
Hugging Face dataset.

Input:
    outputs/high_risk_evidence.csv

Output:
    outputs/model_output_explanations.csv

Each run processes up to MAX_CUSTOMERS_PER_RUN customers from
high_risk_evidence.csv in a single LLM call.
"""

import csv
import io
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

load_dotenv()
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "gemini")

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
DEFAULT_HF_DATASET_REPO = "scotiabank-big-data-team/2026-scotiabank-imi-big-data-ai"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SECONDS_BETWEEN_CALLS = 10
MAX_CUSTOMERS_PER_RUN = 600
LLM_BATCH_SIZE = 300

GGUF_LOCAL_PATH = os.getenv("GGUF_LOCAL_PATH", "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
GGUF_HF_REPO = os.getenv("GGUF_HF_REPO", "bartowski/Llama-3.2-3B-Instruct-GGUF")
GGUF_FILENAME = os.getenv("GGUF_FILENAME", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
LOCAL_LLM_N_CTX = int(os.getenv("LOCAL_LLM_N_CTX", "4096"))
LOCAL_LLM_N_BATCH = int(os.getenv("LOCAL_LLM_N_BATCH", "256"))
LOCAL_LLM_N_GPU_LAYERS = int(os.getenv("LOCAL_LLM_N_GPU_LAYERS", "0"))

HF_HIGH_RISK_EVIDENCE_PATH = os.getenv(
    "HF_HIGH_RISK_EVIDENCE_PATH",
    "outputs/high_risk_evidence.csv",
)


def default_explanation_output_path(model_backend: str) -> str:
    if model_backend == "gemini":
        return "outputs/gemini_output_explanations.csv"
    if model_backend == "local_llm":
        return "outputs/llama_output_explanations.csv"
    return "outputs/model_output_explanations.csv"


HF_EXPLANATION_OUTPUT_PATH = os.getenv(
    "HF_EXPLANATION_OUTPUT_PATH",
    default_explanation_output_path(MODEL_BACKEND),
)

SYSTEM_PROMPT = """You are an expert Anti-Money Laundering (AML) explanation writer embedded in a financial intelligence pipeline. Your sole task is to read structured customer evidence records and produce a concise, professional, plain-language narrative for each customer that an AML analyst can immediately act on.

You are not a scoring model. You do not re-score, disagree with, or second-guess the risk scores you receive. Your job is to translate machine-generated evidence into a coherent investigative summary.


== INPUT FORMAT ==

You will receive a JSON array. Each element represents one flagged customer and contains the following fields:

customer_id: Unique customer identifier.
final_hybrid_score: Ensemble risk score from 0.0 to 1.0. All customers in this batch exceed 0.60.
hybrid_risk_category: Score band label — Very High or High.
coverage: Data completeness ratio. 0.0 means full transaction history is available. 1.0 means no transaction history exists and the score is derived entirely from profile-based fallback heuristics. Values in between indicate partial history.
primary_rule_typology: The single typology domain where the customer scored highest under deterministic rules.
rules_triggered: Total count of hard regulatory flags tripped.
rule_indicator_trace: The exact list of rules fired with severity codes, e.g. "human_trafficking: R_HT_02(H) | structuring: R_ST_01(M)". This is your primary evidence source.
rule_human_trafficking: Count of human trafficking rules triggered.
rule_structuring_layering: Count of structuring and layering rules triggered.
rule_behavioural_profile: Count of behavioural and profile anomaly rules triggered.
rule_trade_shell: Count of trade-based ML and shell entity rules triggered.
rule_cross_border_geo: Count of cross-border and geographic risk rules triggered.
if_score_max: Severity of the customer's single most anomalous behavioural pattern, from 0.0 to 1.0.
if_score_human_trafficking: Isolation forest anomaly score for human trafficking behaviour.
if_score_structuring_layering: Isolation forest anomaly score for structuring and layering behaviour.
if_score_behavioural_profile: Isolation forest anomaly score for behavioural and profile anomalies.
if_score_trade_shell: Isolation forest anomaly score for trade-based ML and shell entity behaviour.
if_score_cross_border_geo: Isolation forest anomaly score for cross-border and geographic risk behaviour.
cluster_primary_typology: The dominant AML typology of the peer cluster this customer belongs to.
cluster_risk_tier: Baseline risk level of that peer cluster.


== AML INDICATOR REFERENCE ==

When writing explanations, you must cite indicators from the table below using their Indicator Id. Only cite indicators that are genuinely supported by the evidence fields you have received. Do not cite indicators absent from this list. Do not invent indicator IDs. If a rule ID in rule_indicator_trace does not map to an indicator in this list, do not cite it.

-- Structuring & Layering --
ATYPICAL-006 | Temporal Transaction Pattern | Suspicious pattern emerges from client's transactions (e.g. same time of day) | Risk: Medium
ATYPICAL-007 | Quick In-Out | Atypical transfers by client on in-and-out basis, or cash deposit followed immediately by wire transfer out | Risk: High
ATYPICAL-008 | Same-Day Turnover | Funds transferred in and out of account on same day or within relatively short period | Risk: High
BEHAV-001 | Location Hopping | Client conducts transactions at different physical locations | Risk: Medium
PROF-006 | Fund Movement | Large and/or rapid movement of funds not commensurate with client's financial profile | Risk: High
PROF-007 | Round Sums | Rounded sum transactions atypical of what would be expected from client | Risk: Low
STRUCT-001 | Cash Structuring | Multiple cash deposits below $10,000 within short timeframe to avoid reporting requirements | Risk: High
STRUCT-006 | Short Period Multiple Transactions | Multiple transactions conducted below reporting threshold within short period | Risk: High

-- Behavioural & Profile Anomalies --
ACCT-001 | Dormant Activation | Inactive account begins to see financial activity (deposits, wire transfers, withdrawals) | Risk: Medium
ACCT-002 | Periodic Patterns | Accounts receive periodical deposits and are inactive at other periods without logical explanation | Risk: Medium
ACCT-003 | Credit Surge | Sudden increase in credit card usage or applications for new credit | Risk: Medium
ACCT-004 | Abrupt Change | Abrupt change in account activity | Risk: Medium
PROD-008 | Credit Card Abuse | Credit card transactions and payments exceptionally high including excessive cash advances, balance transfers, or luxury items | Risk: High
PROF-001 | Activity vs Expectation | Transactional activity far exceeds projected activity at account opening or relationship beginning | Risk: High
PROF-002 | Financial Standing | Transactional activity inconsistent with client's apparent financial standing, usual pattern, or occupation | Risk: High
PROF-003 | Geographic Volume | Volume of transactional activity exceeds norm for geographical area | Risk: Medium
PROF-005 | Living Beyond Means | Client appears to be living beyond their means; significantly more spending than income | Risk: Medium
PROF-008 | Transaction Type/Size | Size or type of transactions atypical of what is expected from client | Risk: Medium
PROF-010 | Sudden Change | Sudden change in client's financial profile, pattern of activity or transactions | Risk: High

-- Trade-Based ML & Shell Entities --
GATE-001 | Atypical Account Use | Gatekeeper utilizing their account for transactions not typical of their business | Risk: High
PML-TBML-02 | Sector Deviation | Entity has business activities or a model that is outside the norm for its sector | Risk: Medium
PML-TBML-03 | Counterparty Risk | The entity transacts with a large number of entities in high-volume, high-demand, or unrelated sectors | Risk: High
PML-TBML-04 | Volume Spike | Entity receives a sudden inflow of large-value electronic funds transfers | Risk: High
PML-TBML-08 | Round Sums | Orders or receives payments for goods in round figures or in increments of approximately US$50,000 | Risk: High
PROF-004 | Business Activity | Transactional activity inconsistent with declared business | Risk: High

-- Cross-Border & Geographic Risk --
GEO-001 | Drug Jurisdictions | Transactions with jurisdictions known to produce or transit drugs or precursor chemicals | Risk: High
GEO-002 | High ML/TF Risk | Transactions with jurisdictions known to be at higher risk of ML/TF | Risk: High
GEO-003 | Locations of Concern | Transaction/business activity involving locations of concern (ongoing conflicts, weak ML/TF controls) | Risk: High
GEO-004 | FATF Non-Cooperative | Transactions involving countries deemed high risk or non-cooperative by FATF | Risk: High
GEO-005 | Frequent Overseas Transfers | Client makes frequent overseas transfers not in line with their financial profile | Risk: Medium
WIRE-008 | Volume Mismatch | Large wire transfers or high volume through account that does not fit expected pattern | Risk: High
WIRE-010 | Multiple Senders/Receivers | Client sending to or receiving wire transfers from multiple clients | Risk: Medium

-- Human Trafficking --
HT-SEX-01 | Retail & Gift Card Anomalies | Rounded sum purchases at grocery stores and/or other retailers that sell gift cards and/or prepaid credit cards | Risk: Medium
HT-SEX-02 | Retail & Gift Card Anomalies | Atypical high-value purchases at convenience stores (likely for gift cards or money transfers) | Risk: Medium
HT-SEX-03 | Lifestyle & Luxury Spend | Luxury purchases inconsistent with reported income or occupation | Risk: High
HT-SEX-04 | Logistics & Victim Maintenance | Frequent low-value payments for parking | Risk: Low
HT-SEX-05 | Logistics & Victim Maintenance | Frequent purchases for food delivery services (often multiple times per day) | Risk: Low
HT-SEX-06 | Digital Integration | Transfers to virtual currency, online gambling, or investments | Risk: Medium
HT-SEX-07 | Geographic & Travel Patterns | Location of accommodation bookings corresponds to location of cash deposits across multiple cities | Risk: High
HT-SEX-08 | Geographic & Travel Patterns | Payments to online accommodation or travel websites in non-residential cities | Risk: Medium
HT-SEX-10 | International Recruitment & Travel | Large international travel purchases targeting high-risk source countries | Risk: High
HT-SEX-12 | Illicit Storefront Operations | Merchant POS transactions at Spa/Massage/Escort establishments after business hours (10 PM to 6 AM) | Risk: High
HT-SEX-13 | Asset Procurement & Management | Multiple real estate purchases or high-value transfers disproportionate to reported income | Risk: High
HT-SEX-14 | Asset Procurement & Management | Monthly payments to multiple individuals or entities involved in residential rentals | Risk: Medium


== OUTPUT FORMAT ==

Output a CSV with exactly two columns: customer_id and explanation.

The first row must be the header: customer_id,explanation

Each subsequent row contains one customer. The explanation field must be wrapped in double-quotes. Any double-quote character that appears inside the explanation text must be escaped as two consecutive double-quotes ("").

Example of correct CSV structure:
customer_id,explanation
C-00412,"This customer's activity is most consistent with..."
C-00887,"The primary concern for this customer is..."

Do not output anything before the header row. Do not output anything after the last customer row. Do not add commentary, summaries, or blank lines between rows.


== EXPLANATION RULES ==

Each explanation is a single paragraph of 3 to 5 sentences written in plain English for a trained AML analyst. The explanation must not exceed 2,000 characters including the header line for that customer.

1. LEAD WITH THE DOMINANT CONCERN. Open with the primary typology. State what the overall pattern suggests before citing any specifics.

2. CITE INDICATORS BY ID. When referencing a red flag, name it naturally and append the indicator ID in parentheses. Example: "The customer shows signs of cash structuring (STRUCT-001), with multiple sub-threshold deposits spread across short periods." Only cite indicators genuinely supported by the evidence you have received.

3. TRANSLATE SCORES INTO PLAIN LANGUAGE. Do not write raw decimal values from anomaly scores. Use relative language: "a notably elevated anomaly signature", "the strongest behavioural signal", "a secondary but corroborating pattern". Reserve superlatives for if_score_max values above 0.80.

4. USE PEER CONTEXT PURPOSEFULLY. If cluster_primary_typology aligns with the rule evidence, briefly note that the customer's behaviour is consistent with a known high-risk peer group. If it diverges, note the discrepancy as an additional reason for scrutiny.

5. FLAG LOW COVERAGE. If coverage is 0.60 or above, you must include this sentence verbatim: "Note: limited transaction history is available for this customer; the risk assessment relies partly on profile-based heuristics and should be weighted accordingly."

6. DO NOT SPECULATE. Do not assert that money laundering or trafficking has occurred. Use hedged language: "is consistent with", "suggests", "warrants investigation for", "may indicate".

7. MENTION CO-OCCURRING TYPOLOGIES. If two or more domains have a rule count above 0 or an isolation forest score above 0.50, note the secondary typology. This helps the analyst understand whether this is a single-typology case or a multi-scheme profile.

8. NEVER EXPOSE INTERNALS. Do not mention field names, model names, cluster IDs, raw score decimals, or any system architecture detail in the output.

9. IF rules_triggered IS 0 BUT anomaly scores are elevated, reflect that the flag is anomaly-driven rather than rule-confirmed and apply appropriately cautious language.

10. IF A FIELD IS NULL OR MISSING for a given customer, omit that dimension from the narrative entirely. Do not fabricate values."""

INPUT_FIELDS = [
    "customer_id",
    "final_hybrid_score",
    "hybrid_risk_category",
    "coverage",
    "primary_rule_typology",
    "rules_triggered",
    "rule_indicator_trace",
    "rule_human_trafficking",
    "rule_structuring_layering",
    "rule_behavioural_profile",
    "rule_trade_shell",
    "rule_cross_border_geo",
    "if_score_max",
    "if_score_human_trafficking",
    "if_score_structuring_layering",
    "if_score_behavioural_profile",
    "if_score_trade_shell",
    "if_score_cross_border_geo",
    "cluster_primary_typology",
    "cluster_risk_tier",
]

TRACE_TYPOLOGY_MAP = {
    "struct": "rule_structuring_layering",
    "structuring": "rule_structuring_layering",
    "structuring_layering": "rule_structuring_layering",
    "behav": "rule_behavioural_profile",
    "behavioural": "rule_behavioural_profile",
    "behavioural_profile": "rule_behavioural_profile",
    "trade": "rule_trade_shell",
    "trade_shell": "rule_trade_shell",
    "geo": "rule_cross_border_geo",
    "cross_border_geo": "rule_cross_border_geo",
    "cross-border": "rule_cross_border_geo",
    "ht": "rule_human_trafficking",
    "human_trafficking": "rule_human_trafficking",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_hf_repo_id(repo: str | None) -> str:
    repo = (repo or "").strip()
    if not repo:
        return DEFAULT_HF_DATASET_REPO
    if repo.startswith("https://") or repo.startswith("http://"):
        parts = [p for p in urlparse(repo).path.strip("/").split("/") if p]
        if len(parts) >= 3 and parts[0] == "datasets":
            return "/".join(parts[1:3])
    return repo


def download_hf_csv(repo_id: str, token: str, path_in_repo: str) -> pd.DataFrame:
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        repo_type="dataset",
        token=token,
    )
    return pd.read_csv(local_path)


def upload_hf_file(api: HfApi, repo_id: str, token: str, local_path: Path, path_in_repo: str) -> None:
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )


def normalize_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value)


def count_trace_indicators(trace: object) -> dict[str, int]:
    counts = {
        "rule_human_trafficking": 0,
        "rule_structuring_layering": 0,
        "rule_behavioural_profile": 0,
        "rule_trade_shell": 0,
        "rule_cross_border_geo": 0,
    }
    if pd.isna(trace) or not str(trace).strip():
        return counts

    for segment in str(trace).split("|"):
        segment = segment.strip()
        if ":" not in segment:
            continue
        prefix, items = segment.split(":", 1)
        field_name = TRACE_TYPOLOGY_MAP.get(prefix.strip().lower())
        if not field_name:
            continue
        counts[field_name] += len([item for item in items.split(",") if item.strip()])
    return counts


def build_input_record(row: pd.Series) -> dict:
    trace_counts = count_trace_indicators(row.get("rule_indicator_trace"))
    record = {
        "customer_id": normalize_value(row.get("customer_id")),
        "final_hybrid_score": normalize_value(row.get("final_hybrid_score")),
        "hybrid_risk_category": normalize_value(row.get("hybrid_risk_category")),
        "coverage": normalize_value(row.get("coverage")),
        "primary_rule_typology": normalize_value(row.get("primary_rule_typology")),
        "rules_triggered": normalize_value(row.get("rules_triggered")),
        "rule_indicator_trace": normalize_value(row.get("rule_indicator_trace")),
        "rule_human_trafficking": trace_counts["rule_human_trafficking"],
        "rule_structuring_layering": trace_counts["rule_structuring_layering"],
        "rule_behavioural_profile": trace_counts["rule_behavioural_profile"],
        "rule_trade_shell": trace_counts["rule_trade_shell"],
        "rule_cross_border_geo": trace_counts["rule_cross_border_geo"],
        "if_score_max": normalize_value(row.get("if_score_max")),
        "if_score_human_trafficking": normalize_value(row.get("if_score_human_trafficking")),
        "if_score_structuring_layering": normalize_value(row.get("if_score_structuring_layering")),
        "if_score_behavioural_profile": normalize_value(row.get("if_score_behavioural_profile")),
        "if_score_trade_shell": normalize_value(row.get("if_score_trade_shell")),
        "if_score_cross_border_geo": normalize_value(row.get("if_score_cross_border_geo")),
        "cluster_primary_typology": normalize_value(row.get("cluster_primary_typology")),
        "cluster_risk_tier": normalize_value(row.get("cluster_risk_tier")),
    }
    return {field: record.get(field) for field in INPUT_FIELDS}


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def parse_llm_csv(text: str, expected_ids: list[str]) -> list[dict[str, str]]:
    cleaned = strip_code_fences(text)
    if not cleaned:
        raise ValueError("Model returned an empty response.")

    reader = csv.reader(io.StringIO(cleaned))
    rows = [row for row in reader if row]
    if not rows:
        raise ValueError("Model returned no CSV rows.")

    header = [cell.strip() for cell in rows[0]]
    if header != ["customer_id", "explanation"]:
        raise ValueError(f"Unexpected CSV header: {header}")

    parsed = []
    for row in rows[1:]:
        if len(row) != 2:
            raise ValueError(f"Malformed CSV row: {row}")
        parsed.append({
            "customer_id": row[0].strip(),
            "explanation": row[1].replace("\r", " ").replace("\n", " ").strip(),
        })

    parsed_ids = [row["customer_id"] for row in parsed]
    if len(parsed_ids) != len(expected_ids):
        raise ValueError(f"Expected {len(expected_ids)} rows, got {len(parsed_ids)}.")
    if set(parsed_ids) != set(expected_ids):
        raise ValueError(f"Response customer IDs did not match request batch: {parsed_ids}")

    parsed_map = {row["customer_id"]: row["explanation"] for row in parsed}
    return [{"customer_id": customer_id, "explanation": parsed_map[customer_id]} for customer_id in expected_ids]


def parse_llm_csv_partial(text: str, expected_ids: list[str]) -> list[dict[str, str]]:
    cleaned = strip_code_fences(text)
    parsed_map = {}

    if not cleaned:
        return [{"customer_id": customer_id, "explanation": "unsuccessful"} for customer_id in expected_ids]

    try:
        reader = csv.reader(io.StringIO(cleaned))
        rows = [row for row in reader if row]
    except Exception:
        return [{"customer_id": customer_id, "explanation": "unsuccessful"} for customer_id in expected_ids]

    if not rows:
        return [{"customer_id": customer_id, "explanation": "unsuccessful"} for customer_id in expected_ids]

    start_index = 1 if [cell.strip() for cell in rows[0]] == ["customer_id", "explanation"] else 0
    for row in rows[start_index:]:
        if len(row) != 2:
            continue
        customer_id = row[0].strip()
        explanation = row[1].replace("\r", " ").replace("\n", " ").strip()
        if customer_id in expected_ids and explanation:
            parsed_map[customer_id] = explanation

    return [
        {"customer_id": customer_id, "explanation": parsed_map.get(customer_id, "unsuccessful")}
        for customer_id in expected_ids
    ]


def render_output_csv(explanations: list[dict[str, str]]) -> str:
    lines = ["customer_id,explanation"]
    for row in explanations:
        customer_id = str(row["customer_id"])
        explanation = str(row["explanation"]).replace('"', '""').replace("\r", " ").replace("\n", " ").strip()
        lines.append(f'{customer_id},"{explanation}"')
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Backend initialisation
# ---------------------------------------------------------------------------

if MODEL_BACKEND == "gemini":
    import google.generativeai as genai

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is required when MODEL_BACKEND=gemini.")

    genai.configure(api_key=GEMINI_API_KEY)
    MODEL = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
    )
elif MODEL_BACKEND == "local_llm":
    from llama_cpp import Llama

    if not Path(GGUF_LOCAL_PATH).exists():
        print("GGUF model not found locally. Downloading from Hugging Face...")
        Path(GGUF_LOCAL_PATH).parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=GGUF_HF_REPO,
            filename=GGUF_FILENAME,
            local_dir=str(Path(GGUF_LOCAL_PATH).parent),
        )
        print(f"Model downloaded to: {GGUF_LOCAL_PATH}")

    print("Loading local GGUF model...")
    print(
        f"Local LLM config: n_ctx={LOCAL_LLM_N_CTX}, "
        f"n_batch={LOCAL_LLM_N_BATCH}, n_gpu_layers={LOCAL_LLM_N_GPU_LAYERS}"
    )
    try:
        MODEL = Llama(
            model_path=GGUF_LOCAL_PATH,
            n_ctx=LOCAL_LLM_N_CTX,
            n_batch=min(LOCAL_LLM_N_BATCH, LOCAL_LLM_N_CTX),
            n_gpu_layers=LOCAL_LLM_N_GPU_LAYERS,
            offload_kqv=LOCAL_LLM_N_GPU_LAYERS > 0,
            op_offload=LOCAL_LLM_N_GPU_LAYERS > 0,
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize the local GGUF model. "
            "This environment is currently failing inside llama.cpp context creation, "
            "which is commonly caused by an incompatible Metal-enabled llama-cpp-python "
            "build on macOS. Try reinstalling a CPU-only build of llama-cpp-python or "
            "adjust LOCAL_LLM_N_CTX / LOCAL_LLM_N_GPU_LAYERS in .env. "
            f"Current config: GGUF_LOCAL_PATH={GGUF_LOCAL_PATH}, "
            f"LOCAL_LLM_N_CTX={LOCAL_LLM_N_CTX}, "
            f"LOCAL_LLM_N_BATCH={LOCAL_LLM_N_BATCH}, "
            f"LOCAL_LLM_N_GPU_LAYERS={LOCAL_LLM_N_GPU_LAYERS}. "
            f"Original error: {exc}"
        ) from exc
else:
    raise ValueError("MODEL_BACKEND must be 'gemini' or 'local_llm'.")


def call_llm(user_message: str) -> str:
    if MODEL_BACKEND == "gemini":
        last_error = "[ERROR: no response generated]"
        for attempt in range(3):
            try:
                response = MODEL.generate_content(
                    user_message,
                    generation_config=genai.types.GenerationConfig(temperature=0.2),
                )
                text = getattr(response, "text", "")
                if text and text.strip():
                    return text.strip()
                raise ValueError("Gemini returned an empty text response.")
            except Exception as exc:
                last_error = f"[ERROR: {exc}]"
                err_str = str(exc)
                if "429" in err_str and attempt < 2:
                    match = re.search(r"retry in ([\d.]+)s", err_str)
                    wait = float(match.group(1)) + 2 if match else 60
                    print(f"Rate limited. Waiting {wait:.0f}s before retry {attempt + 2}/3...")
                    time.sleep(wait)
                    continue
                break
        return last_error

    try:
        response = MODEL.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=2048,
            temperature=0.2,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"[ERROR: {exc}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is required.")

    hf_repo = normalize_hf_repo_id(HF_DATASET_REPO)

    print("=" * 70)
    print(f"AML EXPLANATION ENGINE — Powered by {MODEL_BACKEND}")
    print("=" * 70)
    print(f"HF dataset repo:            {hf_repo}")
    print(f"HF evidence input:          {HF_HIGH_RISK_EVIDENCE_PATH}")
    print(f"HF explanation output:      {HF_EXPLANATION_OUTPUT_PATH}")
    print(f"Total customers this run:   {MAX_CUSTOMERS_PER_RUN}")
    print(f"Customers per API call:     {LLM_BATCH_SIZE}")
    if MODEL_BACKEND == "gemini":
        print(f"Rate limit pause:           {SECONDS_BETWEEN_CALLS}s")

    evidence_df = download_hf_csv(hf_repo, HF_TOKEN, HF_HIGH_RISK_EVIDENCE_PATH)
    if evidence_df.empty:
        print("\nNo high-risk evidence rows found. Writing header-only CSV.")
        final_rows = []
    else:
        evidence_df = evidence_df.sort_values("final_hybrid_score", ascending=False).reset_index(drop=True)
        if MAX_CUSTOMERS_PER_RUN > 0:
            evidence_df = evidence_df.head(MAX_CUSTOMERS_PER_RUN).copy()
        batch_size = max(1, LLM_BATCH_SIZE)
        total_customers = len(evidence_df)
        total_batches = (total_customers + batch_size - 1) // batch_size if total_customers > 0 else 0
        print(f"\nCustomers queued: {total_customers:,} across {total_batches} batch(es)")

        final_rows = []
        errors = []
        local_checkpoint = Path("./model_output_explanations_checkpoint.csv")

        for batch_index, start in enumerate(range(0, len(evidence_df), batch_size), 1):
            batch_df = evidence_df.iloc[start:start + batch_size].copy()
            expected_ids = batch_df["customer_id"].astype(str).tolist()
            payload = [build_input_record(row) for _, row in batch_df.iterrows()]
            user_message = json.dumps(payload, ensure_ascii=False)

            print(f"\n[{batch_index}/{total_batches}] Explaining {len(batch_df)} customer(s): {expected_ids}")

            response_text = call_llm(user_message)
            if response_text.startswith("[ERROR"):
                print(f"  Batch failed: {response_text}")
                errors.append({"batch": batch_index, "customer_ids": expected_ids, "error": response_text})
                batch_rows = [
                    {"customer_id": customer_id, "explanation": "unsuccessful"}
                    for customer_id in expected_ids
                ]
            else:
                try:
                    batch_rows = parse_llm_csv(response_text, expected_ids)
                except Exception as exc:
                    print(f"  Batch returned partially invalid CSV: {exc}")
                    errors.append({"batch": batch_index, "customer_ids": expected_ids, "error": str(exc)})
                    batch_rows = parse_llm_csv_partial(response_text, expected_ids)

            final_rows.extend(batch_rows)
            checkpoint_text = render_output_csv(final_rows)
            local_checkpoint.write_text(checkpoint_text, encoding="utf-8")

            if MODEL_BACKEND == "gemini" and batch_index < total_batches:
                time.sleep(SECONDS_BETWEEN_CALLS)

        print(f"\nErrors: {len(errors)}")
        if errors:
            print("Failed batches:")
            for error in errors:
                print(f"  Batch {error['batch']}: {error['customer_ids']} -> {error['error']}")

    output_text = render_output_csv(final_rows)

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_output = Path(tmp_dir) / "model_output_explanations.csv"
        local_output.write_text(output_text, encoding="utf-8")
        api = HfApi()
        upload_hf_file(api, hf_repo, HF_TOKEN, local_output, HF_EXPLANATION_OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("EXPLANATIONS COMPLETE")
    print("=" * 70)
    print(f"Rows written:      {len(final_rows):,}")
    print(f"Uploaded to HF:    {hf_repo}/{HF_EXPLANATION_OUTPUT_PATH}")

    if final_rows:
        print("\nSample rows:")
        for row in final_rows[:3]:
            print(f"  {row['customer_id']}: {row['explanation'][:140]}")


if __name__ == "__main__":
    main()
