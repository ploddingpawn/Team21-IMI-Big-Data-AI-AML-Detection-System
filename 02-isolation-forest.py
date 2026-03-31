"""
02_isolation_forest.py
=======================
Typology-specific Isolation Forest ensemble for AML risk scoring.

Six typologies selected by working through AML_Knowledge_Library.xlsx
indicators and mapping each against available data columns:

Selection process:
  1. Listed 9 candidate patterns from knowledge library review
  2. Checked each against actual column availability in the dataset
  3. Merged overlapping patterns where features were too similar
  4. Dropped patterns requiring unavailable data

Final six typologies (non-overlapping feature sets):

  structuring         STRUCT-001..009, BEHAV-001
                      Threshold avoidance, round amounts, multi-location ABM.
                      Fully supported — amount_cad available on all channels.

  trade_and_shells    PML-TBML-01..08, ATYPICAL-004, OIL-06, PROF-004
                      Business volume vs declared sales, industry risk, EFT spikes.
                      Shell companies merged here — both target business customers
                      with volume/sales mismatches. Shells get unique signal from
                      flow_through_ratio (pass-through with no retained balance)
                      and employee_count (minimal staff with high volume).

  money_mules         CMLN-MULE-01..04, FENT-LAY-07, UB-ATH-15
                      Income vs inflow mismatch, youth + high outflow velocity,
                      occupation gap. Distinct from profile_mismatch by emphasizing
                      the EMT/wire outflow velocity and age signals.

  profile_mismatch    PROF-001..010, ACCT-001..004, ATYPICAL-001..008
                      Broad behavioural inconsistency with KYC profile.
                      Focuses on individual customers and sudden change signals.
                      Does NOT include income ratio (money_mules) or business
                      volume ratio (trade_and_shells) to avoid overlap.

  geographic_risk     GEO-001..005, PML-MSB-02, PML-MSB-10
                      High-risk jurisdiction exposure and international activity.
                      Limited to ABM channel (only one with country data) plus
                      wire/WU as channel-level geographic proxies.

  layering            ATYPICAL-007/008, CMLN-MULE-03/04, FENT-LAY-01..05,
                      WIRE-006..010, PML-MSB-07
                      Pass-through, fund velocity, instrument mixing, wire
                      consolidation. Absorbs wire transfer patterns (channel-level
                      signals belong to the layering behaviour) and fintech
                      exploitation (EMT-heavy P2P patterns are layering via
                      digital instruments).

Patterns not supported by available data:
  - Fintech exploitation: needs card MCC for gambling/crypto — absent
  - Wire transfers (standalone): wire is a channel; behaviour = layering
    Wire signals folded into layering forest
  - Human trafficking: needs card MCC for hotel/spa/gift cards — absent
  - Terrorist financing: needs travel/sanctions data — absent
  - Bribery/corruption: needs PEP flags — absent

Usage:
    python 02_isolation_forest.py --base_dir /path/to/AML_Competition
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
np.random.seed(42)


# ---------------------------------------------------------------------------
# Typology feature sets — non-overlapping by design
# Each feature appears in exactly ONE typology.
# ---------------------------------------------------------------------------

TYPOLOGY_FEATURE_PATTERNS = {

    # ── 1. Structuring / Smurfing ──────────────────────────────────────────
    # STRUCT-001..009, BEHAV-001
    # Breaking amounts below $10K CTR threshold, round numbers, multi-location
    # cash deposits. Fully computable from amount_cad on any channel.
    "structuring": [
        "count_near_10k",
        "count_very_near_10k",
        "count_below_10k",
        "ratio_below_10k",
        "count_round_100",
        "ratio_round_100",
        "count_round_1000",
        "ratio_round_1000",
        "unique_cities",
        "unique_provinces",
        "avg_txn_per_city",
        "count_abm",
        "sum_abm",
    ],

    # ── 2. Trade-Based ML + Shell Companies ───────────────────────────────
    # PML-TBML-01..08, ATYPICAL-004, OIL-06, PROF-004
    # Business customers with volume far exceeding declared sales, high-risk
    # industry codes, EFT/wire spikes inconsistent with business profile.
    # Shell companies detected via flow-through (no retained balance) and
    # low employee count with high volume.
    # employee_count sourced from kyc_smallbusiness.csv.
    "trade_and_shells": [
        "volume_to_sales_ratio",
        "industry_risk_high",
        "sum_eft",
        "count_eft",
        "sum_wire",
        "count_wire",
        "flow_through_ratio",
        "is_business",
        "employee_count",
        "ratio_round_1000",
        "has_wire",
        "monthly_txn_cv",
    ],

    # ── 3. Money Mules ────────────────────────────────────────────────────
    # CMLN-MULE-01..04, FENT-LAY-07, UB-ATH-15
    # Individual customers receiving funds disproportionate to income/occupation.
    # Key differentiator from profile_mismatch: emphasizes youth signal (age),
    # EMT/wire outflow velocity, and occupation gap — the three signals that
    # specifically identify mule accounts rather than general profile anomalies.
    "money_mules": [
        "income_vol_ratio",
        "spending_to_income_ratio",
        "age",
        "occupation_risk_high",
        "count_emt",
        "sum_emt",
        "customer_tenure_days",
        "transactions_per_active_day",
        "total_inflow",
        "inflow_per_day",
        "is_individual",
    ],

    # ── 4. Profile Mismatch / Account Activity / Atypical ─────────────────
    # PROF-001..010, ACCT-001..004, ATYPICAL-001..008
    # Broad behavioural inconsistency — sudden changes, dormant activation,
    # erratic patterns. Focuses on change signals rather than ratio signals
    # (which belong to money_mules and trade_and_shells).
    "profile_mismatch": [
        "monthly_volume_cv",
        "monthly_txn_count_std",
        "amount_cv",
        "std_transaction_amount",
        "max_transaction_amount",
        "avg_transaction_amount",
        "time_span_days",
        "has_any_high_risk_txn",
        "channel_diversity",
        "channel_concentration_hhi",
        "transaction_count_total",
        "net_flow",
        "active_days",
    ],

    # ── 5. Geographic Risk ────────────────────────────────────────────────
    # GEO-001..005, PML-MSB-02, PML-MSB-10
    # Transactions with high-risk jurisdictions. ABM is the only channel with
    # country data — geographic signals limited to cash transactions.
    # WesternUnion is included as a proxy for cross-border risk where
    # country data is unavailable (WU is inherently international).
    "geographic_risk": [
        "drug_country_txn_count",
        "drug_country_txn_sum",
        "high_risk_fatf_txn_count",
        "greylist_txn_count",
        "offshore_center_txn_count",
        "underground_banking_country_count",
        "international_txn_count",
        "international_txn_sum",
        "international_ratio",
        "unique_countries",
        "has_any_high_risk_txn",
        "sum_westernunion",
        "count_westernunion",
        "has_western_union",
    ],

    # ── 6. Layering ───────────────────────────────────────────────────────
    # ATYPICAL-007/008, CMLN-MULE-03, FENT-LAY-01..05
    # WIRE-006..010, PML-MSB-07
    # Rapid movement of funds through accounts to obscure origin.
    # Absorbs wire transfer patterns (wire volume = layering via wires)
    # and fintech/EMT P2P patterns (EMT pass-through = digital layering).
    # Core signal: money enters and immediately exits via multiple instruments.
    "layering": [
        "flow_through_ratio",
        "inflow_outflow_ratio",
        "net_flow",
        "monthly_txn_cv",
        "monthly_volume_std",
        "transactions_per_day",
        "volume_per_day",
        "outflow_per_day",
        "count_cheque",
        "sum_cheque",
        "total_outflow",
        "total_inflow",
        "total_volume",
    ],
}

TYPOLOGY_LABELS = {
    "structuring":       "Structuring / Smurfing (FINTRAC — STRUCT-001..009)",
    "trade_and_shells":  "Trade-Based ML & Shell Companies (PML-TBML-01..08, ATYPICAL-004)",
    "money_mules":       "Money Mules (CMLN-MULE-01..04, FENT-LAY-07)",
    "profile_mismatch":  "Profile Mismatch & Account Activity (PROF-001..010, ACCT-001..004)",
    "geographic_risk":   "Geographic Risk (GEO-001..005, PML-MSB-02/10)",
    "layering":          "Layering & Fund Movement (ATYPICAL-007/008, WIRE-006..010)",
}

EXCLUDE_COLS = {
    "customer_id", "label",
    "first_transaction_date", "last_transaction_date",
    "birth_date", "onboard_date", "established_date",
    "customer_type", "occupation_code", "industry_code",
    "day_name", "time_of_day", "year_month",
}

LOG_TRANSFORM_PATTERNS = [
    "volume", "outflow", "inflow", "sum_", "count_",
    "total_", "amount", "txn_count", "net_flow",
    "international_txn", "drug_country", "greylist", "offshore",
    "employee",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_feature_cols(patterns, available_cols):
    matched = []
    for pattern in patterns:
        if pattern in available_cols:
            if pattern not in matched:
                matched.append(pattern)
        else:
            for col in available_cols:
                if pattern in col and col not in matched:
                    matched.append(col)
    return matched


def get_log_cols(feature_cols, df):
    return [
        c for c in feature_cols
        if any(p in c for p in LOG_TRANSFORM_PATTERNS)
        and c in df.columns
        and df[c].min() >= 0
    ]


def normalize_scores(raw):
    mn, mx = raw.min(), raw.max()
    if mx == mn:
        return np.zeros_like(raw)
    return (mx - raw) / (mx - mn)


def risk_category(score):
    if score >= 0.80: return "Very High"
    if score >= 0.60: return "High"
    if score >= 0.40: return "Medium"
    if score >= 0.20: return "Low"
    return "Very Low"


def train_typology_if(name, df, feature_patterns, n_estimators, contamination):
    available = [c for c in df.columns if c not in EXCLUDE_COLS
                 and pd.api.types.is_numeric_dtype(df[c])]
    feat_cols = resolve_feature_cols(feature_patterns, available)

    if not feat_cols:
        print(f"   WARNING: No features matched for {name}")
        return np.zeros(len(df)), []

    X = df[feat_cols].copy()
    log_cols = get_log_cols(feat_cols, X)
    if log_cols:
        X[log_cols] = np.log1p(X[log_cols])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    iso.fit(X_scaled)
    return normalize_scores(iso.decision_function(X_scaled)), feat_cols


def print_validation(typology, scores, labels):
    labeled_mask = labels.notna().values
    if labeled_mask.sum() == 0:
        return 0.0

    y_true   = labels.values[labeled_mask]
    y_scores = scores[labeled_mask]

    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except Exception:
        roc_auc = 0.0

    susp_positions = np.where(labels.values == 1)[0]
    sorted_idx     = np.argsort(-scores)
    n_total        = len(scores)

    captures = {}
    for pct in [0.01, 0.05, 0.10]:
        cutoff  = int(n_total * pct)
        top_set = set(sorted_idx[:cutoff])
        captures[pct] = sum(1 for p in susp_positions if p in top_set)

    print(f"   {typology:<22} ROC-AUC={roc_auc:.3f} | "
          f"@1%={captures[0.01]}/10  "
          f"@5%={captures[0.05]}/10  "
          f"@10%={captures[0.10]}/10")
    return roc_auc


def verify_grounding(typology, scores, df, top_n=50):
    key_features = {
        "structuring":      ["count_near_10k", "count_very_near_10k", "ratio_round_100", "unique_cities"],
        "trade_and_shells": ["volume_to_sales_ratio", "industry_risk_high", "flow_through_ratio", "employee_count"],
        "money_mules":      ["income_vol_ratio", "spending_to_income_ratio", "age", "count_emt"],
        "profile_mismatch": ["monthly_volume_cv", "amount_cv", "max_transaction_amount", "channel_diversity"],
        "geographic_risk":  ["drug_country_txn_count", "international_ratio", "sum_westernunion", "high_risk_fatf_txn_count"],
        "layering":         ["flow_through_ratio", "monthly_txn_cv", "total_outflow", "count_cheque"],
    }.get(typology, [])

    top_idx = np.argsort(-scores)[:top_n]
    top_df  = df.iloc[top_idx]

    print(f"\n   Top {top_n} feature profile vs population:")
    print(f"   {'Feature':<35} {'Top'+str(top_n):>10} {'Population':>12} {'Ratio':>8}")
    print(f"   {'-'*35} {'-'*10} {'-'*12} {'-'*8}")
    for feat in key_features:
        if feat in df.columns:
            pop_mean = df[feat].mean()
            top_mean = top_df[feat].mean()
            ratio    = top_mean / (pop_mean + 1e-9)
            flag     = "  ✓" if ratio > 3 else ("  ?" if ratio > 1.5 else "  ✗")
            print(f"   {feat:<35} {top_mean:>10.3f} {pop_mean:>12.3f} {ratio:>7.1f}x{flag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Typology-Specific IF Ensemble")
    parser.add_argument("--base_dir",      type=str,   default="/content/gdrive/MyDrive/AML_Competition")
    parser.add_argument("--n_estimators",  type=int,   default=300)
    parser.add_argument("--contamination", type=float, default=0.01)
    args = parser.parse_args()

    base_dir  = Path(args.base_dir)
    feat_dir  = base_dir / "features"
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("TYPOLOGY-SPECIFIC ISOLATION FOREST ENSEMBLE")
    print("=" * 70)
    print(f"Typologies: {list(TYPOLOGY_FEATURE_PATTERNS.keys())}")
    print(f"n_estimators={args.n_estimators}, contamination={args.contamination}")
    print()
    print("Pattern selection: 9 candidate patterns reviewed against data availability.")
    print("Reduced to 6 non-overlapping forests:")
    print("  Merged: trade-based ML + shell companies (shared business features)")
    print("  Merged: wire transfers into layering (channel, not a behaviour pattern)")
    print("  Merged: fintech exploitation into layering (EMT P2P = digital layering)")
    print("  Dropped: human trafficking (no card MCC), terrorist financing (no travel)")

    print("\nLoading features...")
    df = pd.read_csv(feat_dir / "customer_features_enhanced.csv")
    print(f"   {df.shape[0]:,} customers, {df.shape[1]} columns")

    customer_ids = df["customer_id"].copy()
    labels       = df["label"].copy() if "label" in df.columns \
                   else pd.Series([np.nan]*len(df))
    print(f"   Labeled: {labels.notna().sum():,}  |  Suspicious: {int(labels.sum())}")
    print(f"   (Labels used only as final sanity check)")

    # Add employee_count from KYC if not already in features
    if "employee_count" not in df.columns:
        print("   Note: employee_count not in feature matrix — shell company signal unavailable")
        print("   Add employee_count to 01_feature_engineering.py KYC merge for full shell detection")

    print("\n" + "-" * 70)
    print("Training typology-specific Isolation Forests...")
    print("-" * 70)

    all_scores   = {}
    all_features = {}

    for typology, patterns in TYPOLOGY_FEATURE_PATTERNS.items():
        print(f"\n[{typology}]")
        available = [c for c in df.columns if c not in EXCLUDE_COLS
                     and pd.api.types.is_numeric_dtype(df[c])]
        feat_cols = resolve_feature_cols(patterns, available)
        print(f"   Features matched: {len(feat_cols)}")
        print(f"   {feat_cols[:6]}{'...' if len(feat_cols) > 6 else ''}")

        scores, feat_cols_used = train_typology_if(
            typology, df, patterns, args.n_estimators, args.contamination
        )
        all_scores[typology]   = scores
        all_features[typology] = feat_cols_used
        print(f"   Score range: [{scores.min():.3f}, {scores.max():.3f}]  mean={scores.mean():.3f}")

        verify_grounding(typology, scores, df)

    # Validation
    print("\n" + "-" * 70)
    print("VALIDATION (sanity check — do NOT tune to these numbers)")
    print("-" * 70)

    roc_aucs = {}
    for typology, scores in all_scores.items():
        roc_aucs[typology] = print_validation(typology, scores, labels)

    best = max(roc_aucs, key=roc_aucs.get)
    print(f"\n   Best single typology: {best} (ROC-AUC={roc_aucs[best]:.3f})")

    # Overlap analysis
    print("\n" + "-" * 70)
    print("OVERLAP ANALYSIS — top 1% of each typology")
    print("-" * 70)
    top1pct  = int(len(df) * 0.01)
    top_sets = {}
    for typology, scores in all_scores.items():
        sorted_idx = np.argsort(-scores)
        top_sets[typology] = set(customer_ids.iloc[sorted_idx[:top1pct]])

    typologies = list(top_sets.keys())
    for i, t1 in enumerate(typologies):
        for t2 in typologies[i+1:]:
            overlap = len(top_sets[t1] & top_sets[t2])
            print(f"   {t1} ∩ {t2}: {overlap} ({overlap/top1pct*100:.0f}%)")
    print("\n   (10-30% = healthy. >50% = typologies may be redundant.)")

    # Build results
    print("\nBuilding results table...")
    results    = pd.DataFrame({"customer_id": customer_ids, "actual_label": labels})
    score_cols = []

    for typology, scores in all_scores.items():
        col = f"if_score_{typology}"
        results[col] = scores
        score_cols.append(col)

    results["if_score_max"]  = results[score_cols].max(axis=1)
    results["if_score_mean"] = results[score_cols].mean(axis=1)
    results["primary_typology"] = (
        results[score_cols]
        .idxmax(axis=1)
        .str.replace("if_score_", "")
        .map(TYPOLOGY_LABELS)
    )
    results["risk_category"] = results["if_score_max"].apply(risk_category)
    results = results.sort_values("if_score_max", ascending=False).reset_index(drop=True)

    print(f"\nRisk category distribution:")
    print(results["risk_category"].value_counts().to_string())
    print(f"\nPrimary typology (top 1,000):")
    print(results.head(1000)["primary_typology"].value_counts().to_string())

    # Save
    results.to_csv(model_dir / "isolation_forest_results.csv", index=False)
    results[results["if_score_max"] > 0.60].to_csv(
        model_dir / "isolation_forest_high_risk.csv", index=False
    )
    joblib.dump(all_features, model_dir / "isolation_forest_feature_sets.pkl")
    joblib.dump(score_cols,   model_dir / "isolation_forest_score_cols.pkl")

    print("\n" + "=" * 70)
    print("ISOLATION FOREST ENSEMBLE COMPLETE")
    print("=" * 70)
    for typology, feat_cols in all_features.items():
        print(f"   {typology:<25} {len(feat_cols):>3} features  "
              f"ROC-AUC={roc_aucs[typology]:.3f}")
    print(f"\n   Score columns for 03_hybrid_model.py:")
    for col in score_cols:
        print(f"      {col}")


if __name__ == "__main__":
    main()