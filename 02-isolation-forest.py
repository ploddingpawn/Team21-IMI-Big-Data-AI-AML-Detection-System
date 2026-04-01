"""
02_isolation_forest.py
=======================
Typology-specific Isolation Forest ensemble for AML risk scoring.

Five typologies mapped directly from AML-indicator-DB.xlsx.
One Isolation Forest is trained per typology, each on its own dedicated
feature set. The five scores are then combined in 03_hybrid_model.py.

  1. structuring_layering
       STRUCT-001..009, ATYPICAL-006..008, BEHAV-001, PROF-006/007
       Threshold avoidance, round amounts, multi-location ABM deposits,
       rapid pass-through (same-day in/out), EFT/EMT velocity.
       Merged from old structuring + layering — both share the core
       signal of unusual transaction sizing and fund velocity.

  2. behavioural_profile
       PROF-001..010, ACCT-001..004, ATYPICAL-001..008
       Activity inconsistent with KYC profile: spending vs income,
       sudden change in volume patterns, erratic transaction sizing,
       dormant account reactivation, credit card usage surge (ACCT-003).
       Merged from old money_mules + profile_mismatch — both target
       customers whose behaviour doesn't match their declared profile.

  3. trade_shell
       PML-TBML-02..08, PROF-004, GATE-001
       Business volume far exceeding declared sales, high-risk industry
       codes, EFT inflow spikes (PML-TBML-04 — direct from eft.csv),
       shell company signal via flow-through + low employee count.

  4. cross_border_geo
       GEO-001..005, WIRE-008, WIRE-010
       Transactions with FATF blacklist / greylist / drug-transit
       jurisdictions, frequent overseas transfers, wire volume mismatch.
       Card merchant country data now available (card_unique_countries).

  5. human_trafficking
       HT-SEX-01..14 (full set — card.csv, eft.csv, emt.csv available)
       Card MCC signals: after-hours spa/massage (HT-SEX-12), luxury
       spend (HT-SEX-03), accommodation (HT-SEX-08), travel (HT-SEX-10),
       parking/delivery (HT-SEX-04/05). EMT/EFT debit outflows for
       rental payments (HT-SEX-14). Multi-city ABM night cash (HT-SEX-07).

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
# Typology feature sets — aligned to AML-indicator-DB.xlsx
# Each list entry is an exact column name or substring pattern.
# resolve_feature_cols() tries exact match first, then substring.
# ---------------------------------------------------------------------------

TYPOLOGY_FEATURE_PATTERNS = {

    # ── 1. Structuring & Layering ──────────────────────────────────────────
    # STRUCT-001: Cash deposits below $10K (ABM cash_indicator)
    # STRUCT-003: Transactions $9,000–$9,999 (near-threshold avoidance)
    # STRUCT-006: High transaction velocity in short period
    # ATYPICAL-007: Rapid in-and-out / cash deposit then immediate EFT out
    # ATYPICAL-008: Funds in and out on same day
    # BEHAV-001:   Multi-location transactions (location hopping)
    # PROF-006:    Large rapid fund movement inconsistent with profile
    # PROF-007:    Rounded sum transactions
    "structuring_layering": [
        "count_near_10k",
        "count_very_near_10k",
        "count_below_10k",
        "ratio_below_10k",
        "cash_struct_count",           # STRUCT-001: ABM cash below $10K
        "count_round_100",
        "ratio_round_100",
        "count_round_1000",
        "ratio_round_1000",
        "flow_through_ratio",          # ATYPICAL-007/008: same-day pass-through
        "transactions_per_active_day", # STRUCT-006: velocity
        "volume_per_day",
        "unique_cities",               # BEHAV-001: location hopping
        "unique_provinces",
        "count_abm",
        "sum_abm",
        "count_eft",                   # ATYPICAL-007: rapid EFT pass-through
        "sum_eft",
        "count_emt",                   # ATYPICAL-007: rapid EMT pass-through
        "sum_emt",
        "count_wire",
        "sum_wire",                    # PROF-006: large rapid wire movement
        "count_cheque",
        # ── New from updated 01_feature_engineering ──────────────────────
        "max_days_between_txns",                      # ACCT-001: gap signals dormancy then sudden reactivation
        "total_volume_vs_ind_median",                 # STRUCT-006/PROF-006: volume vs individual peer median
        "total_volume_vs_bus_median",                 # STRUCT-006/PROF-006: volume vs business peer median
        "total_volume_above_ind_p90",                 # binary: exceeds 90th pct individual peer volume
        "total_volume_above_bus_p90",                 # binary: exceeds 90th pct business peer volume
        "transactions_per_active_day_vs_ind_median",  # STRUCT-006: velocity vs individual peer
        "transactions_per_active_day_vs_bus_median",  # STRUCT-006: velocity vs business peer
        "transactions_per_active_day_above_ind_p90",
        "transactions_per_active_day_above_bus_p90",
        "flow_through_ratio_vs_ind_median",           # ATYPICAL-007/008: pass-through vs peer baseline
        "flow_through_ratio_vs_bus_median",
        "outflow_per_day",                               # STRUCT-006: daily outflow velocity
        "sum_cheque",                                    # STRUCT-009: cheque volume (cheque structuring pattern)
        "has_emt",                                       # ATYPICAL-007: EMT channel presence flag
        "structuring_layering_risk",                  # typology composite — direct signal for this IF
    ],

    # ── 2. Behavioural & Profile Anomalies ────────────────────────────────
    # PROF-001: Activity far exceeds projected at account opening
    # PROF-002: Activity inconsistent with financial standing / occupation
    # PROF-005: Living beyond means — more spending than income
    # PROF-008: Transaction type/size atypical of expectations
    # PROF-010: Sudden change in financial profile
    # ACCT-001: Dormant account reactivation
    # ACCT-002: Periodic deposits with inactivity otherwise
    # ACCT-003: Sudden increase in credit card usage
    # ACCT-004: Abrupt change in account activity
    "behavioural_profile": [
        "spending_to_income_ratio",    # PROF-002/005: financial standing
        "income_vol_ratio",            # PROF-001: activity vs expectation
        "monthly_volume_cv",           # PROF-010 / ACCT-004: sudden change
        "monthly_txn_count_std",       # ACCT-002: periodicity / erratic patterns
        "amount_cv",                   # PROF-008: transaction size atypical
        "std_transaction_amount",
        "max_transaction_amount",
        "avg_transaction_amount",
        "channel_diversity",           # ACCT-002: channel mixing
        "channel_concentration_hhi",
        "count_card",                  # ACCT-003: credit card usage surge
        "sum_card",
        "transaction_count_total",     # PROF-003: volume vs geographic norm
        "net_flow",
        "active_days",
        "time_span_days",
        "customer_tenure_days",        # ACCT-001: new customer high activity
        "is_individual",
        "age",
        "occupation_risk_high",        # PROF-002: occupation gap
        # ── New from updated 01_feature_engineering ──────────────────────
        "max_days_between_txns",              # ACCT-001: dormant account reactivation signal
        "total_outflow_vs_ind_median",        # PROF-002/005: outflow vs individual peer
        "total_outflow_vs_bus_median",        # PROF-002/005: outflow vs business peer
        "total_outflow_above_ind_p90",        # binary: individual outflow outlier
        "total_outflow_above_bus_p90",        # binary: business outflow outlier
        "total_inflow_vs_ind_median",         # PROF-001: inflow vs individual peer expectation
        "total_inflow_vs_bus_median",
        "total_inflow_above_ind_p90",
        "total_inflow_above_bus_p90",
        "card_volume_30d",                    # ACCT-003: 30-day card volume (surge detection)
        "card_volume_90d",                    # ACCT-003: 90-day card volume
        "card_30d_ratio",                     # ACCT-003: recent share of lifetime card spend
        "card_90d_ratio",
        "card_ecommerce_count",               # ACCT-003: e-commerce usage surge
        "card_ecommerce_sum",
        "evening_transaction_ratio",          # ATYPICAL-006: unusual hour concentration
        "total_volume",                                  # PROF-001: absolute activity level
        "total_outflow",                                 # PROF-005: absolute spending level
        "transaction_count_debit",                       # PROF-008: debit transaction count
        "transaction_count_credit",                      # PROF-001: credit transaction count
        "median_transaction_amount",                     # PROF-008: typical transaction size
        "behavioural_profile_risk",           # typology composite — direct signal for this IF
        "card_30d_ratio",                      # ACCT-003: Jan card share of lifetime spend
    ],

    # ── 3. Trade-Based ML & Shell Entities ────────────────────────────────
    # PML-TBML-02: Business activities outside sector norm
    # PML-TBML-03: Transacts with large number of entities
    # PML-TBML-04: Sudden large EFT inflow (direct from eft.csv)
    # PML-TBML-08: Round-figure payments (~$50K increments)
    # PROF-004:    Activity inconsistent with declared business
    # GATE-001:    Gatekeeper pass-through (atypical account use)
    "trade_shell": [
        "volume_to_sales_ratio",       # PROF-004 / PML-TBML-05: volume vs declared sales
        "industry_risk_high",          # PML-TBML-02: sector deviation
        "employee_count",              # shell signal: high volume, minimal staff
        "flow_through_ratio",          # GATE-001: pass-through / gatekeeper
        "eft_credit_sum",              # PML-TBML-04: sudden large EFT inflow (direct)
        "eft_credit_count",
        "sum_eft",
        "count_eft",
        "sum_wire",                    # PML-TBML-03: high-value wire to multiple parties
        "count_wire",
        "ratio_round_1000",            # PML-TBML-08: round sum invoices (~$50K)
        "count_round_1000",
        "monthly_txn_cv",              # PML-TBML-04: sudden spike pattern
        "is_business",
        "has_wire",
        "has_eft",
        "inflow_per_day",
        "total_inflow",
        # ── New from updated 01_feature_engineering ──────────────────────
        "eft_debit_count",                    # GATE-001: EFT outflow (pass-through debit side)
        "eft_debit_sum",
        "total_inflow_vs_bus_median",         # PML-TBML-04: inflow spike vs business peer
        "total_inflow_above_bus_p90",         # binary: business inflow outlier
        "total_volume_vs_bus_median",         # PROF-004: overall volume vs business peer
        "total_volume_above_bus_p90",
        "flow_through_ratio_vs_bus_median",   # GATE-001: pass-through vs business peer baseline
        "avg_txn_per_city",                   # PML-TBML-03: counterparty proxy via geographic spread
        "flow_through_ratio_above_ind_p90",              # GATE-001: individual pass-through outlier flag
        "flow_through_ratio_above_bus_p90",              # GATE-001: business pass-through outlier flag
        "trade_shell_risk",                   # typology composite — direct signal for this IF
    ],

    # ── 4. Cross-Border & Geographic Risk ─────────────────────────────────
    # GEO-001: Drug-producing / transit jurisdictions
    # GEO-002: Jurisdictions at higher ML/TF risk
    # GEO-003: Locations of concern (conflict, weak controls, secretive banking)
    # GEO-004: FATF non-cooperative / high-risk countries
    # GEO-005: Frequent overseas transfers not in line with financial profile
    # WIRE-008: Large wire transfers to foreign accounts (volume mismatch)
    # WIRE-010: Wire transfers from/to multiple parties
    "cross_border_geo": [
        "drug_country_txn_count",           # GEO-001: MX, CO, PE, BO, etc.
        "high_risk_fatf_txn_count",         # GEO-002/004: FATF blacklist (KP, IR, MM)
        "greylist_txn_count",               # GEO-003: FATF greylist
        "offshore_center_txn_count",        # GEO-003: secretive banking / offshore
        "underground_banking_country_count",# GEO-003: CN/HK/PH underground banking
        "international_txn_count",          # GEO-005: frequency of overseas transfers
        "international_txn_sum",
        "international_ratio",              # GEO-005: proportion of cross-border activity
        "unique_countries",
        "has_any_high_risk_txn",
        "sum_westernunion",                 # GEO-005: WU inherently international
        "count_westernunion",
        "has_western_union",
        "sum_wire",                         # WIRE-008: large wire to foreign accounts
        "count_wire",                       # WIRE-010: multiple wire parties
        "card_unique_countries",            # GEO-005: card merchant countries (direct)
        # ── New from updated 01_feature_engineering ──────────────────────
        "card_unique_cities",                # GEO-005: card spend city spread (non-residential pattern)
        "ht_source_country_txn_count",       # GEO-005 / HT-SEX-10: txns to high-risk source countries
        "cross_border_geo_risk",             # typology composite — direct signal for this IF
    ],

    # ── 5. Human Trafficking ──────────────────────────────────────────────
    # HT-SEX-01/02: Retail/convenience purchases — gift card proxy (MCC 5411/5912/5310)
    # HT-SEX-03:    Luxury spend vs income (MCC 5094/5944)
    # HT-SEX-04/05: Parking + food delivery — victim maintenance (MCC 7523/5814)
    # HT-SEX-06:    Digital / crypto / gambling (MCC 4816/7995)
    # HT-SEX-07:    Multi-city ABM cash at night (accommodation pattern)
    # HT-SEX-08:    Accommodation card spend in non-residential cities (MCC 7011)
    # HT-SEX-10:    Airfare to high-risk source countries (MCC 4511/4722)
    # HT-SEX-12:    After-hours spa/massage/escort card (MCC 7297/7298)
    # HT-SEX-13:    High-value transfers disproportionate to income
    # HT-SEX-14:    Rental/maintenance payments via EMT/EFT (direct)
    "human_trafficking": [
        "card_adult_afterhours_count", # HT-SEX-12: after-hours spa/massage (card MCC direct)
        "card_afterhours_count",       # HT-SEX-12: general after-hours card activity
        "night_transaction_ratio",     # HT-SEX-12: after-hours all-channel signal
        "night_abm_count",             # HT-SEX-07: ABM cash at night across cities
        "unique_cities",               # HT-SEX-07/08: multi-city travel pattern
        "unique_provinces",
        "card_accommodation_count",    # HT-SEX-08: accommodation in non-residential cities
        "card_accommodation_sum",
        "card_luxury_sum",             # HT-SEX-03: luxury spend vs income
        "card_luxury_count",
        "card_travel_sum",             # HT-SEX-10: airfare to source countries
        "card_travel_count",
        "ht_source_country_txn_count", # HT-SEX-10: txns to high-risk source countries
        "card_parking_count",          # HT-SEX-04: frequent parking (victim transit)
        "card_delivery_count",         # HT-SEX-05: food delivery (victim maintenance)
        "card_digital_count",          # HT-SEX-06: virtual currency / online gambling
        "card_retail_count",           # HT-SEX-01/02: retail/gift card purchases
        "emt_debit_count",             # HT-SEX-14: e-transfer rental payments (direct)
        "eft_debit_count",             # HT-SEX-14: EFT rental payments (direct)
        "spending_to_income_ratio",    # HT-SEX-13: disproportionate to income
        "sum_westernunion",            # HT-SEX-10: WU to source countries
        "ht_accommodation_sector",     # HT-SEX-07: business in accommodation/hospitality
        "is_individual",
        "weekend_ratio",               # HT-SEX-12: weekend activity concentration
        "card_retail_sum",                   # HT-SEX-01/02: retail spend total (gift card proxy)
        "card_digital_sum",                  # HT-SEX-06: digital/crypto/gambling spend total
        "card_unique_cities",                # HT-SEX-07/08: card spend city spread
        "emt_credit_count",                  # HT-SEX-09/14: incoming EMT (victim payment receipt signal)
        "emt_credit_sum",
        "emt_debit_sum",                     # HT-SEX-14: outgoing rental/maintenance sum
        "eft_debit_sum",                     # HT-SEX-14: EFT maintenance payment sum
        "avg_txn_per_city",                  # HT-SEX-07: activity spread across cities
        "card_volume_30d",                   # HT-SEX-12: recent card surge (operational tempo)
        "card_30d_ratio",
        "overall_typology_max_risk",         # cross-typology: any typology elevated (HT often co-occurs)
        "typology_breadth",                  # number of typologies > 0.3 (multi-typology red flag)
        "night_transaction_count",                       # HT-SEX-12: absolute night txn count
        "evening_transaction_count",                     # HT-SEX-12: evening activity count
        "weekend_transaction_count",                     # HT-SEX-12: weekend activity count
        "human_trafficking_risk",            # typology composite — direct signal for this IF
    ],
}

TYPOLOGY_LABELS = {
    "structuring_layering": "Structuring & Layering (STRUCT-001..009, ATYPICAL-006..008, BEHAV-001)",
    "behavioural_profile":  "Behavioural & Profile Anomalies (PROF-001..010, ACCT-001..004)",
    "trade_shell":          "Trade-Based ML & Shell Entities (PML-TBML-02..08, PROF-004, GATE-001)",
    "cross_border_geo":     "Cross-Border & Geographic Risk (GEO-001..005, WIRE-008/010)",
    "human_trafficking":    "Human Trafficking (HT-SEX-01..14)",
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
    "employee", "night_", "ht_source", "card_",
    "max_days_between",
    "avg_txn_per",
    "_vs_ind_median",
    "_vs_bus_median",
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
        "structuring_layering": ["count_near_10k", "flow_through_ratio", "ratio_round_100", "unique_cities"],
        "behavioural_profile":  ["spending_to_income_ratio", "monthly_volume_cv", "card_30d_ratio", "income_vol_ratio"],
        "trade_shell":          ["volume_to_sales_ratio", "trade_shell_risk", "flow_through_ratio", "employee_count"],
        "cross_border_geo":     ["high_risk_fatf_txn_count", "international_ratio", "sum_westernunion", "drug_country_txn_count"],
        "human_trafficking":    ["card_adult_afterhours_count", "human_trafficking_risk", "ht_source_country_txn_count", "card_luxury_sum"],
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
    print("5 Typologies — AML-indicator-DB.xlsx")
    print("=" * 70)
    for name, label in TYPOLOGY_LABELS.items():
        print(f"  [{name}] {label}")
    print(f"\nn_estimators={args.n_estimators}, contamination={args.contamination}")
    print("\nDesign decisions:")
    print("  Merged: structuring + layering → structuring_layering")
    print("          (both share threshold/velocity signals; EFT/EMT now available directly)")
    print("  Merged: profile_mismatch + money_mules → behavioural_profile")
    print("          (both target PROF indicators; card ACCT-003 now available)")
    print("  Added:  human_trafficking (HT-SEX-01..14 — card MCC, EMT/EFT direct)")
    print("  Kept:   trade_shell, cross_border_geo (clean separation, distinct signals)")

    print("\nLoading features...")
    df = pd.read_csv(feat_dir / "customer_features_enhanced.csv")
    print(f"   {df.shape[0]:,} customers, {df.shape[1]} columns")

    customer_ids = df["customer_id"].copy()
    labels       = df["label"].copy() if "label" in df.columns \
                   else pd.Series([np.nan]*len(df))
    print(f"   Labeled: {labels.notna().sum():,}  |  Suspicious: {int(labels.sum())}")
    print("   (Labels used only as final sanity check — not for training)")

    # Check for key new features produced by updated 01_feature_engineering.py
    expected = ["card_adult_afterhours_count", "ht_source_country_txn_count",
                "ht_accommodation_sector", "cash_struct_count",
                "card_luxury_sum", "card_accommodation_count",
                "eft_credit_sum", "emt_debit_count"]
    missing = [f for f in expected if f not in df.columns]
    if missing:
        print(f"\n   NOTE: Missing features — re-run 01_feature_engineering.py: {missing}")

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