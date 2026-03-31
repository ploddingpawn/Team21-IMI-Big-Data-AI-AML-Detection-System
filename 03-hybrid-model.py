"""
03_hybrid_model.py
==================
Hybrid AML risk scoring: Typology IF ensemble + Rule-based scoring
+ KMeans clustering, fused into a final ranked score.

═══════════════════════════════════════════════════════════════════
ARCHITECTURE
═══════════════════════════════════════════════════════════════════

Layer 1 — Typology Isolation Forests  (weight: 0.60)
  Five IF scores from 02_isolation_forest.py, one per knowledge-library
  typology. Combined as a weighted average across typologies, where
  weights reflect observed signal quality (ROC-AUC from IF validation):
    structuring_layering 0.30 | trade_shell 0.25 | behavioural_profile 0.20
    cross_border_geo     0.13 | human_trafficking 0.12

Layer 2 — Rule-Based Hard Flags  (weight: 0.30)
  Deterministic vectorised checks against FINTRAC/FinCEN thresholds.
  One rule score per typology, combined with the same weights as Layer 1.
  Vectorised — computed across all 61K customers simultaneously, not
  row-by-row. Rules fire independently of the statistical model, ensuring
  regulatory thresholds are always reflected in the final score.

Layer 3 — KMeans Cluster Risk Tier  (weight: 0.10)
  Clusters customers in the 5D IF score space. k is auto-selected by
  silhouette score rather than defaulted to 5 — in practice the score
  space tends to have 1 large low-risk mass and 2-3 high-risk profiles,
  so k=3 or k=4 often outperforms k=5. Cluster risk tier (0/0.5/1.0)
  based on mean IF score provides a soft group-level corroboration signal.

Ensemble:
  final_score = 0.60 * if_weighted + 0.30 * rule_weighted + 0.10 * cluster_tier
  Normalised to [0,1], ranked. Top 1% flagged as predicted suspicious.

Usage:
    python 03_hybrid_model.py --base_dir /path/to/AML_Competition
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, silhouette_score

warnings.filterwarnings("ignore")
np.random.seed(42)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IF_SCORE_COLS = [
    "if_score_structuring_layering",
    "if_score_behavioural_profile",
    "if_score_trade_shell",
    "if_score_cross_border_geo",
    "if_score_human_trafficking",
]

# Typology weights — reflect ROC-AUC from 02_isolation_forest.py validation.
# Downweight cross_border_geo and human_trafficking due to data sparsity.
# Must sum to 1.0.
TYPOLOGY_WEIGHTS = {
    "if_score_structuring_layering": 0.30,
    "if_score_behavioural_profile":  0.20,
    "if_score_trade_shell":          0.25,
    "if_score_cross_border_geo":     0.13,
    "if_score_human_trafficking":    0.12,
}

TYPOLOGY_LABELS = {
    "if_score_structuring_layering": "Structuring & Layering",
    "if_score_behavioural_profile":  "Behavioural & Profile Anomalies",
    "if_score_trade_shell":          "Trade-Based ML & Shell Entities",
    "if_score_cross_border_geo":     "Cross-Border & Geographic Risk",
    "if_score_human_trafficking":    "Human Trafficking",
}

# Ensemble layer weights
W_IF      = 0.60
W_RULE    = 0.30
W_CLUSTER = 0.10


# ---------------------------------------------------------------------------
# Layer 2 — Rule-based scoring (vectorised, per typology)
#
# Each function takes the full feature DataFrame and returns a Series of
# scores in [0,1] for all customers simultaneously. No iterrows().
# Scores are capped at 1.0 per typology and combined with TYPOLOGY_WEIGHTS.
# ---------------------------------------------------------------------------

def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    """Return a column as Series, or a zero-filled Series if absent."""
    return df[name].fillna(default) if name in df.columns \
           else pd.Series(default, index=df.index)


def rules_structuring_layering(df: pd.DataFrame) -> pd.Series:
    """
    STRUCT-003:       $9,000–$9,999 transactions (CTR threshold avoidance)
    STRUCT-003 (VH):  $9,800–$9,999 (very high confidence structuring)
    STRUCT-006:       > 10 transactions per active day (velocity)
    ATYPICAL-007/008: flow_through > 0.85 + volume > $50K (same-day turnover)
    PROF-007:         > 60% round-number amounts and >= 20 transactions
    """
    near  = _col(df, "count_near_10k")
    vnear = _col(df, "count_very_near_10k")
    rr    = _col(df, "ratio_round_100")
    txn   = _col(df, "transaction_count_total")
    ftr   = _col(df, "flow_through_ratio")
    vol   = _col(df, "total_volume")
    tpad  = _col(df, "transactions_per_active_day")

    s = (
        np.where(near  >= 5, 0.30, np.where(near  >= 2, 0.15, 0.0)) +
        np.where(vnear >= 3, 0.20, 0.0) +
        np.where((rr > 0.80) & (txn >= 20), 0.15, np.where((rr > 0.60) & (txn >= 20), 0.08, 0.0)) +
        np.where((ftr > 0.90) & (vol > 100_000), 0.35, np.where((ftr > 0.80) & (vol > 50_000), 0.18, 0.0)) +
        np.where(tpad > 20, 0.15, np.where(tpad > 10, 0.08, 0.0))
    )
    return pd.Series(s, index=df.index).clip(0, 1)


def rules_behavioural_profile(df: pd.DataFrame) -> pd.Series:
    """
    PROF-002: Spending > 5x declared income (financial standing inconsistency)
    PROF-004: Volume > 10x declared sales (business activity mismatch)
    PROF-005: Total volume > 20x income (living beyond means)
    ACCT-003: Card spend > 50% of income (credit card usage surge)
    """
    income   = _col(df, "income")
    outflow  = _col(df, "total_outflow")
    spi      = _col(df, "spending_to_income_ratio")
    sales    = _col(df, "sales")
    vol      = _col(df, "total_volume")
    vsr      = _col(df, "volume_to_sales_ratio")
    ivr      = _col(df, "income_vol_ratio")
    card_sum = _col(df, "sum_card")

    s = (
        np.where((income > 0) & (outflow > 10_000) & (spi > 10), 0.35,
        np.where((income > 0) & (outflow > 10_000) & (spi > 5),  0.18, 0.0)) +
        np.where((sales > 0) & (vol > 50_000) & (vsr > 20), 0.30,
        np.where((sales > 0) & (vol > 50_000) & (vsr > 10), 0.15, 0.0)) +
        np.where((income > 0) & (ivr > 50), 0.15,
        np.where((income > 0) & (ivr > 20), 0.08, 0.0)) +
        np.where((income > 0) & (card_sum > income * 0.5) & (card_sum > 10_000), 0.10, 0.0)
    )
    return pd.Series(s, index=df.index).clip(0, 1)


def rules_trade_shell(df: pd.DataFrame) -> pd.Series:
    """
    PML-TBML-04: EFT credit inflow > $100K (sudden large electronic transfer)
    WIRE-008:    Wire volume > $100K (volume mismatch for account type)
    GATE-001:    Business flow-through > 0.90 with high volume (pass-through)
    PML-TBML-08: > 40% round-$1000 amounts on business accounts
    Shell signal: <= 2 employees + volume > 20x declared sales
    """
    eft_cr = _col(df, "eft_credit_sum")
    wire   = _col(df, "sum_wire")
    ftr    = _col(df, "flow_through_ratio")
    vol    = _col(df, "total_volume")
    is_biz = _col(df, "is_business")
    rr1k   = _col(df, "ratio_round_1000")
    emp    = _col(df, "employee_count")
    vsr    = _col(df, "volume_to_sales_ratio")
    sales  = _col(df, "sales")

    s = (
        np.where(eft_cr > 500_000, 0.30, np.where(eft_cr > 100_000, 0.15, 0.0)) +
        np.where(wire   > 500_000, 0.25, np.where(wire   > 100_000, 0.12, 0.0)) +
        np.where((is_biz == 1) & (ftr > 0.90) & (vol > 100_000), 0.25,
        np.where((is_biz == 1) & (ftr > 0.80) & (vol > 50_000),  0.12, 0.0)) +
        np.where((is_biz == 1) & (rr1k > 0.60), 0.15,
        np.where((is_biz == 1) & (rr1k > 0.40), 0.08, 0.0)) +
        np.where((is_biz == 1) & (emp <= 2) & (sales > 0) & (vsr > 20), 0.20, 0.0)
    )
    return pd.Series(s, index=df.index).clip(0, 1)


def rules_cross_border_geo(df: pd.DataFrame) -> pd.Series:
    """
    GEO-002/004: Any FATF blacklist transaction — hard flag (KP, IR, MM)
    GEO-001:     Drug-transit country transactions (MX, CO, PE, etc.)
    GEO-003:     Greylist / offshore center exposure
    GEO-005:     > 50% international transactions + >= 5 international txns
    WU proxy:    WesternUnion volume as inherently international signal
    WIRE-010:    High wire count (multiple counterparties)
    """
    fatf    = _col(df, "high_risk_fatf_txn_count")
    drug    = _col(df, "drug_country_txn_count")
    grey    = _col(df, "greylist_txn_count")
    offshore= _col(df, "offshore_center_txn_count")
    intl_r  = _col(df, "international_ratio")
    intl_n  = _col(df, "international_txn_count")
    wu      = _col(df, "sum_westernunion")
    wire_n  = _col(df, "count_wire")

    s = (
        np.where(fatf    >= 3, 0.50, np.where(fatf    >= 1, 0.35, 0.0)) +
        np.where(drug    >= 5, 0.20, np.where(drug    >= 1, 0.10, 0.0)) +
        np.where(grey    >= 10, 0.15, np.where(grey   >= 3, 0.08, 0.0)) +
        np.where(offshore >= 3, 0.15, np.where(offshore >= 1, 0.08, 0.0)) +
        np.where((intl_r > 0.80) & (intl_n >= 10), 0.15,
        np.where((intl_r > 0.50) & (intl_n >= 5),  0.08, 0.0)) +
        np.where(wu > 10_000, 0.10, np.where(wu > 1_000, 0.05, 0.0)) +
        np.where(wire_n >= 10, 0.10, np.where(wire_n >= 5, 0.05, 0.0))
    )
    return pd.Series(s, index=df.index).clip(0, 1)


def rules_human_trafficking(df: pd.DataFrame) -> pd.Series:
    """
    HT-SEX-01/02: High retail card + round amounts (gift card proxy)
    HT-SEX-03:    Luxury spend inconsistent with income
    HT-SEX-04/05: Parking + food delivery (victim maintenance pattern)
    HT-SEX-07:    Multi-city ABM + night cash withdrawals
    HT-SEX-08:    Accommodation card spend
    HT-SEX-10:    Travel/airfare or source-country transactions
    HT-SEX-12:    After-hours adult MCC card txns (7297/7298 — direct)
    HT-SEX-13:    Total volume disproportionate to income
    HT-SEX-14:    Regular EMT/EFT outgoing payments (rental pattern)
    """
    retail   = _col(df, "card_retail_count")
    rr       = _col(df, "ratio_round_100")
    luxury   = _col(df, "card_luxury_sum")
    income   = _col(df, "income")
    spi      = _col(df, "spending_to_income_ratio")
    cities   = _col(df, "unique_cities")
    night_abm= _col(df, "night_abm_count")
    accom    = _col(df, "card_accommodation_count")
    travel   = _col(df, "card_travel_sum")
    ht_src   = _col(df, "ht_source_country_txn_count")
    adult_ah = _col(df, "card_adult_afterhours_count")
    card_ah  = _col(df, "card_afterhours_count")
    night_r  = _col(df, "night_transaction_ratio")
    vol      = _col(df, "total_volume")
    emt_d    = _col(df, "emt_debit_count")
    eft_d    = _col(df, "eft_debit_count")
    parking  = _col(df, "card_parking_count")
    delivery = _col(df, "card_delivery_count")
    txn      = _col(df, "transaction_count_total")

    s = (
        # HT-SEX-01/02
        np.where((retail >= 10) & (rr > 0.50), 0.15,
        np.where((retail >= 5)  & (rr > 0.50), 0.08, 0.0)) +
        # HT-SEX-03
        np.where((luxury > 10_000) & ((income == 0) | (spi > 3)), 0.25,
        np.where((luxury > 5_000)  & ((income == 0) | (spi > 2)), 0.12, 0.0)) +
        # HT-SEX-04/05
        np.where((parking >= 20) & (delivery >= 20), 0.10,
        np.where((parking >= 10) & (delivery >= 10), 0.05, 0.0)) +
        # HT-SEX-07
        np.where((cities >= 5) & (night_abm >= 5), 0.30,
        np.where((cities >= 3) & (night_abm >= 3), 0.20,
        np.where((cities >= 3) & (night_abm >= 1), 0.10, 0.0))) +
        # HT-SEX-08
        np.where(accom >= 5, 0.15, np.where(accom >= 3, 0.08, 0.0)) +
        # HT-SEX-10
        np.where((travel > 5_000) | (ht_src >= 5), 0.20,
        np.where((travel > 2_000) | (ht_src >= 3), 0.10, 0.0)) +
        # HT-SEX-12 — direct MCC signal prioritised, general fallback
        np.where(adult_ah >= 3, 0.40,
        np.where(adult_ah >= 1, 0.25,
        np.where((card_ah >= 10) & (night_r > 0.40) & (txn >= 10), 0.10, 0.0))) +
        # HT-SEX-13
        np.where((income > 0) & (vol > 50_000) & (spi > 10), 0.15, 0.0) +
        # HT-SEX-14
        np.where((emt_d >= 8) | (eft_d >= 8), 0.15,
        np.where((emt_d >= 4) | (eft_d >= 4), 0.08, 0.0))
    )
    return pd.Series(s, index=df.index).clip(0, 1)


def apply_rule_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all per-typology rule scores for every customer simultaneously.
    Returns a DataFrame with one score column per typology, a weighted
    combined score, and a rules_triggered count.
    """
    scores = {
        "rule_structuring_layering": rules_structuring_layering(df),
        "rule_behavioural_profile":  rules_behavioural_profile(df),
        "rule_trade_shell":          rules_trade_shell(df),
        "rule_cross_border_geo":     rules_cross_border_geo(df),
        "rule_human_trafficking":    rules_human_trafficking(df),
    }
    out = pd.DataFrame(scores, index=df.index)

    # Weighted combination using same typology weights as IF layer
    weight_map = {
        "rule_structuring_layering": TYPOLOGY_WEIGHTS["if_score_structuring_layering"],
        "rule_behavioural_profile":  TYPOLOGY_WEIGHTS["if_score_behavioural_profile"],
        "rule_trade_shell":          TYPOLOGY_WEIGHTS["if_score_trade_shell"],
        "rule_cross_border_geo":     TYPOLOGY_WEIGHTS["if_score_cross_border_geo"],
        "rule_human_trafficking":    TYPOLOGY_WEIGHTS["if_score_human_trafficking"],
    }
    out["rule_score_weighted"] = sum(out[col] * w for col, w in weight_map.items())
    out["rules_triggered"]     = (out[list(scores.keys())] > 0).sum(axis=1)
    out["primary_rule_typology"] = (
        out[list(scores.keys())].idxmax(axis=1)
        .str.replace("rule_", "")
        .map({
            "structuring_layering": TYPOLOGY_LABELS["if_score_structuring_layering"],
            "behavioural_profile":  TYPOLOGY_LABELS["if_score_behavioural_profile"],
            "trade_shell":          TYPOLOGY_LABELS["if_score_trade_shell"],
            "cross_border_geo":     TYPOLOGY_LABELS["if_score_cross_border_geo"],
            "human_trafficking":    TYPOLOGY_LABELS["if_score_human_trafficking"],
        })
    )
    return out


# ---------------------------------------------------------------------------
# Layer 3 — KMeans clustering
# ---------------------------------------------------------------------------

def select_k(X: np.ndarray, k_range: range) -> int:
    """
    Choose k by silhouette score on a subsample.

    The silhouette score measures how similar each customer is to its own
    cluster compared to the nearest other cluster. A score near +1 means
    the customer fits well in its cluster; near 0 means ambiguous; near -1
    means it may belong in the neighbouring cluster.

    We auto-select k because the IF score space does NOT have 5 natural
    groupings just because there are 5 typologies. The majority of customers
    are low-risk across all typologies and form one large mass. Forcing k=5
    wastes clusters by fragmenting that mass into indistinct sub-groups.
    The silhouette criterion finds the k that creates the most compact,
    well-separated groupings given the actual score distribution.
    """
    n_sample   = min(5000, len(X))
    sample_idx = np.random.choice(len(X), n_sample, replace=False)
    Xs         = X[sample_idx]

    sil = {}
    print(f"\n  Silhouette scores (n_sample={n_sample:,}):")
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        sil[k] = silhouette_score(Xs, labels)
        print(f"     k={k}: {sil[k]:.4f}")

    best_k = max(sil, key=sil.get)
    print(f"  → Best k by silhouette: {best_k}")
    return best_k


def profile_clusters(df: pd.DataFrame, if_cols: list) -> pd.DataFrame:
    """
    Profile each cluster by the mean IF score of its members.

    How it works:
      1. Compute the mean of each typology IF score within each cluster.
         This tells us what kind of risk the typical cluster member has.
      2. Normalise these means across clusters so we can compare relative
         dominance — which typology is most elevated in this cluster vs
         the others. A cluster where structuring_layering averages 0.4
         while the population average is 0.2 is a structuring-dominant cluster.
      3. The primary typology label is the highest normalised dimension,
         but only if it exceeds 0.25 — below this no typology dominates
         and the cluster is labelled General Low Risk.
      4. Risk tier is based on the mean of ALL IF scores for that cluster
         (not the max). Mean is more representative of the typical member;
         max is distorted by a handful of extreme outliers within the cluster.

    Risk tiers:
      High   (1.0) — cluster mean risk > 67th percentile across clusters
      Medium (0.5) — cluster mean risk > 33rd percentile
      Low    (0.0) — cluster mean risk <= 33rd percentile
    """
    profile = df.groupby("cluster")[if_cols].mean()
    p_norm  = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    def assign_label(row):
        return "General — Low Risk" if row.max() < 0.25 \
               else TYPOLOGY_LABELS.get(row.idxmax(), row.idxmax())

    primary   = p_norm.apply(assign_label, axis=1)
    mean_risk = profile.mean(axis=1)
    p33, p67  = mean_risk.quantile(0.33), mean_risk.quantile(0.67)
    risk_tier = mean_risk.apply(lambda x: 1.0 if x > p67 else (0.5 if x > p33 else 0.0))

    summary = pd.DataFrame({"primary_typology": primary, "risk_tier": risk_tier})

    k = len(profile)
    short = [c.replace("if_score_", "")[:14] for c in if_cols]
    print(f"\n  Cluster profiles (k={k}, mean IF scores):")
    print("  {:>3}  {:>7}  {}  {:>5}  Label".format(
        "Cls", "n", "  ".join(f"{h:>14}" for h in short), "Tier"))
    print("  " + "-" * 110)
    for c in profile.index:
        n     = (df["cluster"] == c).sum()
        vals  = "  ".join(f"{profile.loc[c, col]:>14.3f}" for col in if_cols)
        tier  = {0.0: "LOW", 0.5: "MED", 1.0: "HIGH"}[risk_tier[c]]
        label = summary.loc[c, "primary_typology"]
        print(f"  {c:>3}  {n:>7,}  {vals}  {tier:>5}  {label}")

    return summary


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def risk_category(score: float) -> str:
    if score >= 0.80: return "Very High"
    if score >= 0.60: return "High"
    if score >= 0.40: return "Medium"
    if score >= 0.20: return "Low"
    return "Very Low"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AML Hybrid Model")
    parser.add_argument("--base_dir", type=str, default="/content/gdrive/MyDrive/AML_Competition")
    parser.add_argument("--k_min",   type=int, default=3,
                        help="Min k for silhouette search (default: 3)")
    parser.add_argument("--k_max",   type=int, default=8,
                        help="Max k for silhouette search (default: 8)")
    parser.add_argument("--k",       type=int, default=0,
                        help="Fix k directly (0 = auto-select, default)")
    args = parser.parse_args()

    base_dir  = Path(args.base_dir)
    feat_dir  = base_dir / "features"
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("HYBRID AML MODEL — IF + Rules + KMeans")
    print("5 Typologies — AML-indicator-DB.xlsx")
    print("=" * 70)
    print(f"\nEnsemble weights:  IF={W_IF}  |  Rules={W_RULE}  |  Cluster={W_CLUSTER}")
    tw = {k.replace("if_score_", ""): v for k, v in TYPOLOGY_WEIGHTS.items()}
    print(f"Typology weights:  {tw}")
    print("(Weights grounded in IF validation ROC-AUC — not tuned on labels)")

    # ── Load ───────────────────────────────────────────────────────────────
    print("\nLoading data...")
    iso_results = pd.read_csv(model_dir / "isolation_forest_results.csv")
    df_features = pd.read_csv(feat_dir  / "customer_features_enhanced.csv")
    print(f"   IF results:     {len(iso_results):,} customers")
    print(f"   Feature matrix: {df_features.shape}")

    avail_if_cols = [c for c in IF_SCORE_COLS if c in iso_results.columns]
    missing_if    = [c for c in IF_SCORE_COLS if c not in iso_results.columns]
    if missing_if:
        print(f"   WARNING: Missing IF columns — run 02_isolation_forest.py first: {missing_if}")

    labels = iso_results["actual_label"].copy() if "actual_label" in iso_results.columns \
             else pd.Series([np.nan] * len(iso_results))

    # ── Layer 2: Rule-based scoring (vectorised) ───────────────────────────
    print("\n" + "-" * 70)
    print("Layer 2 — Rule-based scoring (vectorised, per typology)...")
    print("-" * 70)

    rule_out    = apply_rule_scores(df_features)
    df_features = pd.concat([df_features, rule_out], axis=1)

    rule_cols = [c for c in rule_out.columns
                 if c.startswith("rule_") and
                 c not in ("rule_score_weighted", "rules_triggered", "primary_rule_typology")]

    print(f"\n  Per-typology rule scores:")
    print(f"  {'Typology':<30}  {'Triggered':>10}  {'Mean':>6}  {'P95':>6}  {'Max':>6}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*6}")
    for col in rule_cols:
        name = col.replace("rule_", "")
        n    = (df_features[col] > 0).sum()
        print(f"  {name:<30}  {n:>10,}  "
              f"{df_features[col].mean():>6.3f}  "
              f"{df_features[col].quantile(0.95):>6.3f}  "
              f"{df_features[col].max():>6.3f}")

    print(f"\n  Weighted rule score — "
          f"mean={df_features['rule_score_weighted'].mean():.3f}  "
          f"p95={df_features['rule_score_weighted'].quantile(0.95):.3f}  "
          f"max={df_features['rule_score_weighted'].max():.3f}")
    print(f"  Customers with any rule triggered: "
          f"{(df_features['rules_triggered'] > 0).sum():,}")

    # ── Layer 3: KMeans clustering ─────────────────────────────────────────
    print("\n" + "-" * 70)
    print("Layer 3 — KMeans clustering on typology IF scores...")
    print("-" * 70)

    if_scores_df = iso_results[["customer_id"] + avail_if_cols].copy()
    df_cluster   = df_features.merge(if_scores_df, on="customer_id", how="left")
    df_cluster[avail_if_cols] = df_cluster[avail_if_cols].fillna(0)
    X_cluster    = df_cluster[avail_if_cols].values

    k_range = range(args.k_min, args.k_max + 1)
    best_k  = args.k if args.k > 0 else select_k(X_cluster, k_range)

    print(f"\n  Training KMeans with k={best_k}...")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    kmeans.fit(X_cluster)
    df_cluster["cluster"] = kmeans.labels_

    cluster_summary = profile_clusters(df_cluster, avail_if_cols)
    df_cluster["cluster_primary_typology"] = df_cluster["cluster"].map(cluster_summary["primary_typology"])
    df_cluster["cluster_risk_tier"]        = df_cluster["cluster"].map(cluster_summary["risk_tier"])

    # ── Ensemble fusion ────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("Ensemble fusion...")
    print("-" * 70)

    # Weighted IF score (Layer 1)
    iso_results["if_score_weighted"] = sum(
        iso_results[col] * TYPOLOGY_WEIGHTS[col]
        for col in avail_if_cols if col in TYPOLOGY_WEIGHTS
    )

    merge_cluster = df_cluster[["customer_id", "cluster",
                                 "cluster_risk_tier", "cluster_primary_typology"]]
    merge_rules   = df_features[["customer_id", "rule_score_weighted",
                                  "rules_triggered", "primary_rule_typology"] + rule_cols]

    hybrid = (
        iso_results[["customer_id", "if_score_weighted", "if_score_max",
                      "if_score_mean", "primary_typology", "actual_label"] + avail_if_cols]
        .merge(merge_cluster, on="customer_id", how="inner")
        .merge(merge_rules,   on="customer_id", how="inner")
    )

    hybrid["final_hybrid_score"] = (
        hybrid["if_score_weighted"]   * W_IF   +
        hybrid["rule_score_weighted"] * W_RULE +
        hybrid["cluster_risk_tier"]   * W_CLUSTER
    )

    # Normalise to [0, 1]
    mn = hybrid["final_hybrid_score"].min()
    mx = hybrid["final_hybrid_score"].max()
    hybrid["final_hybrid_score"] = (hybrid["final_hybrid_score"] - mn) / (mx - mn)

    hybrid = hybrid.sort_values("final_hybrid_score", ascending=False).reset_index(drop=True)
    hybrid["hybrid_risk_category"] = hybrid["final_hybrid_score"].apply(risk_category)

    print(f"\n  Component correlations with final score:")
    for col in ["if_score_weighted", "rule_score_weighted", "cluster_risk_tier"]:
        print(f"    {col:<30}: r={hybrid[col].corr(hybrid['final_hybrid_score']):.3f}")

    print(f"\n  Risk category distribution:")
    print(hybrid["hybrid_risk_category"].value_counts().to_string())
    print(f"\n  Primary typology (top 1,000 customers):")
    print(hybrid.head(1000)["cluster_primary_typology"].value_counts().to_string())

    # ── Validation ─────────────────────────────────────────────────────────
    labeled = hybrid["actual_label"].notna()
    if labeled.sum() > 0:
        susp     = hybrid[hybrid["actual_label"] == 1]
        susp_idx = set(susp.index)
        n_total  = len(hybrid)

        try:
            roc_hybrid = roc_auc_score(hybrid.loc[labeled, "actual_label"],
                                        hybrid.loc[labeled, "final_hybrid_score"])
            roc_if_wtd = roc_auc_score(hybrid.loc[labeled, "actual_label"],
                                        hybrid.loc[labeled, "if_score_weighted"])
            roc_if_max = roc_auc_score(hybrid.loc[labeled, "actual_label"],
                                        hybrid.loc[labeled, "if_score_max"])
            roc_rule   = roc_auc_score(hybrid.loc[labeled, "actual_label"],
                                        hybrid.loc[labeled, "rule_score_weighted"])
        except Exception:
            roc_hybrid = roc_if_wtd = roc_if_max = roc_rule = 0.0

        print("\n" + "=" * 70)
        print("FINAL VALIDATION  (directional only — 10 positives, high variance)")
        print("=" * 70)
        print(f"  ROC-AUC:")
        print(f"    Hybrid (weighted ensemble):  {roc_hybrid:.3f}")
        print(f"    IF weighted alone:           {roc_if_wtd:.3f}")
        print(f"    IF max alone:                {roc_if_max:.3f}")
        print(f"    Rules alone:                 {roc_rule:.3f}")

        print(f"\n  Capture rates:")
        for pct in [0.01, 0.05, 0.10]:
            cutoff   = int(n_total * pct)
            captured = len(susp_idx & set(range(cutoff)))
            print(f"    Top {pct*100:.0f}% ({cutoff:>5,} customers): "
                  f"{captured}/10 captured ({captured*10:.0f}%)")

        print(f"\n  Suspicious customer breakdown:")
        print(f"  {'#':>3}  {'Rank':>7}  {'Hybrid':>7}  {'IF_w':>6}  "
              f"{'Rule':>6}  {'Clust':>5}  Typology")
        print(f"  {'-'*80}")
        for i, (idx, row) in enumerate(susp.iterrows(), 1):
            print(f"  {i:>3}  {idx+1:>7,}  {row['final_hybrid_score']:>7.3f}  "
                  f"{row['if_score_weighted']:>6.3f}  "
                  f"{row['rule_score_weighted']:>6.3f}  "
                  f"{row['cluster_risk_tier']:>5.1f}  "
                  f"{str(row['cluster_primary_typology'])[:35]}")

    # ── Save ───────────────────────────────────────────────────────────────
    hybrid.to_csv(model_dir / "hybrid_model_results.csv", index=False)
    hybrid[hybrid["final_hybrid_score"] > 0.60].to_csv(
        model_dir / "hybrid_model_high_risk.csv", index=False)

    results = hybrid[["customer_id", "final_hybrid_score"]].copy()
    results["predicted_label"] = (
        results["final_hybrid_score"] >= results["final_hybrid_score"].quantile(0.99)
    ).astype(int)
    results = results.rename(columns={"final_hybrid_score": "risk_score"})[
        ["customer_id", "predicted_label", "risk_score"]]
    results.to_csv(model_dir / "results.csv", index=False)

    evidence_cols = (
        ["customer_id", "final_hybrid_score", "hybrid_risk_category",
         "cluster_primary_typology", "primary_rule_typology",
         "if_score_weighted", "if_score_max", "rule_score_weighted",
         "rules_triggered", "cluster_risk_tier"]
        + avail_if_cols + rule_cols
    )
    hybrid[[c for c in evidence_cols if c in hybrid.columns]]\
        [hybrid["final_hybrid_score"] > 0.60]\
        .to_csv(model_dir / "high_risk_evidence.csv", index=False)

    joblib.dump(kmeans, model_dir / "kmeans_model.pkl")
    joblib.dump(best_k, model_dir / "kmeans_k.pkl")
    cluster_summary.to_csv(model_dir / "cluster_typology_profile.csv")

    print("\n" + "=" * 70)
    print("HYBRID MODEL COMPLETE")
    print("=" * 70)
    print(f"   Customers scored:    {len(hybrid):,}")
    print(f"   High-risk (>0.60):   {(hybrid['final_hybrid_score']>0.60).sum():,}")
    print(f"   Flagged (top 1%):    {results['predicted_label'].sum():,}")
    print(f"   Clusters used:       k={best_k}")
    print(f"   Score range:         [{results['risk_score'].min():.3f}, "
          f"{results['risk_score'].max():.3f}]")
    print(f"\n   Output files:")
    print(f"     results.csv                ← competition submission")
    print(f"     hybrid_model_results.csv   ← full scored table")
    print(f"     high_risk_evidence.csv     ← investigator evidence")


if __name__ == "__main__":
    main()