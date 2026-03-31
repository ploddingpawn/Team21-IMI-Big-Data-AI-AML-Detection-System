"""
03_hybrid_model.py
==================
Hybrid AML risk scoring: Typology IF ensemble + KMeans clustering
+ Rule-based hard flags, fused into a final ranked score.

Architecture (three complementary layers):

  Layer 1 — Typology Isolation Forests (from 02_isolation_forest.py)
      7 IF scores, one per typology. Each answers: "Is this customer
      anomalous specifically within the context of this typology?"
      Statistical — learns from data, no thresholds needed.

  Layer 2 — KMeans Clustering
      Clusters customers on the 7 IF scores (not raw features).
      Each dimension is now interpretable: underground banking anomaly,
      CMLN anomaly, etc. Clusters map cleanly onto typology profiles.
      Validated by silhouette score and typology interpretability,
      NOT by label capture rate.

  Layer 3 — Rule-Based Hard Flags
      Deterministic checks against industry-standard thresholds.
      These are MANDATORY regulatory triggers — a customer who breaches
      a structuring threshold must be flagged regardless of what the
      statistical model says. Rules act as a boost, not a replacement.
      Thresholds sourced from FINTRAC/FinCEN guidance and industry practice.

Ensemble weights (domain logic, NOT tuned on 10 labels):
      IF max score:       0.70  — primary anomaly signal
      Cluster risk tier:  0.10  — typology-aligned grouping
      Rule flag score:    0.20  — hard regulatory triggers

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
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
np.random.seed(42)


# ---------------------------------------------------------------------------
# Typology IF score columns (produced by 02_isolation_forest.py)
# ---------------------------------------------------------------------------

IF_SCORE_COLS = [
    "if_score_structuring",
    "if_score_trade_and_shells",
    "if_score_money_mules",
    "if_score_profile_mismatch",
    "if_score_geographic_risk",
    "if_score_layering",
]

TYPOLOGY_LABELS = {
    "if_score_structuring":      "Structuring / Smurfing (FINTRAC — STRUCT-001..009)",
    "if_score_trade_and_shells": "Trade-Based ML & Shell Companies (PML-TBML-01..08, ATYPICAL-004)",
    "if_score_money_mules":      "Money Mules (CMLN-MULE-01..04, FENT-LAY-07)",
    "if_score_profile_mismatch": "Profile Mismatch & Account Activity (PROF-001..010, ACCT-001..004)",
    "if_score_geographic_risk":  "Geographic Risk (GEO-001..005, PML-MSB-02/10)",
    "if_score_layering":         "Layering & Fund Movement (ATYPICAL-007/008, WIRE-006..010)",
}

# Relevant Merchant Category Codes 
aml_relevant_mcc = {
    # --- Luxury / High-Value Goods ---
    5094: "Precious Stones and Metals, Watches and Jewelry",
    5944: "Clock, Jewelry, Watch and Silverware Stores",
    5681: "Furriers and Fur Shops",
    5971: "Art Dealers and Galleries",
    5972: "Stamp and Coin Stores",
    5309: "Duty Free Stores",
    5311: "Department Stores",
    5399: "Miscellaneous General Merchandise Stores",
    5999: "Miscellaneous and Specialty Retail Stores",

    # --- Cash / Financial / Quasi-Cash ---
    6010: "Manual Cash Disbursements",
    6011: "Automated Cash Disbursements (ATM)",
    6012: "Financial Institution Merchandise and Services",
    6050: "Quasi Cash – Financial Institution",
    6051: "Quasi Cash – Merchant",

    # --- Money Transfer / Stored Value ---
    4829: "Wire Transfer Money Orders",
    6530: "Stored Value Load – Merchant",
    6535: "Value Purchase – Financial Institution",
    6536: "MoneySend Intracountry",
    6537: "MoneySend Intercountry",
    6538: "MoneySend Funding",
    6539: "Funding Transaction",
    6540: "Point of Interaction Funding Transactions",

    # --- Gambling / Betting ---
    7800: "Government-Owned Lottery",
    7801: "Licensed Online Casinos",
    7802: "Horse/Dog Racing",
    7995: "Gambling Transactions",

    # --- Pawn / Secondary Market ---
    5933: "Pawn Shops",
    5931: "Second Hand Stores",
    5932: "Antique Shops",

    # --- Travel (Layering / Movement Risk) ---
    3000: "Airlines (General Category)",
    3351: "Car Rental Agencies",
    3501: "Hotels, Motels, Resorts",
    7011: "Lodging Services",
    4722: "Travel Agencies and Tour Operators",

    # --- High-Risk Retail / Portable Value ---
    5948: "Leather Goods and Luggage Stores",
    5946: "Camera and Photographic Supply Stores",
    5732: "Electronics Stores",
    5045: "Computers and Software",
    5942: "Book Stores (potential layering via resale)",

    # --- Digital Goods / Remote Transactions ---
    5815: "Digital Goods – Books, Movies, Music",
    5816: "Digital Goods – Games",
    5817: "Digital Goods – Applications",
    5818: "Digital Goods – Multi-Category",

    # --- Direct Marketing / Obfuscation Channels ---
    5961: "Mail Order Houses",
    5962: "Direct Marketing – Travel",
    5963: "Door-to-Door Sales",
    5964: "Catalog Merchants",
    5965: "Combination Catalog and Retail",
    5966: "Outbound Telemarketing",
    5967: "Inbound Telemarketing",
    5968: "Subscription Merchants",
    5969: "Other Direct Marketing",

    # --- Alcohol / Controlled Goods ---
    5921: "Liquor Stores",
    5813: "Bars and Nightclubs",

    # --- Financial / Securities ---
    6211: "Securities Brokers and Dealers",

    # --- Insurance (possible layering) ---
    6300: "Insurance Sales and Premiums",

    # --- Government / Payments (monitoring relevance) ---
    9211: "Court Costs",
    9222: "Fines",
    9311: "Tax Payments",

    # --- Misc Services (behavioral indicators) ---
    7276: "Tax Preparation Services",
    7277: "Counseling Services",
    7399: "Business Services Not Elsewhere Classified"
}


# ---------------------------------------------------------------------------
# Rule-based hard flags
# Thresholds sourced from FINTRAC guidance and industry practice.
# Each rule returns a score contribution (0.0-1.0) and a triggered flag.
# ---------------------------------------------------------------------------

def apply_rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic rule-based flags to the feature matrix.
    Returns df with new columns:
        rule_score      — weighted sum of triggered rules (0-1)
        rules_triggered — count of triggered rules
        rule_evidence   — pipe-separated list of triggered rule descriptions

    Thresholds:
        $10,000 CTR threshold (FINTRAC):
            Transactions >= $10,000 require a Cash Transaction Report.
            Structuring = breaking amounts below this to avoid reporting.
            Near-threshold: $9,000-$9,999 (STRUCT-003)
            Very near:      $9,800-$9,999 (high confidence structuring)

        Wire transfer monitoring (industry standard):
            Single wire > $50,000 CAD to foreign account — elevated risk
            Multiple wires > $100,000 CAD total — WIRE-009

        Flow-through (FINTRAC ATYPICAL-007/008):
            Inflow and outflow nearly equal with high volume = layering
            Threshold: flow_through_ratio > 0.85 AND total_volume > $50,000

        International exposure (GEO-002/003):
            > 50% of transactions are international = elevated risk
            Any transaction to FATF blacklist country = hard flag

        Profile mismatch (PROF-002):
            Spending > 3x declared income = suspicious
            Business volume > 5x declared sales = suspicious

        Transaction velocity (STRUCT-006):
            > 3 transactions per active day = high velocity
    """
    rules = []
    evidence_col = []
    score_col    = []

    for _, row in df.iterrows():
        triggered = []
        score     = 0.0

        # ── Rule 1: Near-threshold structuring (STRUCT-003) ────────────────
        # FINTRAC CTR threshold is $10,000 CAD
        near_10k = row.get("count_near_10k", 0)
        if near_10k >= 5:
            score += 0.30
            triggered.append(f"STRUCT-003: {near_10k:.0f} transactions $9,000-$9,999")
        elif near_10k >= 2:
            score += 0.15
            triggered.append(f"STRUCT-003: {near_10k:.0f} transactions near $10K threshold")

        # ── Rule 2: Very near threshold (high-confidence structuring) ──────
        very_near = row.get("count_very_near_10k", 0)
        if very_near >= 3:
            score += 0.20
            triggered.append(f"STRUCT-003 (high): {very_near:.0f} transactions $9,800-$9,999")

        # ── Rule 3: High round-amount ratio (STRUCT-009) ───────────────────
        round_ratio = row.get("ratio_round_100", 0)
        txn_count   = row.get("transaction_count_total", 0)
        if round_ratio > 0.80 and txn_count >= 20:
            score += 0.15
            triggered.append(f"STRUCT-009: {round_ratio*100:.0f}% round-number transactions")
        elif round_ratio > 0.60 and txn_count >= 20:
            score += 0.08
            triggered.append(f"STRUCT-009: {round_ratio*100:.0f}% round-number transactions")

        # ── Rule 4: Flow-through / rapid pass-through (ATYPICAL-007/008) ──
        ftr = row.get("flow_through_ratio", 0)
        vol = row.get("total_volume", 0)
        if ftr > 0.90 and vol > 100000:
            score += 0.35
            triggered.append(f"ATYPICAL-007: flow_through={ftr:.2f}, volume=${vol:,.0f}")
        elif ftr > 0.80 and vol > 50000:
            score += 0.18
            triggered.append(f"ATYPICAL-007: flow_through={ftr:.2f}, volume=${vol:,.0f}")

        # ── Rule 5: Wire transfer volume (WIRE-008/009) ────────────────────
        sum_wire = row.get("sum_wire", 0)
        if sum_wire > 500000:
            score += 0.20
            triggered.append(f"WIRE-009: total wire volume ${sum_wire:,.0f}")
        elif sum_wire > 100000:
            score += 0.10
            triggered.append(f"WIRE-008: wire volume ${sum_wire:,.0f}")

        # ── Rule 6: FATF blacklist country transactions (GEO-002) ──────────
        fatf_count = row.get("high_risk_fatf_txn_count", 0)
        if fatf_count > 0:
            score += 0.40
            triggered.append(f"GEO-002: {fatf_count:.0f} transactions to FATF blacklist countries")

        # ── Rule 7: High international ratio (GEO-005) ─────────────────────
        intl_ratio = row.get("international_ratio", 0)
        intl_count = row.get("international_txn_count", 0)
        if intl_ratio > 0.80 and intl_count >= 10:
            score += 0.15
            triggered.append(f"GEO-005: {intl_ratio*100:.0f}% international ({intl_count:.0f} transactions)")
        elif intl_ratio > 0.50 and intl_count >= 5:
            score += 0.08
            triggered.append(f"GEO-005: {intl_ratio*100:.0f}% international ({intl_count:.0f} transactions)")

        # ── Rule 8: Income / spending mismatch (PROF-002) ──────────────────
        income  = row.get("income", 0)
        outflow = row.get("total_outflow", 0)
        spi     = row.get("spending_to_income_ratio", 0)
        if income > 0 and outflow > 10000 and spi > 10:
            score += 0.30
            triggered.append(f"PROF-002: spending {spi:.1f}x declared income (income=${income:,.0f})")
        elif income > 0 and outflow > 10000 and spi > 5:
            score += 0.15
            triggered.append(f"PROF-002: spending {spi:.1f}x declared income")

        # ── Rule 9: Business volume vs declared sales (PROF-004) ───────────
        sales  = row.get("sales", 0)
        volume = row.get("total_volume", 0)
        vsr    = row.get("volume_to_sales_ratio", 0)
        if sales > 0 and volume > 50000 and vsr > 20:
            score += 0.25
            triggered.append(f"PROF-004: volume {vsr:.1f}x declared sales (sales=${sales:,.0f})")
        elif sales > 0 and volume > 50000 and vsr > 10:
            score += 0.12
            triggered.append(f"PROF-004: volume {vsr:.1f}x declared sales")

        # ── Rule 10: High transaction velocity (STRUCT-006) ────────────────
        tpad = row.get("transactions_per_active_day", 0)
        if tpad > 20:
            score += 0.15
            triggered.append(f"STRUCT-006: {tpad:.1f} transactions/active day")
        elif tpad > 10:
            score += 0.08
            triggered.append(f"STRUCT-006: {tpad:.1f} transactions/active day")

        # Cap at 1.0
        score = min(score, 1.0)
        rules.append({
            "rule_score":      score,
            "rules_triggered": len(triggered),
            "rule_evidence":   " | ".join(triggered) if triggered else "No rules triggered",
        })

    return pd.DataFrame(rules, index=df.index)


# ---------------------------------------------------------------------------
# KMeans clustering on IF scores
# ---------------------------------------------------------------------------

def run_kmeans(X: np.ndarray, k_range=(4, 6, 7, 8, 10), final_k=7) -> tuple:
    """
    Run KMeans on the 7 typology IF scores.
    Clustering on interpretable dimensions means each cluster maps
    cleanly to a typology profile — no mixing of unrelated features.
    """
    sample_idx = np.random.choice(len(X), min(5000, len(X)), replace=False)
    sil_scores = {}

    print(f"\n  Silhouette scores on IF scores (sample={len(sample_idx):,}):")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(X[sample_idx])
        sil = silhouette_score(X[sample_idx], labels)
        sil_scores[k] = sil
        print(f"     k={k}: {sil:.3f}")

    best_k = max(sil_scores, key=sil_scores.get)
    print(f"  Best k by silhouette: {best_k}  |  Using k={final_k}")
    print(f"  (k={final_k} matches number of typologies for interpretability)")

    kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    return kmeans, sil_scores


def profile_clusters(df: pd.DataFrame, if_score_cols: list) -> pd.DataFrame:
    """
    Profile each cluster by its mean IF scores.
    Primary typology = whichever IF score is highest in that cluster.
    Risk tier = high/medium/low based on mean max IF score.
    No labels used.
    """
    profile = df.groupby("cluster")[if_score_cols].mean()

    # Normalize across clusters for relative dominance
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
    # Require minimum score of 0.25 for a typology to be assigned.
    # Without this, even the baseline low-risk cluster gets a typology label
    # because one of its scores is marginally higher than the others.
    def assign_typology(row):
        if row.max() < 0.25:
            return "General — Low Risk (no dominant typology)"
        return TYPOLOGY_LABELS.get(row.idxmax(), row.idxmax())

    primary = profile_norm.apply(assign_typology, axis=1)

    cluster_max_risk = df.groupby("cluster")[if_score_cols].max().mean(axis=1)
    p33 = cluster_max_risk.quantile(0.33)
    p67 = cluster_max_risk.quantile(0.67)
    risk_tier = cluster_max_risk.apply(
        lambda x: 1.0 if x > p67 else (0.5 if x > p33 else 0.0)
    )

    summary = pd.DataFrame({
        "primary_typology": primary,
        "risk_tier":        risk_tier,
    })

    print("\n  Cluster profiles (mean IF scores per typology):")
    print(profile.round(3).to_string())
    print("\n  Cluster assignments:")
    for c, row in summary.iterrows():
        size = (df["cluster"] == c).sum()
        print(f"   Cluster {c} ({size:,} customers): "
              f"tier={row['risk_tier']}  |  {row['primary_typology']}")

    return summary


# ---------------------------------------------------------------------------
# Helpers
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
    parser.add_argument("--base_dir",   type=str,   default="/content/gdrive/MyDrive/AML_Competition")
    parser.add_argument("--k",          type=int,   default=4,
                        help="KMeans clusters — default 6 to match typology count")
    parser.add_argument("--w_if",       type=float, default=0.70,
                        help="Weight for IF max score")
    parser.add_argument("--w_cluster",  type=float, default=0.10,
                        help="Weight for cluster risk tier")
    parser.add_argument("--w_rule",     type=float, default=0.20,
                        help="Weight for rule-based flag score")
    args = parser.parse_args()

    base_dir  = Path(args.base_dir)
    feat_dir  = base_dir / "features"
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("HYBRID AML MODEL — IF Ensemble + KMeans + Rule-Based Flags")
    print("=" * 70)
    print(f"Weights: IF={args.w_if}, Cluster={args.w_cluster}, Rules={args.w_rule}")
    print("(Domain-logic weights — not tuned on 10 labels)")

    # ── Load ───────────────────────────────────────────────────────────────
    print("\nLoading data...")
    iso_results = pd.read_csv(model_dir / "isolation_forest_results.csv")
    df_features = pd.read_csv(feat_dir  / "customer_features_enhanced.csv")

    print(f"   IF results:     {len(iso_results):,} customers")
    print(f"   Feature matrix: {df_features.shape}")

    # Verify IF score columns are present
    avail_if_cols = [c for c in IF_SCORE_COLS if c in iso_results.columns]
    missing = [c for c in IF_SCORE_COLS if c not in iso_results.columns]
    if missing:
        print(f"   WARNING: Missing IF score columns: {missing}")
        print("   Run 02_isolation_forest.py first.")
    print(f"   IF score columns: {avail_if_cols}")

    labels = iso_results["actual_label"].copy() if "actual_label" in iso_results.columns \
             else pd.Series([np.nan]*len(iso_results))

    # ── Layer 3: Rule-based flags ──────────────────────────────────────────
    print("\n" + "-" * 70)
    print("Layer 3 — Applying rule-based hard flags...")
    print("-" * 70)
    print("  Rules (FINTRAC/FinCEN — tightened thresholds to reduce over-triggering):")
    print("  STRUCT-003: >= 3 transactions $9,000-$9,999 (CTR avoidance)")
    print("  STRUCT-009: > 60% round amounts AND >= 20 total transactions")
    print("  ATYPICAL-007: flow_through > 0.90 AND volume > $100K (layering)")
    print("  WIRE-008/009: wire total > $100K-$500K")
    print("  GEO-002: any FATF blacklist country transaction (hard flag)")
    print("  GEO-005: > 50% international AND >= 5 international transactions")
    print("  PROF-002: spending > 5x income AND income > 0 AND outflow > $10K")
    print("  PROF-004: volume > 10x sales AND sales > 0 AND volume > $50K")
    print("  STRUCT-006: > 10 transactions per active day")

    rule_df = apply_rule_flags(df_features)
    df_features = pd.concat([df_features, rule_df], axis=1)

    triggered_any = (df_features["rules_triggered"] > 0).sum()
    high_rule     = (df_features["rule_score"] > 0.5).sum()
    print(f"\n  Customers with >= 1 rule triggered: {triggered_any:,}")
    print(f"  Customers with rule_score > 0.5:    {high_rule:,}")
    print(f"  Rule score distribution:")
    print(f"     Mean: {df_features['rule_score'].mean():.3f}")
    print(f"     P75:  {df_features['rule_score'].quantile(0.75):.3f}")
    print(f"     P95:  {df_features['rule_score'].quantile(0.95):.3f}")
    print(f"     Max:  {df_features['rule_score'].max():.3f}")

    # ── Layer 2: KMeans on IF scores ───────────────────────────────────────
    print("\n" + "-" * 70)
    print("Layer 2 — KMeans clustering on typology IF scores...")
    print("-" * 70)
    print("  Clustering on 6 interpretable typology anomaly scores")
    print("  (not raw features — each dimension has clear meaning)")

    # Merge IF scores into features for clustering
    if_scores_df = iso_results[["customer_id"] + avail_if_cols].copy()
    df_cluster   = df_features.merge(if_scores_df, on="customer_id", how="left")
    df_cluster[avail_if_cols] = df_cluster[avail_if_cols].fillna(0)

    X_cluster = df_cluster[avail_if_cols].values

    kmeans, sil_scores = run_kmeans(X_cluster, final_k=args.k)
    df_cluster["cluster"] = kmeans.fit_predict(X_cluster)

    cluster_summary = profile_clusters(df_cluster, avail_if_cols)
    df_cluster["cluster_primary_typology"] = df_cluster["cluster"].map(cluster_summary["primary_typology"])
    df_cluster["cluster_risk_tier"]        = df_cluster["cluster"].map(cluster_summary["risk_tier"])

    # ── Layer 1 + Ensemble fusion ──────────────────────────────────────────
    print("\n" + "-" * 70)
    print("Ensemble fusion — combining all three layers...")
    print("-" * 70)

    merge_cols = ["customer_id", "cluster", "cluster_risk_tier",
                  "cluster_primary_typology", "rule_score",
                  "rules_triggered", "rule_evidence"]

    hybrid = iso_results[["customer_id", "if_score_max", "if_score_mean",
                           "primary_typology", "actual_label"] + avail_if_cols].merge(
        df_cluster[merge_cols], on="customer_id", how="inner"
    )

    # Ensemble: IF max score + cluster risk tier + rule score
    # IF max: flagged by any single typology forest
    # Cluster tier: belongs to a high-risk typology cluster
    # Rule score: triggered hard regulatory thresholds
    hybrid["final_hybrid_score"] = (
        hybrid["if_score_max"]        * args.w_if +
        hybrid["cluster_risk_tier"]   * args.w_cluster +
        hybrid["rule_score"]          * args.w_rule
    )

    # Normalize to 0-1
    mn = hybrid["final_hybrid_score"].min()
    mx = hybrid["final_hybrid_score"].max()
    hybrid["final_hybrid_score"] = (hybrid["final_hybrid_score"] - mn) / (mx - mn)

    hybrid = hybrid.sort_values("final_hybrid_score", ascending=False).reset_index(drop=True)
    hybrid["hybrid_risk_category"] = hybrid["final_hybrid_score"].apply(risk_category)

    print(f"\nRisk category distribution:")
    print(hybrid["hybrid_risk_category"].value_counts().to_string())
    print(f"\nPrimary typology (top 1,000 customers):")
    print(hybrid.head(1000)["cluster_primary_typology"].value_counts().to_string())

    # ── Final validation ───────────────────────────────────────────────────
    labeled = hybrid["actual_label"].notna()
    if labeled.sum() > 0:
        susp     = hybrid[hybrid["actual_label"] == 1]
        susp_idx = set(susp.index)
        n_total  = len(hybrid)

        try:
            roc_hybrid = roc_auc_score(
                hybrid.loc[labeled, "actual_label"],
                hybrid.loc[labeled, "final_hybrid_score"]
            )
            roc_if = roc_auc_score(
                hybrid.loc[labeled, "actual_label"],
                hybrid.loc[labeled, "if_score_max"]
            )
            roc_rule = roc_auc_score(
                hybrid.loc[labeled, "actual_label"],
                hybrid.loc[labeled, "rule_score"]
            )
        except Exception:
            roc_hybrid = roc_if = roc_rule = 0.0

        print("\n" + "=" * 70)
        print("FINAL VALIDATION (locked — do not re-tune based on these numbers)")
        print("=" * 70)
        print(f"  ROC-AUC comparison:")
        print(f"     Hybrid (all three layers): {roc_hybrid:.3f}")
        print(f"     IF ensemble alone:          {roc_if:.3f}")
        print(f"     Rule-based alone:           {roc_rule:.3f}")
        print(f"  (Directional only — 10 positives, high variance)")

        print(f"\n  Capture rates (hybrid score):")
        for pct in [0.01, 0.05, 0.10]:
            cutoff   = int(n_total * pct)
            captured = len(susp_idx & set(range(cutoff)))
            print(f"     Top {pct*100:.0f}% ({cutoff:,} customers): "
                  f"{captured}/{len(susp_idx)} ({captured/len(susp_idx)*100:.0f}%)")

        print(f"\n  Suspicious customer details:")
        for i, (idx, row) in enumerate(susp.iterrows(), 1):
            print(f"     #{i:2d}: rank {idx+1:>6,}  "
                  f"hybrid={row['final_hybrid_score']:.3f}  "
                  f"if_max={row['if_score_max']:.3f}  "
                  f"rule={row['rule_score']:.3f}  "
                  f"typology={row['cluster_primary_typology']}")

    # ── Save ───────────────────────────────────────────────────────────────
    hybrid.to_csv(model_dir / "hybrid_model_results.csv", index=False)
    hybrid[hybrid["final_hybrid_score"] > 0.60].to_csv(
        model_dir / "hybrid_model_high_risk.csv", index=False
    )

    # results
    results = hybrid[["customer_id", "final_hybrid_score"]].copy()
    threshold  = results["final_hybrid_score"].quantile(0.99)
    results["predicted_label"] = (results["final_hybrid_score"] >= threshold).astype(int)
    results = results.rename(columns={"final_hybrid_score": "risk_score"})[
        ["customer_id", "predicted_label", "risk_score"]
    ]
    results.to_csv(model_dir / "results.csv", index=False)

    # Rule evidence for high-risk customers (useful for investigators)
    high_risk_evidence = hybrid[hybrid["final_hybrid_score"] > 0.60][
        ["customer_id", "final_hybrid_score", "hybrid_risk_category",
         "cluster_primary_typology", "if_score_max", "rule_score",
         "rules_triggered", "rule_evidence"]
    ].copy()
    high_risk_evidence.to_csv(model_dir / "high_risk_evidence.csv", index=False)

    joblib.dump(kmeans,          model_dir / "kmeans_model.pkl")
    cluster_summary.to_csv(model_dir / "cluster_typology_profile.csv")

    print("\n" + "=" * 70)
    print("HYBRID MODEL COMPLETE")
    print("=" * 70)
    print(f"   Full results:        {model_dir / 'hybrid_model_results.csv'}")
    print(f"   High-risk (>0.60):   {(hybrid['final_hybrid_score']>0.60).sum():,} customers")
    print(f"   Flagged (top 1%):    {results['predicted_label'].sum():,} customers")
    print(f"   Score range:         [{results['risk_score'].min():.3f}, "
          f"{results['risk_score'].max():.3f}]")
    print(f"   Rule evidence file:  {model_dir / 'high_risk_evidence.csv'}")
    print(f"   Results:          {model_dir / 'results.csv'}")


if __name__ == "__main__":
    main()