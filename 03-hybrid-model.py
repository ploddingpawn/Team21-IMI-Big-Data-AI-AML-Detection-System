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
  Clusters customers in the 5D RULE score space (not IF space). Rule scores
  are grounded in FINTRAC/FinCEN regulatory thresholds and have low correlation
  with IF scores (r≈0.34), so clustering here groups customers by which
  regulatory patterns they co-trigger — independent group-level evidence.
  Continuous cluster score [0,1] = base tier × within-cluster distance from
  centroid, so outliers within a high-risk regulatory cohort score higher
  than typical members. k auto-selected by silhouette score.

Ensemble:
  final_score = 0.60 * if_weighted + 0.30 * rule_weighted + 0.10 * cluster_score
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
    7399: "Business Services Not Elsewhere Classified",

    # --- General Stores ---
    5499: "Miscellaneous Food Stores–Convenience Stores, Markets, Specialty Stores, and Vending Machines",
    7210: "Cleaning, Garment and Laundry Services",
    7211: "Laundry Services–Family and Commercial",
    5300: "Wholesale Clubs",
    5310: "Discount Stores",
    5311: "Department Stores",  # already included but belongs here conceptually
    5331: "Variety Stores",
    5399: "Miscellaneous General Merchandise Stores",  # already included
    5411: "Grocery Stores, Supermarkets",
    5499: "Miscellaneous Food Stores–Convenience Stores, Markets, Specialty Stores, and Vending Machines",
    5541: "Service Stations (With or Without Ancillary Services)",
    5542: "Automated Fuel Dispensers"
}

# Indicator maps to associated MCC codes
aml_indicator_to_mcc = {
    # --- High Value / Luxury Spending ---
    "ATYPICAL-001": [
        5094, 5944, 5681, 5971, 5972, 5309,
        5311, 5399, 5999
    ],

    # --- Excessive Credit / Cash Advances / High Spend ---
    "ATYPICAL-006": [
        6010, 6011, 6012, 6050, 6051,  # cash / quasi-cash
        5094, 5944, 5971, 5309,        # luxury
        5311, 5399, 5999               # general retail
    ],

    # --- Rapid Movement of Funds / Flow-Through ---
    "ATYPICAL-007": [
        4829,
        6530, 6535, 6536, 6537, 6538, 6539, 6540,
        6010, 6011, 6050, 6051
    ],

    # --- Incoming then Outgoing Funds Quickly ---
    "ATYPICAL-008": [
        4829,
        6530, 6535, 6536, 6537, 6538, 6539, 6540
    ],

    # --- Structuring / Smurfing ---
    "STRUCT-001": [
        6010, 6011, 6050, 6051,
        5300, 5310, 5331, 5411, 5499, 5541, 5542
    ],

    # --- Use of Multiple Accounts / Movement Channels ---
    "ACCT-001": [
        4829,
        6530, 6535, 6536, 6537, 6538, 6539, 6540,
        6211
    ],

    # --- Gambling Activity ---
    "GAMBLING-001": [
        7800, 7801, 7802, 7995
    ],

    # --- Use of Cash-Equivalent Instruments ---
    "CASH-001": [
        6010, 6011, 6012, 6050, 6051
    ],

    # --- Use of Money Transfer / Remittance ---
    "TRANSFER-001": [
        4829,
        6536, 6537, 6538, 6539, 6540
    ],

    # --- Use of General Retail for Value Cycling ---
    "RETAIL-001": [
        5300, 5310, 5311, 5331, 5399, 5411, 5499,
        5541, 5542
    ],

    # --- Purchase of Resellable / Portable Goods ---
    "RETAIL-002": [
        5948, 5946, 5732, 5045, 5942,
        5311, 5399, 5999
    ],

    # --- Pawn / Secondary Market Activity ---
    "SECONDARY-001": [
        5933, 5931, 5932
    ],

    # --- Travel / Geographic Movement ---
    "TRAVEL-001": [
        3000, 3351, 3501, 7011, 4722
    ],

    # --- Digital / Remote Transactions ---
    "DIGITAL-001": [
        5815, 5816, 5817, 5818,
        5961, 5962, 5964, 5965, 5966, 5967, 5968, 5969
    ],

    # --- Obfuscated / Non-Face-to-Face Transactions ---
    "REMOTE-001": [
        5961, 5962, 5963, 5964, 5965, 5966, 5967, 5968, 5969
    ],

    # --- Alcohol / Lifestyle Indicators ---
    "LIFESTYLE-001": [
        5921, 5813
    ],

    # --- Insurance / Financial Layering ---
    "INSURANCE-001": [
        6300
    ],

    # --- Government / Legal Payments ---
    "GOV-001": [
        9211, 9222, 9311
    ],

    # --- Service-Based / Cash-Friendly Businesses ---
    "SERVICE-001": [
        7210, 7211, 7276, 7277, 7399
    ]
}


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
    STRUCT-001:   Sub-$10K cash deposits (count_below_10k, cash_struct_count, ratio_below_10k)
    STRUCT-006:   Burst velocity — transactions_per_active_day + volume_per_day
    ATYPICAL-006: Suspicious temporal pattern — night_transaction_ratio, weekend_ratio
    ATYPICAL-007: Quick in-out / pass-through — flow_through_ratio + volume + EFT/EMT
    BEHAV-001:    Location hopping — unique_cities, unique_provinces
    PROF-006:     Large/rapid fund movement — total_volume vs income, volume_per_day
    PROF-007:     Round-sum transactions — ratio_round_100, ratio_round_1000
    """
    # ── STRUCT-001: cash structuring ─────────────────────────────────────────
    below10k     = _col(df, "count_below_10k")
    ratio_b10k   = _col(df, "ratio_below_10k")
    cash_struct  = _col(df, "cash_struct_count")

    s_struct001 = (
        np.where(below10k >= 20, 0.20, np.where(below10k >= 10, 0.10, 0.0)) +
        np.where((ratio_b10k > 0.90) & (below10k >= 10), 0.10, 0.0) +
        np.where(cash_struct >= 10, 0.15, np.where(cash_struct >= 5, 0.07, 0.0))
    )

    # ── STRUCT-006: short-period velocity ────────────────────────────────────
    tpad   = _col(df, "transactions_per_active_day")
    vpd    = _col(df, "volume_per_day")

    s_struct006 = (
        np.where(tpad > 20, 0.20, np.where(tpad > 10, 0.10, 0.0)) +
        np.where(vpd > 5_000, 0.10, np.where(vpd > 2_000, 0.05, 0.0))
    )

    # ── ATYPICAL-006: temporal pattern ───────────────────────────────────────
    night_r  = _col(df, "night_transaction_ratio")
    wknd_r   = _col(df, "weekend_ratio")
    txn      = _col(df, "transaction_count_total")

    s_aty006 = (
        np.where((night_r > 0.60) & (txn >= 20), 0.12, np.where(night_r > 0.40, 0.06, 0.0)) +
        np.where((wknd_r > 0.70) & (txn >= 20), 0.10, 0.0)
    )

    # ── ATYPICAL-007: quick in-out / pass-through ─────────────────────────────
    ftr  = _col(df, "flow_through_ratio")
    vol  = _col(df, "total_volume")
    ceft = _col(df, "count_eft")
    cemt = _col(df, "count_emt")

    s_aty007 = (
        np.where((ftr > 0.95) & (vol > 100_000), 0.40,
        np.where((ftr > 0.90) & (vol > 50_000),  0.25,
        np.where((ftr > 0.80) & (vol > 20_000),  0.12, 0.0))) +
        np.where((ceft + cemt > 20) & (ftr > 0.70), 0.10, 0.0)
    )

    # ── BEHAV-001: location hopping ──────────────────────────────────────────
    cities = _col(df, "unique_cities")
    provs  = _col(df, "unique_provinces")

    s_behav001 = (
        np.where(cities >= 5, 0.20, np.where(cities >= 3, 0.10, 0.0)) +
        np.where(provs  >= 3, 0.10, np.where(provs  >= 2, 0.05, 0.0))
    )

    # ── PROF-006: large/rapid fund movement ──────────────────────────────────
    income = _col(df, "income")
    spi    = _col(df, "spending_to_income_ratio")

    s_prof006 = (
        np.where((vol > 500_000) & (income > 0) & (spi > 5), 0.30,
        np.where((vol > 200_000) & (income > 0) & (spi > 3), 0.18, 0.0)) +
        np.where(vpd > 10_000, 0.15, np.where(vpd > 3_000, 0.08, 0.0))
    )

    # ── PROF-007: round sums ─────────────────────────────────────────────────
    rr    = _col(df, "ratio_round_100")
    rr1k  = _col(df, "ratio_round_1000")
    cr1k  = _col(df, "count_round_1000")

    s_prof007 = (
        np.where((rr > 0.80) & (txn >= 20), 0.15, np.where((rr > 0.60) & (txn >= 20), 0.08, 0.0)) +
        np.where((rr1k > 0.40) & (txn >= 10), 0.10, 0.0) +
        np.where(cr1k >= 5, 0.05, 0.0)
    )

    s = s_struct001 + s_struct006 + s_aty006 + s_aty007 + s_behav001 + s_prof006 + s_prof007
    return pd.Series(s, index=df.index).clip(0, 1)


def rules_behavioural_profile(df: pd.DataFrame) -> pd.Series:
    """
    ACCT-001:  Dormant activation — old account suddenly active
    ACCT-002:  Periodic patterns — erratic monthly frequency
    ACCT-003:  Credit surge — high card spend relative to total volume/income
    ACCT-004:  Abrupt change — high CV across volume and amounts
    PROD-008:  Credit card abuse — very high card spend + luxury
    PROF-001:  Activity vs expectation — income_vol_ratio, rapid new account
    PROF-002:  Financial standing — spending > 5x income
    PROF-005:  Living beyond means — spending > 3x income + luxury
    PROF-008:  Transaction type/size atypical — amount_cv, max_amount
    PROF-010:  Sudden change — monthly_volume_cv, monthly_txn_count_std
    """
    income    = _col(df, "income")
    outflow   = _col(df, "total_outflow")
    spi       = _col(df, "spending_to_income_ratio")
    vol       = _col(df, "total_volume")
    ivr       = _col(df, "income_vol_ratio")
    card_sum  = _col(df, "sum_card")
    card_cnt  = _col(df, "count_card")
    lux_sum   = _col(df, "card_luxury_sum")
    lux_cnt   = _col(df, "card_luxury_count")
    tenure    = _col(df, "customer_tenure_days")
    tspan     = _col(df, "time_span_days")
    active    = _col(df, "active_days")
    mv_cv     = _col(df, "monthly_volume_cv")
    mt_std    = _col(df, "monthly_txn_count_std")
    amt_cv    = _col(df, "amount_cv")
    std_amt   = _col(df, "std_transaction_amount")
    max_amt   = _col(df, "max_transaction_amount")
    avg_amt   = _col(df, "avg_transaction_amount")
    txn       = _col(df, "transaction_count_total")
    is_ind    = _col(df, "is_individual")
    hhi       = _col(df, "channel_concentration_hhi")
    max_gap   = _col(df, "max_days_between_txns")
    card_30d  = _col(df, "card_volume_30d")
    card_90d  = _col(df, "card_volume_90d")

    # ── ACCT-001: dormant activation ─────────────────────────────────────────
    s_acct001 = (
        np.where((max_gap > 180) & (vol > 10_000), 0.25,
        np.where((max_gap > 90)  & (vol >  5_000), 0.15, 0.0)) +
        np.where((mv_cv > 3.0)   & (tenure > 180), 0.12, 0.0)
    )

    # ── ACCT-002: periodic/erratic patterns ──────────────────────────────────
    s_acct002 = (
        np.where(mt_std > 50, 0.12, 0.0) +
        np.where((mv_cv > 2.0) & (active < tspan * 0.30), 0.15, 0.0) +
        np.where((hhi > 0.85) & (mt_std > 20), 0.08, 0.0)
    )

    # ── ACCT-003: card usage surge ───────────────────────────────────────────
    surge_ratio = np.where(card_90d > 0, card_30d / card_90d, 0.0)
    s_acct003 = (
        np.where((surge_ratio > 0.75) & (card_30d > 10_000), 0.20,
        np.where((surge_ratio > 0.50) & (card_30d >  5_000), 0.10, 0.0)) +
        np.where((income > 0) & (card_sum > income * 0.70) & (card_sum > 20_000), 0.15, 0.0) +
        np.where((amt_cv > 2.0) & (card_cnt > 20), 0.08, 0.0)
    )

    # ── ACCT-004: abrupt change ───────────────────────────────────────────────
    s_acct004 = (
        np.where(mv_cv > 2.5, 0.15, 0.0) +
        np.where(mt_std > 30, 0.10, 0.0) +
        np.where(amt_cv > 3.0, 0.12, 0.0) +
        np.where(std_amt > 5_000, 0.08, 0.0)
    )

    # ── PROD-008: credit card abuse ───────────────────────────────────────────
    s_prod008 = (
        np.where(card_sum > 50_000, 0.20, np.where(card_sum > 20_000, 0.10, 0.0)) +
        np.where((lux_sum > 15_000) & ((income == 0) | (spi > 3)), 0.20, 0.0) +
        np.where((lux_cnt >= 5) & (income > 0) & (lux_sum > income * 0.5), 0.10, 0.0)
    )

    # ── PROF-001: activity vs expectation ────────────────────────────────────
    s_prof001 = (
        np.where((income > 0) & (ivr > 50), 0.20, np.where((income > 0) & (ivr > 20), 0.10, 0.0)) +
        np.where((vol > 100_000) & (tenure < 180), 0.15, 0.0)
    )

    # ── PROF-002: financial standing ─────────────────────────────────────────
    s_prof002 = (
        np.where((income > 0) & (outflow > 10_000) & (spi > 10), 0.35,
        np.where((income > 0) & (outflow > 10_000) & (spi > 5),  0.18, 0.0))
    )

    # ── PROF-005: living beyond means ────────────────────────────────────────
    s_prof005 = (
        np.where((income > 0) & (spi > 5), 0.20, np.where((income > 0) & (spi > 3), 0.10, 0.0)) +
        np.where((lux_sum > 5_000) & (spi > 2), 0.10, 0.0)
    )

    # ── PROF-008: transaction type/size atypical ──────────────────────────────
    s_prof008 = (
        np.where((amt_cv > 3.0) & (txn >= 20), 0.15, 0.0) +
        np.where((max_amt > 50_000) & ((income == 0) | (max_amt > income * 2)), 0.20, 0.0) +
        np.where((avg_amt > 10_000) & (is_ind == 1), 0.12, 0.0)
    )

    # ── PROF-010: sudden change ───────────────────────────────────────────────
    s_prof010 = (
        np.where(mv_cv > 3.0, 0.20, np.where(mv_cv > 2.0, 0.10, 0.0)) +
        np.where((mt_std > 40) & (mv_cv > 1.5), 0.12, 0.0) +
        np.where(amt_cv > 4.0, 0.10, 0.0)
    )

    s = (s_acct001 + s_acct002 + s_acct003 + s_acct004 + s_prod008
         + s_prof001 + s_prof002 + s_prof005 + s_prof008 + s_prof010)
    return pd.Series(s, index=df.index).clip(0, 1)


def rules_trade_shell(df: pd.DataFrame) -> pd.Series:
    """
    GATE-001:     Business account used as pure pass-through conduit
    PML-TBML-02:  Sector deviation — industry_risk_high + volume_to_sales
    PML-TBML-03:  Counterparty risk — many wire/EFT counterparties
    PML-TBML-04:  Volume spike — sudden large EFT credit inflow
    PML-TBML-08:  Round-sum invoices — ratio_round_1000 on business accounts
    PROF-004:     Business activity mismatch — volume vs declared sales + employees
    """
    eft_cr   = _col(df, "eft_credit_sum")
    eft_cnt  = _col(df, "count_eft")
    wire     = _col(df, "sum_wire")
    wire_cnt = _col(df, "count_wire")
    ftr      = _col(df, "flow_through_ratio")
    vol      = _col(df, "total_volume")
    net_flow = _col(df, "net_flow")
    is_biz   = _col(df, "is_business")
    rr1k     = _col(df, "ratio_round_1000")
    cr1k     = _col(df, "count_round_1000")
    emp      = _col(df, "employee_count")
    vsr      = _col(df, "volume_to_sales_ratio")
    sales    = _col(df, "sales")
    ind_risk = _col(df, "industry_risk_high")
    mv_cv    = _col(df, "monthly_txn_cv")
    inflow_d = _col(df, "inflow_per_day")

    # ── GATE-001: pass-through / gatekeeper ──────────────────────────────────
    s_gate001 = (
        np.where((is_biz == 1) & (ftr > 0.92) & (vol > 200_000), 0.35,
        np.where((is_biz == 1) & (ftr > 0.85) & (vol > 100_000), 0.20, 0.0)) +
        np.where((vol > 100_000) & (net_flow.abs() < vol * 0.02), 0.15, 0.0)
    )

    # ── PML-TBML-02: sector deviation ────────────────────────────────────────
    s_tbml02 = (
        np.where((ind_risk == 1) & (vsr > 5), 0.25, np.where((ind_risk == 1) & (vsr > 2), 0.12, 0.0)) +
        np.where((is_biz == 1) & (ind_risk == 1) & (wire_cnt > 5), 0.10, 0.0)
    )

    # ── PML-TBML-03: counterparty risk ───────────────────────────────────────
    s_tbml03 = (
        np.where((is_biz == 1) & (wire_cnt > 20), 0.25,
        np.where((is_biz == 1) & (wire_cnt > 10), 0.12, 0.0)) +
        np.where((is_biz == 1) & (eft_cnt > 50), 0.15, 0.0) +
        np.where(wire > 500_000, 0.20, 0.0)
    )

    # ── PML-TBML-04: EFT volume spike ────────────────────────────────────────
    s_tbml04 = (
        np.where(eft_cr > 500_000, 0.35, np.where(eft_cr > 100_000, 0.18, 0.0)) +
        np.where((mv_cv > 2.0) & (eft_cr > 50_000), 0.10, 0.0) +
        np.where((is_biz == 1) & (inflow_d > 5_000), 0.10, 0.0)
    )

    # ── PML-TBML-08: round-sum invoices ──────────────────────────────────────
    s_tbml08 = (
        np.where((is_biz == 1) & (rr1k > 0.60), 0.20,
        np.where((is_biz == 1) & (rr1k > 0.40), 0.10, 0.0)) +
        np.where((is_biz == 1) & (cr1k >= 10), 0.10, 0.0)
    )

    # ── PROF-004: business activity mismatch ─────────────────────────────────
    s_prof004 = (
        np.where((sales > 0) & (vol > 50_000) & (vsr > 20), 0.30,
        np.where((sales > 0) & (vol > 50_000) & (vsr > 10), 0.15, 0.0)) +
        np.where((is_biz == 1) & (emp <= 2) & (sales > 0) & (vsr > 20), 0.20, 0.0)
    )

    s = s_gate001 + s_tbml02 + s_tbml03 + s_tbml04 + s_tbml08 + s_prof004
    return pd.Series(s, index=df.index).clip(0, 1)


def rules_cross_border_geo(df: pd.DataFrame) -> pd.Series:
    """
    GEO-001:   Drug-producing/transit jurisdiction transactions
    GEO-002:   FATF blacklist transactions (highest severity)
    GEO-003:   Greylist / offshore / underground banking exposure
    GEO-004:   FATF non-cooperative — floor signal via has_any_high_risk_txn
    GEO-005:   Frequent overseas transfers — ratio, count, WU, card countries
    WIRE-008:  Wire volume mismatch vs income
    WIRE-010:  Multiple wire counterparties
    """
    fatf     = _col(df, "high_risk_fatf_txn_count")
    drug     = _col(df, "drug_country_txn_count")
    grey     = _col(df, "greylist_txn_count")
    offshore = _col(df, "offshore_center_txn_count")
    ug_bank  = _col(df, "underground_banking_country_count")
    any_hr   = _col(df, "has_any_high_risk_txn")
    intl_r   = _col(df, "international_ratio")
    intl_n   = _col(df, "international_txn_count")
    u_ctry   = _col(df, "unique_countries")
    wu_sum   = _col(df, "sum_westernunion")
    wu_cnt   = _col(df, "count_westernunion")
    card_c   = _col(df, "card_unique_countries")
    wire_sum = _col(df, "sum_wire")
    wire_n   = _col(df, "count_wire")
    income   = _col(df, "income")

    # ── GEO-001: drug jurisdictions ───────────────────────────────────────────
    s_geo001 = np.where(drug >= 5, 0.25, np.where(drug >= 1, 0.12, 0.0))

    # ── GEO-002: FATF blacklist (highest single rule weight) ─────────────────
    s_geo002 = np.where(fatf >= 3, 0.55, np.where(fatf >= 1, 0.40, 0.0))

    # ── GEO-003: greylist / offshore / underground banking ────────────────────
    s_geo003 = (
        np.where(grey >= 10, 0.18, np.where(grey >= 3, 0.10, 0.0)) +
        np.where(offshore >= 3, 0.18, np.where(offshore >= 1, 0.10, 0.0)) +
        np.where(ug_bank >= 2, 0.10, 0.0)
    )

    # ── GEO-004: any FATF risk floor ─────────────────────────────────────────
    s_geo004 = np.where(any_hr == 1, 0.10, 0.0)

    # ── GEO-005: frequent overseas transfers ──────────────────────────────────
    s_geo005 = (
        np.where((intl_r > 0.80) & (intl_n >= 10), 0.18,
        np.where((intl_r > 0.50) & (intl_n >= 5),  0.10, 0.0)) +
        np.where(u_ctry >= 5, 0.10, 0.0) +
        np.where(wu_sum > 10_000, 0.12, np.where(wu_sum > 1_000, 0.06, 0.0)) +
        np.where(wu_cnt >= 5, 0.08, 0.0) +
        np.where(card_c >= 4, 0.08, 0.0)
    )

    # ── WIRE-008: wire volume mismatch ────────────────────────────────────────
    s_wire008 = (
        np.where(wire_sum > 500_000, 0.30, np.where(wire_sum > 100_000, 0.15, 0.0)) +
        np.where((income > 0) & (wire_sum > income), 0.10, 0.0)
    )

    # ── WIRE-010: multiple wire counterparties ────────────────────────────────
    s_wire010 = np.where(wire_n >= 15, 0.18, np.where(wire_n >= 8, 0.10, 0.0))

    s = s_geo001 + s_geo002 + s_geo003 + s_geo004 + s_geo005 + s_wire008 + s_wire010
    return pd.Series(s, index=df.index).clip(0, 1)


def rules_human_trafficking(df: pd.DataFrame) -> pd.Series:
    """
    HT-SEX-01:  Rounded retail/gift card purchases
    HT-SEX-02:  High-value convenience store purchases
    HT-SEX-03:  Luxury spend vs income
    HT-SEX-04:  Parking charges — victim transit
    HT-SEX-05:  Food delivery — victim maintenance
    HT-SEX-06:  Digital/crypto/gambling spend
    HT-SEX-07:  Multi-city ABM + night cash withdrawals
    HT-SEX-08:  Accommodation card spend
    HT-SEX-10:  Airfare / travel to source countries
    HT-SEX-12:  After-hours adult MCC transactions (direct: 7297/7298)
    HT-SEX-13:  Disproportionate volume vs income
    HT-SEX-14:  Recurring EMT/EFT outflows (rental/venue payments)
    """
    retail    = _col(df, "card_retail_count")
    rr        = _col(df, "ratio_round_100")
    avg_amt   = _col(df, "avg_transaction_amount")
    lux_sum   = _col(df, "card_luxury_sum")
    lux_cnt   = _col(df, "card_luxury_count")
    income    = _col(df, "income")
    spi       = _col(df, "spending_to_income_ratio")
    parking   = _col(df, "card_parking_count")
    delivery  = _col(df, "card_delivery_count")
    digital   = _col(df, "card_digital_count")
    cities    = _col(df, "unique_cities")
    night_abm = _col(df, "night_abm_count")
    accom_cnt = _col(df, "card_accommodation_count")
    accom_sum = _col(df, "card_accommodation_sum")
    travel    = _col(df, "card_travel_sum")
    ht_src    = _col(df, "ht_source_country_txn_count")
    wu_sum    = _col(df, "sum_westernunion")
    adult_ah  = _col(df, "card_adult_afterhours_count")
    card_ah   = _col(df, "card_afterhours_count")
    night_r   = _col(df, "night_transaction_ratio")
    vol       = _col(df, "total_volume")
    emt_d     = _col(df, "emt_debit_count")
    eft_d     = _col(df, "eft_debit_count")
    txn       = _col(df, "transaction_count_total")

    # ── HT-SEX-01: rounded retail/gift card purchases ─────────────────────────
    s_01 = np.where((retail >= 10) & (rr > 0.50), 0.18,
           np.where((retail >= 5)  & (rr > 0.50), 0.10, 0.0))

    # ── HT-SEX-02: high-value convenience store ───────────────────────────────
    s_02 = np.where((retail >= 10) & (avg_amt > 200), 0.12, 0.0)

    # ── HT-SEX-03: luxury spend vs income ────────────────────────────────────
    s_03 = (
        np.where((lux_sum > 10_000) & ((income == 0) | (spi > 3)), 0.28,
        np.where((lux_sum >  5_000) & ((income == 0) | (spi > 2)), 0.15, 0.0)) +
        np.where(lux_cnt >= 5, 0.08, 0.0)
    )

    # ── HT-SEX-04: parking charges ────────────────────────────────────────────
    s_04 = np.where((parking >= 20) & (cities >= 3), 0.15,
           np.where(parking >= 10, 0.08, 0.0))

    # ── HT-SEX-05: food delivery ──────────────────────────────────────────────
    s_05 = np.where((delivery >= 20) & (accom_cnt >= 3), 0.15,
           np.where(delivery >= 10, 0.08, 0.0))

    # ── HT-SEX-06: digital / crypto / gambling ───────────────────────────────
    s_06 = np.where(digital >= 5, 0.15, np.where(digital >= 2, 0.08, 0.0))

    # ── HT-SEX-07: multi-city ABM + night cash ────────────────────────────────
    s_07 = np.where((cities >= 5) & (night_abm >= 5), 0.35,
           np.where((cities >= 3) & (night_abm >= 3), 0.22,
           np.where((cities >= 3) & (night_abm >= 1), 0.12, 0.0)))

    # ── HT-SEX-08: accommodation spend ───────────────────────────────────────
    s_08 = (
        np.where(accom_cnt >= 5, 0.18, np.where(accom_cnt >= 3, 0.12, 0.0)) +
        np.where((accom_cnt >= 3) & (cities >= 2), 0.0, 0.0) +  # cities captured in s_07
        np.where(accom_sum > 2_000, 0.08, 0.0)
    )

    # ── HT-SEX-10: travel / source countries ─────────────────────────────────
    s_10 = (
        np.where(ht_src >= 5, 0.25, np.where(ht_src >= 3, 0.15, 0.0)) +
        np.where(travel > 5_000, 0.22, np.where(travel > 2_000, 0.12, 0.0)) +
        np.where((wu_sum > 2_000) & (ht_src >= 1), 0.10, 0.0)
    )

    # ── HT-SEX-12: after-hours adult MCC ─────────────────────────────────────
    s_12 = np.where(adult_ah >= 3, 0.45,
           np.where(adult_ah >= 1, 0.28,
           np.where((card_ah >= 10) & (night_r > 0.40) & (txn >= 10), 0.12, 0.0)))

    # ── HT-SEX-13: total volume vs income ────────────────────────────────────
    s_13 = np.where((income > 0) & (vol > 50_000) & (spi > 10), 0.18,
           np.where((income > 0) & (vol > 200_000) & (spi > 5), 0.12, 0.0))

    # ── HT-SEX-14: recurring EMT/EFT (venue payments) ────────────────────────
    max_eft_emt = np.maximum(emt_d.values, eft_d.values)
    s_14 = np.where(max_eft_emt >= 8, 0.18, np.where(max_eft_emt >= 4, 0.10, 0.0))

    s = s_01 + s_02 + s_03 + s_04 + s_05 + s_06 + s_07 + s_08 + s_10 + s_12 + s_13 + s_14
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


def profile_clusters(df: pd.DataFrame, rule_cols: list) -> pd.DataFrame:
    """
    Profile each cluster by the mean RULE score of its members.

    Clustering is performed in rule score space (not IF space), so each
    cluster groups customers by which regulatory thresholds they co-trigger,
    independent of how statistically anomalous they look to the IF model.

    How it works:
      1. Compute the mean of each typology rule score within each cluster.
         This tells us which regulatory threshold patterns the typical
         cluster member fires — e.g. a cluster dominated by high
         rule_cross_border_geo means its members share international
         transaction patterns that trigger GEO-001..005 / WIRE-008/010.
      2. Normalise these means across clusters to identify relative dominance.
         A cluster where rule_structuring_layering averages 0.45 while
         others average 0.05 is a structuring-dominant regulatory cohort.
      3. The primary typology label is the highest normalised dimension,
         but only if it exceeds 0.25 — below this no typology dominates
         and the cluster is labelled General — Low Risk.
      4. Risk tier is based on the mean of ALL rule scores for that cluster.
         Mean is more representative than max; max is distorted by a handful
         of extreme rule violators within the cluster.

    Risk tiers:
      High   (1.0) — cluster mean rule risk > 67th percentile across clusters
      Medium (0.5) — cluster mean rule risk > 33rd percentile
      Low    (0.0) — cluster mean rule risk <= 33rd percentile
    """
    profile = df.groupby("cluster")[rule_cols].mean()
    p_norm  = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    # Map rule column names to display labels
    rule_label_map = {
        "rule_structuring_layering": TYPOLOGY_LABELS["if_score_structuring_layering"],
        "rule_behavioural_profile":  TYPOLOGY_LABELS["if_score_behavioural_profile"],
        "rule_trade_shell":          TYPOLOGY_LABELS["if_score_trade_shell"],
        "rule_cross_border_geo":     TYPOLOGY_LABELS["if_score_cross_border_geo"],
        "rule_human_trafficking":    TYPOLOGY_LABELS["if_score_human_trafficking"],
    }

    def assign_label(row):
        return "General — Low Risk" if row.max() < 0.25 \
               else rule_label_map.get(row.idxmax(), row.idxmax())

    primary   = p_norm.apply(assign_label, axis=1)
    mean_risk = profile.mean(axis=1)
    p33, p67  = mean_risk.quantile(0.33), mean_risk.quantile(0.67)
    risk_tier = mean_risk.apply(lambda x: 1.0 if x > p67 else (0.5 if x > p33 else 0.0))

    summary = pd.DataFrame({"primary_typology": primary, "risk_tier": risk_tier})

    k = len(profile)
    short = [c.replace("rule_", "")[:14] for c in rule_cols]
    print(f"\n  Cluster profiles (k={k}, mean rule scores):")
    print("  {:>3}  {:>7}  {}  {:>5}  Label".format(
        "Cls", "n", "  ".join(f"{h:>14}" for h in short), "Tier"))
    print("  " + "-" * 110)
    for c in profile.index:
        n     = (df["cluster"] == c).sum()
        vals  = "  ".join(f"{profile.loc[c, col]:>14.3f}" for col in rule_cols)
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
    parser.add_argument("--w_if",       type=float, default=0.60,
                        help="Weight for IF weighted score (default: 0.60)")
    parser.add_argument("--w_cluster",  type=float, default=0.10,
                        help="Weight for rule-space cluster score (default: 0.10)")
    parser.add_argument("--w_rule",     type=float, default=0.30,
                        help="Weight for rule-based flag score (default: 0.30)")
    args = parser.parse_args()

    base_dir  = Path(args.base_dir)
    feat_dir  = base_dir / "features"
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("HYBRID AML MODEL — IF + Rules + KMeans (rule-space clusters)")
    print("5 Typologies — AML-indicator-DB.xlsx")
    print("=" * 70)
    print(f"\nEnsemble weights:  IF={args.w_if}  |  Rules={args.w_rule}  |  Cluster={args.w_cluster}")
    print(f"KMeans input:      Rule scores (5D) — independent of IF layer")
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

    # ── Layer 3: KMeans clustering on RULE score space ────────────────────
    print("\n" + "-" * 70)
    print("Layer 3 — KMeans clustering on typology RULE scores...")
    print("  (Rationale: rule scores are grounded in FINTRAC/FinCEN regulatory")
    print("   thresholds and have low correlation with IF scores (r≈0.34),")
    print("   so clustering in rule space captures independent group-level")
    print("   evidence — customers who co-trigger the same regulatory patterns.)")
    print("-" * 70)

    # Rule score columns are already on df_features from Layer 2
    df_cluster = df_features[["customer_id"] + rule_cols].copy()
    df_cluster[rule_cols] = df_cluster[rule_cols].fillna(0)
    X_cluster = df_cluster[rule_cols].values

    k_range = range(args.k_min, args.k_max + 1)
    best_k  = args.k if args.k > 0 else select_k(X_cluster, k_range)

    print(f"\n  Training KMeans with k={best_k}...")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    kmeans.fit(X_cluster)
    df_cluster["cluster"] = kmeans.labels_

    cluster_summary = profile_clusters(df_cluster, rule_cols)
    df_cluster["cluster_primary_typology"] = df_cluster["cluster"].map(cluster_summary["primary_typology"])
    df_cluster["cluster_risk_tier"]        = df_cluster["cluster"].map(cluster_summary["risk_tier"])

    # Continuous cluster score: base tier × within-cluster distance from centroid.
    # Customers who are outliers within their rule-based cohort (far from centroid)
    # score higher than typical members — adds within-cluster differentiation.
    all_distances = kmeans.transform(X_cluster)
    own_dist      = all_distances[np.arange(len(df_cluster)), kmeans.labels_]
    cluster_score = np.zeros(len(df_cluster), dtype=float)
    for cid, row in cluster_summary.iterrows():
        mask  = kmeans.labels_ == cid
        base  = row["risk_tier"]
        d     = own_dist[mask]
        d_min, d_max = d.min(), d.max()
        norm  = (d - d_min) / (d_max - d_min + 1e-9)
        cluster_score[mask] = base * (0.5 + 0.5 * norm)
    df_cluster["cluster_score"] = cluster_score

    print(f"\n  Continuous cluster score (rule-space) — "
          f"mean={cluster_score.mean():.3f}  "
          f"p95={np.percentile(cluster_score, 95):.3f}  "
          f"max={cluster_score.max():.3f}")

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
                                 "cluster_risk_tier", "cluster_score",
                                 "cluster_primary_typology"]]
    merge_rules   = df_features[["customer_id", "rule_score_weighted",
                                  "rules_triggered", "primary_rule_typology"] + rule_cols]

    hybrid = (
        iso_results[["customer_id", "if_score_weighted", "if_score_max",
                      "if_score_mean", "primary_typology", "actual_label"] + avail_if_cols]
        .merge(merge_cluster, on="customer_id", how="inner")
        .merge(merge_rules,   on="customer_id", how="inner")
    )

    # cluster_score (continuous, rule-space) replaces the old flat cluster_risk_tier
    hybrid["final_hybrid_score"] = (
        hybrid["if_score_weighted"]   * args.w_if   +
        hybrid["rule_score_weighted"] * args.w_rule +
        hybrid["cluster_score"]       * args.w_cluster
    )

    # Normalise to [0, 1]
    mn = hybrid["final_hybrid_score"].min()
    mx = hybrid["final_hybrid_score"].max()
    hybrid["final_hybrid_score"] = (hybrid["final_hybrid_score"] - mn) / (mx - mn)

    hybrid = hybrid.sort_values("final_hybrid_score", ascending=False).reset_index(drop=True)
    hybrid["hybrid_risk_category"] = hybrid["final_hybrid_score"].apply(risk_category)

    print(f"\n  Component correlations with final score:")
    for col in ["if_score_weighted", "rule_score_weighted", "cluster_score", "cluster_risk_tier"]:
        if col in hybrid.columns:
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
              f"{'Rule':>6}  {'ClsScr':>7}  {'Tier':>4}  Typology")
        print(f"  {'-'*85}")
        for i, (idx, row) in enumerate(susp.iterrows(), 1):
            print(f"  {i:>3}  {idx+1:>7,}  {row['final_hybrid_score']:>7.3f}  "
                  f"{row['if_score_weighted']:>6.3f}  "
                  f"{row['rule_score_weighted']:>6.3f}  "
                  f"{row['cluster_score']:>7.3f}  "
                  f"{row['cluster_risk_tier']:>4.1f}  "
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
         "rules_triggered", "cluster_score", "cluster_risk_tier"]
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