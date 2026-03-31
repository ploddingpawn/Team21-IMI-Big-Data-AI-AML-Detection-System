"""
01_feature_engineering.py
==========================

Customer-level features:
    Volume & velocity, structuring detection, channel-specific aggregations,
    geographic risk, temporal patterns, KYC profile features,
    and seven FINTRAC/FinCEN typology composite scores (grounded in AML_Knowledge_Library.xlsx).

Usage:
    python 01_feature_engineering.py --base_dir /path/to/AML_Competition
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# High-risk occupation codes — derived directly from kyc_occupation_codes.csv
# Grounded in FINTRAC guidance on occupations associated with ML/TF risk.
#
# Selected codes and their titles:
#   11100  Financial auditors and accountants      (gatekeepers, cash handling)
#   11101  Financial and investment analysts        (investment structuring risk)
#   11102  Financial advisors                       (gatekeepers)
#   41101  Lawyers and Quebec notaries              (FINTRAC designated reporting entity)
#   63100  Insurance agents and brokers             (premium financing / placement risk)
#   63101  Real estate agents and salespersons      (FINTRAC designated reporting entity)
#   63210  Hairstylists and barbers                 (cash-intensive, common front business)
#   65100  Cashiers                                 (cash handling)
#   70010  Construction managers                    (construction-sector ML risk)
#   73300  Transport truck drivers                  (cross-border cash / drug trafficking)
#   75200  Taxi and limousine drivers and chauffeurs (cash-intensive)
# ---------------------------------------------------------------------------
HIGH_RISK_OCCUPATION_CODES = {
    11100, 11101, 11102,   # financial professionals (gatekeepers)
    41101,                 # lawyers and notaries (FINTRAC designated)
    63100,                 # insurance agents and brokers
    63101,                 # real estate agents (FINTRAC designated)
    63210,                 # hairstylists and barbers (cash-intensive front)
    65100,                 # cashiers
    70010,                 # construction managers
    73300,                 # transport truck drivers (cross-border)
    75200,                 # taxi and limousine (cash-intensive)
}

# ---------------------------------------------------------------------------
# High-risk industry codes — derived directly from kyc_industry_codes.csv
# Grounded in FINTRAC guidance on sectors with elevated ML/TF exposure.
#
# Selected codes and their titles:
#   4011   Single Family Housing          )
#   4013   Residential Renovation         ) real estate / construction —
#   4021   Manufacturing & Light Ind.     ) FINTRAC designated sector,
#   4022   Commercial Building            ) high cash / layering risk
#   4491   Land Developers                )
#   4581   Taxicab Industry               (cash-intensive)
#   5511   Automobiles, Wholesale         (used vehicle — structuring risk)
#   6311   Automobile (New) Dealers       )
#   6312   Automobile (Used) Dealers      ) FINTRAC high-risk dealers
#   6561   Jewellery Stores               (precious metals/stones — FINTRAC)
#   7214   Investment Companies           (layering / placement)
#   7215   Holding Companies              (shell company risk)
#   7292   Estate, Trust and Agency Funds (trust / nominee risk)
#   7421   Mortgage Brokers               (real estate financing)
#   7511   Operators of Residential Buildings  (real estate)
#   7512   Operators of Non-Residential Buildings
#   7599   Other Real Estate Operators
#   7611   Insurance and Real Estate Agencies
#   7731   Offices of Chartered and Certified Accountants  (gatekeeper)
#   7739   Other Accounting and Bookkeeping Services
#   7761   Offices of Lawyers and Notaries  (FINTRAC designated)
#   7791   Security and Investigation Services  (cash-intensive)
#   8132   Immigration Services           (human trafficking exposure)
#   9111   Hotels and Motor Hotels        (human trafficking exposure)
#   9114   Guest Houses and Tourist Homes (human trafficking exposure)
#   9211   Restaurants, Licensed          (cash-intensive front)
#   9212   Restaurants, Unlicensed        (cash-intensive front)
#   9641   Professional Sports Clubs      (CMLN / luxury goods)
#   9711   Barber Shops                   (cash-intensive front)
#   9712   Beauty Shops                   (cash-intensive front)
#   7499   Other Financial Intermediaries (unregulated financial services)
# ---------------------------------------------------------------------------
HIGH_RISK_INDUSTRY_CODES = {
    4011, 4013, 4021, 4022, 4491,        # real estate & construction
    4581,                                 # taxicab (cash-intensive)
    5511,                                 # automobiles wholesale
    6311, 6312,                           # auto dealers (new & used)
    6561,                                 # jewellery stores
    7214, 7215, 7292,                     # investment/holding/trust
    7421,                                 # mortgage brokers
    7511, 7512, 7599, 7611,               # real estate operators & agencies
    7731, 7739,                           # accountants / bookkeeping
    7761,                                 # lawyers and notaries
    7791,                                 # security services
    8132,                                 # immigration services
    9111, 9114,                           # hotels (trafficking exposure)
    9211, 9212,                           # restaurants (cash front)
    9641,                                 # professional sports
    9711, 9712,                           # barber/beauty (cash front)
    7499,                                 # other financial intermediaries
}

HIGH_RISK_FATF              = ["KP", "IR", "MM"]
GREYLIST                    = ["AL","BB","BF","KH","HR","JM","JO","ML","MZ",
                                "NI","PK","PA","PH","SN","SY","TZ","TR","UG",
                                "AE","VN","YE"]
OFFSHORE                    = ["KY","BM","VG","PA","LI","MC","AD","BS","LU"]
DRUG_COUNTRIES              = ["MX","CO","PE","BO","VE","GT","HN","EC","BR"]
UNDERGROUND_BANKING_COUNTRIES = ["CN","HK","PH","VN","IN","PK"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clip01(series: pd.Series) -> pd.Series:
    return series.clip(0, 1)


def safe_quantile(series: pd.Series, q: float) -> float:
    v = series.quantile(q)
    return v if v > 0 else 1.0


def load_transactions(data_dir: Path) -> pd.DataFrame:
    """Load all seven channel CSVs and return a unified transaction table."""
    channel_files = {
        "ABM":          "abm.csv",
        "Card":         "card.csv",
        "Cheque":       "cheque.csv",
        "EFT":          "eft.csv",
        "EMT":          "emt.csv",
        "WesternUnion": "westernunion.csv",
        "Wire":         "wire.csv",
    }
    frames = []
    for channel, fname in channel_files.items():
        df = pd.read_csv(data_dir / fname)
        df["channel"] = channel
        if "amount_cad" not in df.columns and "amount" in df.columns:
            df = df.rename(columns={"amount": "amount_cad"})
        required = ["customer_id", "amount_cad", "debit_credit",
                    "transaction_datetime", "country", "province", "city"]
        for col in required:
            if col not in df.columns:
                df[col] = np.nan
        frames.append(df)
        print(f"   {channel}: {len(df):,} rows")

    txn = pd.concat(frames, ignore_index=True)
    txn["transaction_datetime"] = pd.to_datetime(txn["transaction_datetime"], errors="coerce")
    txn["hour"]        = txn["transaction_datetime"].dt.hour
    txn["day_of_week"] = txn["transaction_datetime"].dt.dayofweek
    txn["year_month"]  = txn["transaction_datetime"].dt.to_period("M").astype(str)
    txn["date"]        = txn["transaction_datetime"].dt.date
    print(f"\n   Unified: {len(txn):,} transactions, {txn['customer_id'].nunique():,} customers")
    return txn


def attach_kyc_risk_tiers(df_kyc_ind, df_kyc_bus, df_occupation, df_industry):
    """
    Add occupation_risk_high and industry_risk_high flags to KYC tables.
    Matches by integer occupation_code / industry_code from the actual CSVs,
    not by string label — avoids silent mismatches from label variations.
    """
    df_occupation = df_occupation.copy()
    df_industry   = df_industry.copy()
    df_occupation["occupation_code"] = pd.to_numeric(df_occupation["occupation_code"], errors="coerce")
    df_industry["industry_code"]     = pd.to_numeric(df_industry["industry_code"],     errors="coerce")

    df_occupation["occupation_risk_high"] = df_occupation["occupation_code"].isin(HIGH_RISK_OCCUPATION_CODES).astype(int)
    df_industry["industry_risk_high"]     = df_industry["industry_code"].isin(HIGH_RISK_INDUSTRY_CODES).astype(int)

    flagged_occ = df_occupation[df_occupation["occupation_risk_high"] == 1]["occupation_title"].tolist()
    flagged_ind = df_industry[df_industry["industry_risk_high"] == 1]["industry"].tolist()
    print(f"   High-risk occupations flagged ({len(flagged_occ)}): {flagged_occ}")
    print(f"   High-risk industries flagged  ({len(flagged_ind)}): {flagged_ind}")

    df_kyc_ind = df_kyc_ind.copy()
    df_kyc_bus = df_kyc_bus.copy()
    df_kyc_ind["occupation_code"] = pd.to_numeric(df_kyc_ind["occupation_code"], errors="coerce")
    df_kyc_bus["industry_code"]   = pd.to_numeric(df_kyc_bus["industry_code"],   errors="coerce")

    df_kyc_ind = df_kyc_ind.merge(
        df_occupation[["occupation_code", "occupation_risk_high"]], on="occupation_code", how="left"
    )
    df_kyc_bus = df_kyc_bus.merge(
        df_industry[["industry_code", "industry_risk_high"]], on="industry_code", how="left"
    )
    df_kyc_ind["occupation_risk_high"] = df_kyc_ind["occupation_risk_high"].fillna(0)
    df_kyc_bus["industry_risk_high"]   = df_kyc_bus["industry_risk_high"].fillna(0)
    return df_kyc_ind, df_kyc_bus


# ---------------------------------------------------------------------------
# Customer-level features
# ---------------------------------------------------------------------------

def build_customer_features(txn: pd.DataFrame, df_kyc_ind, df_kyc_bus) -> pd.DataFrame:
    print("\nBuilding customer-level features...")

    debits  = txn[txn["debit_credit"] == "D"]
    credits = txn[txn["debit_credit"] == "C"]

    # ── Volume & velocity ──────────────────────────────────────────────────
    f = txn.groupby("customer_id").agg(
        transaction_count_total=("amount_cad", "count"),
        total_volume=("amount_cad", "sum"),
        avg_transaction_amount=("amount_cad", "mean"),
        median_transaction_amount=("amount_cad", "median"),
        std_transaction_amount=("amount_cad", "std"),
        min_transaction_amount=("amount_cad", "min"),
        max_transaction_amount=("amount_cad", "max"),
        first_transaction_date=("transaction_datetime", "min"),
        last_transaction_date=("transaction_datetime", "max"),
        active_days=("date", "nunique"),
    ).reset_index()

    debit_agg = debits.groupby("customer_id").agg(
        transaction_count_debit=("amount_cad", "count"),
        total_outflow=("amount_cad", "sum"),
    ).reset_index()
    credit_agg = credits.groupby("customer_id").agg(
        transaction_count_credit=("amount_cad", "count"),
        total_inflow=("amount_cad", "sum"),
    ).reset_index()

    f = f.merge(debit_agg,  on="customer_id", how="left")
    f = f.merge(credit_agg, on="customer_id", how="left")
    f[["transaction_count_debit","total_outflow",
       "transaction_count_credit","total_inflow"]] = f[[
        "transaction_count_debit","total_outflow",
        "transaction_count_credit","total_inflow"]].fillna(0)

    f["net_flow"]             = f["total_inflow"] - f["total_outflow"]
    f["inflow_outflow_ratio"] = f["total_inflow"] / (f["total_outflow"] + 1)
    f["amount_cv"]            = f["std_transaction_amount"] / (f["avg_transaction_amount"] + 1)

    f["first_transaction_date"] = pd.to_datetime(f["first_transaction_date"])
    f["last_transaction_date"]  = pd.to_datetime(f["last_transaction_date"])
    f["time_span_days"]         = (f["last_transaction_date"] - f["first_transaction_date"]).dt.days + 1

    f["transactions_per_day"]        = f["transaction_count_total"] / f["time_span_days"]
    f["transactions_per_active_day"] = f["transaction_count_total"] / f["active_days"].clip(lower=1)
    f["volume_per_day"]   = f["total_volume"]  / f["time_span_days"]
    f["inflow_per_day"]   = f["total_inflow"]  / f["time_span_days"]
    f["outflow_per_day"]  = f["total_outflow"] / f["time_span_days"]

    # flow_through_ratio: close to 1.0 = funds enter and immediately leave.
    # Primary FINTRAC layering indicator (FINTRAC-2023-OA002 / Project Athena).
    f["flow_through_ratio"] = (
        np.minimum(f["total_inflow"], f["total_outflow"]) / (f["total_volume"] / 2 + 1)
    )

    print(f"   Volume & velocity: {f.shape[1]-1} features")

    # ── Structuring detection ──────────────────────────────────────────────
    near_10k = txn[(txn["amount_cad"] >= 9000) & (txn["amount_cad"] < 10000)].groupby("customer_id").size().reset_index(name="count_near_10k")
    very_near = txn[(txn["amount_cad"] >= 9800) & (txn["amount_cad"] < 10000)].groupby("customer_id").size().reset_index(name="count_very_near_10k")
    below_10k = txn[txn["amount_cad"] < 10000].groupby("customer_id").agg(
        count_below_10k=("amount_cad","count"), sum_below_10k=("amount_cad","sum")
    ).reset_index()
    round_100  = txn[txn["amount_cad"] % 100  == 0].groupby("customer_id").size().reset_index(name="count_round_100")
    round_1000 = txn[txn["amount_cad"] % 1000 == 0].groupby("customer_id").size().reset_index(name="count_round_1000")

    for tmp in [near_10k, very_near, below_10k, round_100, round_1000]:
        f = f.merge(tmp, on="customer_id", how="left")

    struct_cols = ["count_near_10k","count_very_near_10k","count_below_10k",
                   "sum_below_10k","count_round_100","count_round_1000"]
    f[struct_cols] = f[struct_cols].fillna(0)
    f["ratio_below_10k"]  = f["count_below_10k"]  / f["transaction_count_total"]
    f["ratio_round_100"]  = f["count_round_100"]   / f["transaction_count_total"]
    f["ratio_round_1000"] = f["count_round_1000"]  / f["transaction_count_total"]

    # ── Channel features ───────────────────────────────────────────────────
    ch_agg = txn.groupby(["customer_id","channel"]).agg(
        count=("amount_cad","count"), sum=("amount_cad","sum"), mean=("amount_cad","mean")
    ).reset_index()
    for metric in ["count","sum","mean"]:
        pivot = ch_agg.pivot(index="customer_id", columns="channel", values=metric).reset_index()
        pivot.columns = ["customer_id"] + [f"{metric}_{c.lower()}" for c in pivot.columns[1:]]
        f = f.merge(pivot, on="customer_id", how="left")

    ch_cols = [c for c in f.columns if any(x in c for x in ["_abm","_card","_cheque","_eft","_emt","_westernunion","_wire"])]
    f[ch_cols] = f[ch_cols].fillna(0)

    ch_div = txn.groupby("customer_id")["channel"].nunique().reset_index(name="unique_channels")
    f = f.merge(ch_div, on="customer_id", how="left")
    f["channel_diversity"] = f["unique_channels"] / 7

    ch_share = txn.groupby(["customer_id","channel"]).size().reset_index(name="n")
    ch_share["total"]    = ch_share.groupby("customer_id")["n"].transform("sum")
    ch_share["share_sq"] = (ch_share["n"] / ch_share["total"]) ** 2
    hhi = ch_share.groupby("customer_id")["share_sq"].sum().reset_index(name="channel_concentration_hhi")
    f = f.merge(hhi, on="customer_id", how="left")

    f["has_western_union"] = (f.get("count_westernunion", pd.Series(0,index=f.index)) > 0).astype(int)
    f["has_wire"]          = (f.get("count_wire",         pd.Series(0,index=f.index)) > 0).astype(int)

    print(f"   After channels: {f.shape[1]-1} features")

    # ── Geographic risk ────────────────────────────────────────────────────
    geo = txn.groupby("customer_id").agg(
        unique_countries=("country","nunique"),
        unique_provinces=("province","nunique"),
        unique_cities=("city","nunique"),
    ).reset_index()
    f = f.merge(geo, on="customer_id", how="left")
    f["avg_txn_per_city"]     = f["transaction_count_total"] / (f["unique_cities"] + 0.1)
    f["avg_txn_per_province"] = f["transaction_count_total"] / (f["unique_provinces"] + 0.1)

    intl = txn[txn["country"] != "CA"].groupby("customer_id").agg(
        international_txn_count=("amount_cad","count"),
        international_txn_sum=("amount_cad","sum"),
    ).reset_index()
    f = f.merge(intl, on="customer_id", how="left")
    f["international_txn_count"] = f["international_txn_count"].fillna(0)
    f["international_txn_sum"]   = f["international_txn_sum"].fillna(0)
    f["international_ratio"]     = f["international_txn_count"] / f["transaction_count_total"]

    geo_risk_sets = [
        ("drug_country",        DRUG_COUNTRIES,               "drug_country_txn_count"),
        ("high_risk_fatf",      HIGH_RISK_FATF,               "high_risk_fatf_txn_count"),
        ("greylist",            GREYLIST,                     "greylist_txn_count"),
        ("offshore_center",     OFFSHORE,                     "offshore_center_txn_count"),
        ("underground_banking", UNDERGROUND_BANKING_COUNTRIES,"underground_banking_country_count"),
    ]
    for _, countries, col in geo_risk_sets:
        tmp = txn[txn["country"].isin(countries)].groupby("customer_id").size().reset_index(name=col)
        f = f.merge(tmp, on="customer_id", how="left")
        f[col] = f[col].fillna(0)

    f["has_any_high_risk_txn"] = (
        (f["drug_country_txn_count"] > 0) |
        (f["high_risk_fatf_txn_count"] > 0) |
        (f["offshore_center_txn_count"] > 0)
    ).astype(int)

    # ── Temporal patterns ──────────────────────────────────────────────────
    night   = txn[(txn["hour"] >= 0) & (txn["hour"] <= 5)].groupby("customer_id").size().reset_index(name="night_transaction_count")
    weekend = txn[txn["day_of_week"].isin([5,6])].groupby("customer_id").size().reset_index(name="weekend_transaction_count")
    f = f.merge(night, on="customer_id", how="left").merge(weekend, on="customer_id", how="left")
    f["night_transaction_count"]   = f["night_transaction_count"].fillna(0)
    f["weekend_transaction_count"] = f["weekend_transaction_count"].fillna(0)
    f["night_transaction_ratio"]   = f["night_transaction_count"]   / f["transaction_count_total"]
    f["weekend_ratio"]             = f["weekend_transaction_count"] / f["transaction_count_total"]

    monthly = txn.groupby(["customer_id","year_month"]).agg(
        monthly_txn_count=("amount_cad","count"),
        monthly_volume=("amount_cad","sum"),
    ).reset_index()
    monthly_stats = monthly.groupby("customer_id").agg(
        monthly_txn_count_std=("monthly_txn_count","std"),
        monthly_txn_count_mean=("monthly_txn_count","mean"),
        monthly_volume_std=("monthly_volume","std"),
        monthly_volume_mean=("monthly_volume","mean"),
    ).reset_index()
    monthly_stats["monthly_txn_cv"]    = monthly_stats["monthly_txn_count_std"] / (monthly_stats["monthly_txn_count_mean"] + 1)
    monthly_stats["monthly_volume_cv"] = monthly_stats["monthly_volume_std"]    / (monthly_stats["monthly_volume_mean"] + 1)
    f = f.merge(monthly_stats, on="customer_id", how="left")
    for col in ["monthly_txn_count_std","monthly_volume_std","monthly_txn_cv","monthly_volume_cv"]:
        f[col] = f[col].fillna(0)

    print(f"   After temporal: {f.shape[1]-1} features")

    # ── KYC features ──────────────────────────────────────────────────────
    f = f.merge(
        df_kyc_ind[["customer_id","income","occupation_code","birth_date","onboard_date","occupation_risk_high"]],
        on="customer_id", how="left"
    )
    f = f.merge(
        df_kyc_bus[["customer_id","sales","industry_code","employee_count","established_date","industry_risk_high"]],
        on="customer_id", how="left"
    )

    f["customer_type"] = "unknown"
    f.loc[f["income"].notna(), "customer_type"] = "individual"
    f.loc[f["sales"].notna(),  "customer_type"] = "business"
    f["is_business"]   = (f["customer_type"] == "business").astype(int)
    f["is_individual"] = (f["customer_type"] == "individual").astype(int)

    f["birth_date"]   = pd.to_datetime(f["birth_date"],   errors="coerce")
    f["onboard_date"] = pd.to_datetime(f["onboard_date"], errors="coerce")
    f["age"]                  = (pd.Timestamp.now() - f["birth_date"]).dt.days   / 365.25
    f["customer_tenure_days"] = (pd.Timestamp.now() - f["onboard_date"]).dt.days

    # Segment-aware ratios: computed within individual vs business groups separately.
    # Prevents high-volume businesses from appearing suspicious purely by volume.
    for col in ["total_outflow","total_inflow","total_volume",
                "transactions_per_active_day","flow_through_ratio"]:
        for seg, mask in [("ind", f["is_individual"]==1), ("bus", f["is_business"]==1)]:
            seg_median = f.loc[mask, col].median() if mask.sum() > 0 else 1
            seg_p90    = f.loc[mask, col].quantile(0.90) if mask.sum() > 0 else 1
            f[f"{col}_vs_{seg}_median"] = f[col] / (seg_median + 1)
            f[f"{col}_above_{seg}_p90"] = (f[col] > seg_p90).astype(int)

    f["spending_to_income_ratio"] = f["total_outflow"] / (f["income"] + 1)
    f["volume_to_sales_ratio"]    = (f["total_inflow"] + f["total_outflow"]) / (f["sales"] + 1)
    f["income_vol_ratio"]         = f["total_volume"] / (f["income"] + 1)
    f["occupation_risk_high"]     = f["occupation_risk_high"].fillna(0)
    f["industry_risk_high"]       = f["industry_risk_high"].fillna(0)

    # ── Typology composite scores ─────────────────────────────────────────
    # Each composite is built directly from AML_Knowledge_Library.xlsx indicators.
    # Indicator IDs cited in comments trace each feature back to the library.
    # Raw component features are retained separately for the Isolation Forest.

    def pct_norm(col):
        """Normalize a column by its 99th percentile, clipped to [0,1]."""
        p99 = safe_quantile(f[col], 0.99)
        return (f[col].clip(upper=p99) / p99).clip(0, 1)

    z0 = pd.Series(0.0, index=f.index)  # zero series shorthand

    emt_norm  = pct_norm("count_emt")        if "count_emt"        in f.columns else z0
    eft_norm  = pct_norm("count_eft")        if "count_eft"        in f.columns else z0
    wire_norm = pct_norm("sum_wire")         if "sum_wire"         in f.columns else z0
    abm_norm  = pct_norm("count_abm")        if "count_abm"        in f.columns else z0
    wu_norm   = pct_norm("sum_westernunion") if "sum_westernunion" in f.columns else z0
    cheque_norm = pct_norm("count_cheque")   if "count_cheque"     in f.columns else z0

    income_vol_norm = (
        f["income_vol_ratio"].clip(upper=safe_quantile(f["income_vol_ratio"], 0.99)) /
        safe_quantile(f["income_vol_ratio"], 0.99)
    ).clip(0, 1)

    vol_to_sales_norm = (
        f["volume_to_sales_ratio"].clip(upper=safe_quantile(f["volume_to_sales_ratio"], 0.99)) /
        safe_quantile(f["volume_to_sales_ratio"], 0.99)
    ).clip(0, 1)

    spending_income_norm = (
        f["spending_to_income_ratio"].clip(upper=safe_quantile(f["spending_to_income_ratio"], 0.99)) /
        safe_quantile(f["spending_to_income_ratio"], 0.99)
    ).clip(0, 1)

    # ── 1. Underground Banking — FINTRAC 2023-OA002 (Project Athena)
    # Indicators: UB-ATH-01 (sector velocity), UB-ATH-02 (fan-out),
    #             UB-ATH-03 (pass-through), UB-ATH-10/11/12 (cash-to-sector),
    #             UB-ATH-15 (occupation gap)
    ub_china_flag = (f["underground_banking_country_count"] > 0).astype(float)
    f["underground_banking_risk"] = clip01(
        f["flow_through_ratio"].clip(0, 1) * 0.30 +  # UB-ATH-03: pass-through
        wire_norm                           * 0.25 +  # UB-ATH-01: sector velocity via wire
        (eft_norm + emt_norm).clip(0, 1)   * 0.15 +  # UB-ATH-02: fan-out via EFT/EMT
        ub_china_flag                       * 0.15 +  # UB-ATH-03: CN/HK nexus
        f["occupation_risk_high"]           * 0.10 +  # UB-ATH-15: occupation gap
        abm_norm                            * 0.05    # UB-ATH-10: cash-to-sector
    )

    # ── 2. Chinese ML / Mule Networks — FinCEN FIN-2025-A003 (CMLN)
    # Indicators: CMLN-MULE-01 (unexplained wealth), CMLN-MULE-02 (student activity),
    #             CMLN-MULE-03 (fund churn), CMLN-MULE-04 (high churn),
    #             CMLN-TBML-05 (business scale)
    f["cmln_risk"] = clip01(
        income_vol_norm                     * 0.35 +  # CMLN-MULE-01: income vs inflows
        f["flow_through_ratio"].clip(0, 1)  * 0.25 +  # CMLN-MULE-03/04: fund churn
        wire_norm                           * 0.20 +  # CMLN-MULE-02: wire outflows
        (emt_norm + abm_norm).clip(0, 1)    * 0.10 +  # CMLN-MULE-04: high churn channels
        vol_to_sales_norm                   * 0.10    # CMLN-TBML-05: business scale mismatch
    )

    # ── 3. Human Trafficking — FINTRAC 2021-OA-HTS (Project Protect)
    # Indicators: HT-SEX-07/08 (geographic travel patterns),
    #             HT-SEX-09 (fund flow patterns), HT-SEX-12 (after-hours POS),
    #             HT-SEX-11 (storefront EFT), HT-SEX-14 (rental payments)
    f["human_trafficking_risk"] = clip01(
        f["night_transaction_ratio"].clip(0, 1) * 0.30 +  # HT-SEX-12: after-hours activity
        emt_norm                                 * 0.25 +  # HT-SEX-09: EMT fund flow pattern
        f["unique_cities"].clip(upper=20) / 20   * 0.20 +  # HT-SEX-07/08: multi-city travel
        eft_norm                                 * 0.15 +  # HT-SEX-11: storefront EFT
        f["flow_through_ratio"].clip(0, 1)       * 0.10    # HT-SEX-09: pass-through pattern
    )

    # ── 4. Professional ML / TBML — FINTRAC 18/19-SIDEL-025
    # Indicators: PML-TBML-01 (commodity risk), PML-TBML-02 (profile mismatch),
    #             PML-TBML-04 (volume spike), PML-TBML-05/06 (global/regional mirroring),
    #             PML-TBML-08 (round sums), PML-MSB-01 (rapid outflow),
    #             PML-MSB-07 (fan-in consolidation), PML-MSB-08 (multi-location)
    f["trade_based_ml_risk"] = clip01(
        f["industry_risk_high"].astype(float)   * 0.25 +  # PML-TBML-01: high-risk commodity sector
        vol_to_sales_norm                        * 0.25 +  # PML-TBML-02/05: profile mismatch
        f["flow_through_ratio"].clip(0, 1)       * 0.20 +  # PML-MSB-01/07: rapid outflow / fan-in
        (eft_norm + wire_norm).clip(0, 1)        * 0.15 +  # PML-TBML-04: EFT volume spike
        f["ratio_round_100"].clip(0, 1)          * 0.10 +  # PML-TBML-08: round sum invoices
        abm_norm                                 * 0.05    # PML-MSB-08: multi-location cash
    )

    # ── 5. Synthetic Opioid / Fentanyl — FINTRAC 2025-OA-001
    # Indicators: FENT-LAY-01 (pass-through EMT), FENT-LAY-04 (smurfing),
    #             FENT-LAY-05 (mixed channel), FENT-LAY-07 (youth mule),
    #             FENT-DIST-01 (corridor velocity), FENT-DIST-03 (cluster activity)
    f["fentanyl_risk"] = clip01(
        abm_norm                                 * 0.30 +  # FENT-DIST-01: corridor cash deposits
        f["ratio_round_100"].clip(0, 1)          * 0.25 +  # FENT-LAY-04: structuring / smurfing
        (emt_norm + eft_norm).clip(0, 1)         * 0.20 +  # FENT-LAY-01/05: EMT pass-through
        f["unique_cities"].clip(upper=20) / 20   * 0.15 +  # FENT-DIST-03: cluster activity
        wu_norm                                  * 0.10    # FENT-LAY-05: WU mixed channel
    )

    # ── 6. Structuring — FINTRAC General (STRUCT-001 through STRUCT-009)
    # Indicators: STRUCT-001 (cash structuring), STRUCT-003 (threshold avoidance),
    #             STRUCT-006 (short period multiple transactions),
    #             STRUCT-008 (multi-location), BEHAV-001 (location hopping)
    near_10k_norm = pct_norm("count_near_10k") if "count_near_10k" in f.columns else z0
    f["structuring_risk"] = clip01(
        near_10k_norm                            * 0.40 +  # STRUCT-003: threshold avoidance
        f["ratio_round_100"].clip(0, 1)          * 0.25 +  # STRUCT-009: threshold awareness
        f["unique_cities"].clip(upper=20) / 20   * 0.20 +  # STRUCT-008/BEHAV-001: multi-location
        f["transactions_per_active_day"].clip(upper=safe_quantile(
            f["transactions_per_active_day"], 0.99)) /
            safe_quantile(f["transactions_per_active_day"], 0.99) * 0.15  # STRUCT-006: velocity
    )

    # ── 7. Profile Mismatch — FINTRAC General (PROF-001 through PROF-010)
    # Indicators: PROF-001 (activity vs expectation), PROF-002 (financial standing),
    #             PROF-004 (business activity), PROF-006 (fund movement),
    #             PROF-010 (sudden change)
    f["profile_mismatch_risk"] = clip01(
        spending_income_norm                     * 0.40 +  # PROF-002: financial standing
        vol_to_sales_norm                        * 0.30 +  # PROF-004: business activity
        income_vol_norm                          * 0.20 +  # PROF-001: activity vs expectation
        f["monthly_volume_cv"].clip(upper=safe_quantile(
            f["monthly_volume_cv"], 0.99)) /
            safe_quantile(f["monthly_volume_cv"], 0.99) * 0.10  # PROF-010: sudden change
    )

    typology_cols = [
        "underground_banking_risk", "cmln_risk", "human_trafficking_risk",
        "trade_based_ml_risk", "fentanyl_risk", "structuring_risk",
        "profile_mismatch_risk",
    ]
    f["overall_typology_max_risk"] = f[typology_cols].max(axis=1)
    f["typology_breadth"]          = (f[typology_cols] > 0.3).sum(axis=1)

    print(f"   After typology composites: {f.shape[1]-1} features")
    print(f"   Typologies: {typology_cols}")

    # ── Final cleanup ──────────────────────────────────────────────────────
    f = f.replace([np.inf, -np.inf], np.nan)
    for col in f.columns:
        if col in ("customer_id", "customer_type"):
            continue
        if f[col].isnull().sum() > 0:
            f[col] = f[col].fillna(f[col].median() if f[col].dtype == float else 0)

    print(f"\nComplete — {len(f):,} customers, {f.shape[1]-1} features")
    return f



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AML Feature Engineering")
    parser.add_argument("--base_dir", type=str, default="/content/gdrive/MyDrive/AML_Competition",
                        help="Root directory of the competition data")
    args = parser.parse_args()

    base_dir   = Path(args.base_dir)
    data_dir   = base_dir
    output_dir = base_dir / "features"
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("AML FEATURE ENGINEERING")
    print("=" * 70)

    # Load KYC
    df_kyc_ind  = pd.read_csv(data_dir / "kyc_individual.csv")
    df_kyc_bus  = pd.read_csv(data_dir / "kyc_smallbusiness.csv")
    df_occupation = pd.read_csv(data_dir / "kyc_occupation_codes.csv")
    df_industry   = pd.read_csv(data_dir / "kyc_industry_codes.csv")
    df_labels     = pd.read_csv(data_dir / "labels.csv")

    print("\nLoading transactions...")
    txn = load_transactions(data_dir)

    print("\nAttaching KYC risk tiers...")
    df_kyc_ind, df_kyc_bus = attach_kyc_risk_tiers(df_kyc_ind, df_kyc_bus, df_occupation, df_industry)

    # Build customer features
    df = build_customer_features(txn, df_kyc_ind, df_kyc_bus)

    # Attach labels
    df = df.merge(df_labels, on="customer_id", how="left")

    # Quality checks
    inf_cols = [c for c in df.columns if df[c].isin([np.inf, -np.inf]).any()]
    if inf_cols:
        print(f"Fixing inf values in: {inf_cols}")
        for col in inf_cols:
            df[col] = df[col].replace([np.inf, -np.inf], df[col].median())

    dupes = df["customer_id"].duplicated().sum()
    if dupes:
        df = df.drop_duplicates(subset="customer_id", keep="first")
        print(f"Removed {dupes} duplicate customers")

    # Save
    out_main = output_dir / "customer_features_enhanced.csv"
    df.to_csv(out_main, index=False)

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)
    print(f"   Customers:  {len(df):,}")
    print(f"   Features:   {df.shape[1]-2}  (excl. customer_id, label)")
    print(f"   Labeled:    {df['label'].notna().sum():,}  |  Suspicious: {df['label'].sum():.0f}")
    print(f"\n   Output: {out_main}")


if __name__ == "__main__":
    main()