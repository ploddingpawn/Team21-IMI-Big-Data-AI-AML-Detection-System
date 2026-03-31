"""
01_feature_engineering.py
==========================

Customer-level features engineered to support the five AML typologies
defined in AML-indicator-DB.xlsx:

  1. Structuring & Layering         (STRUCT-001..009, ATYPICAL-006..008, BEHAV-001, PROF-006/007)
  2. Behavioural & Profile Anomalies (PROF-001..010, ACCT-001..004, ATYPICAL-001..008)
  3. Trade-Based ML & Shell Entities (PML-TBML-01..08, PROF-004, GATE-001)
  4. Cross-Border & Geographic Risk  (GEO-001..005, WIRE-008/010)
  5. Human Trafficking               (HT-SEX-01..14)

Available channels: abm, card, cheque, eft, emt, westernunion, wire

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
# KYC risk tier codes
# ---------------------------------------------------------------------------

# High-risk occupation codes (kyc_occupation_codes.csv)
# FINTRAC guidance: gatekeepers, cash-intensive, cross-border roles
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

# High-risk industry codes (kyc_industry_codes.csv)
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
    8132,                                 # immigration services (HT exposure — HT-SEX-10)
    9111, 9114,                           # hotels (HT exposure — HT-SEX-07/08)
    9211, 9212,                           # restaurants (cash front)
    9641,                                 # professional sports
    9711, 9712,                           # barber/beauty (cash front)
    7499,                                 # other financial intermediaries
}

# HT-SEX-07/08: accommodation & hospitality sectors (multi-city trafficking pattern)
HT_ACCOMMODATION_INDUSTRY_CODES = {9111, 9114}

# Geographic risk country lists (GEO-001..005)
HIGH_RISK_FATF               = ["KP", "IR", "MM"]
GREYLIST                     = ["AL","BB","BF","KH","HR","JM","JO","ML","MZ",
                                 "NI","PK","PA","PH","SN","SY","TZ","TR","UG",
                                 "AE","VN","YE"]
OFFSHORE                     = ["KY","BM","VG","PA","LI","MC","AD","BS","LU"]
DRUG_COUNTRIES               = ["MX","CO","PE","BO","VE","GT","HN","EC","BR"]
UNDERGROUND_BANKING_COUNTRIES = ["CN","HK","PH","VN","IN","PK"]

# HT-SEX-10: high-risk source countries for international travel / recruitment
HT_SOURCE_COUNTRIES          = ["CN","KR","PH","VN","TH","MX","CO","BR","NG","GH"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clip01(series: pd.Series) -> pd.Series:
    return series.clip(0, 1)


def safe_quantile(series: pd.Series, q: float) -> float:
    v = series.dropna().quantile(q)
    return float(v) if (v is not None and not np.isnan(v) and v > 0) else 1.0


def load_transactions(data_dir: Path) -> pd.DataFrame:
    """Load all seven channel CSVs and return a unified transaction table.

    Card has extra columns (merchant_category, ecommerce_ind, country, province, city)
    which are preserved for card-specific feature engineering (HT-SEX MCC signals).
    All other channels pad missing columns with NaN so the union schema is consistent.
    """
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
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"   SKIP {channel}: {fname} not found")
            continue
        df = pd.read_csv(fpath)
        df["channel"] = channel
        if "amount_cad" not in df.columns and "amount" in df.columns:
            df = df.rename(columns={"amount": "amount_cad"})
        required = ["customer_id", "amount_cad", "debit_credit",
                    "transaction_datetime", "country", "province", "city",
                    "merchant_category", "ecommerce_ind"]
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
    df_occupation = df_occupation.copy()
    df_industry   = df_industry.copy()
    df_occupation["occupation_code"] = pd.to_numeric(df_occupation["occupation_code"], errors="coerce")
    df_industry["industry_code"]     = pd.to_numeric(df_industry["industry_code"],     errors="coerce")

    df_occupation["occupation_risk_high"] = df_occupation["occupation_code"].isin(HIGH_RISK_OCCUPATION_CODES).astype(int)
    df_industry["industry_risk_high"]     = df_industry["industry_code"].isin(HIGH_RISK_INDUSTRY_CODES).astype(int)
    df_industry["ht_accommodation_sector"]= df_industry["industry_code"].isin(HT_ACCOMMODATION_INDUSTRY_CODES).astype(int)

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
        df_industry[["industry_code", "industry_risk_high", "ht_accommodation_sector"]],
        on="industry_code", how="left"
    )
    df_kyc_ind["occupation_risk_high"]    = df_kyc_ind["occupation_risk_high"].fillna(0)
    df_kyc_bus["industry_risk_high"]      = df_kyc_bus["industry_risk_high"].fillna(0)
    df_kyc_bus["ht_accommodation_sector"] = df_kyc_bus["ht_accommodation_sector"].fillna(0)
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
    f[["transaction_count_debit", "total_outflow",
       "transaction_count_credit", "total_inflow"]] = f[[
        "transaction_count_debit", "total_outflow",
        "transaction_count_credit", "total_inflow"]].fillna(0)

    f["net_flow"]             = f["total_inflow"] - f["total_outflow"]
    f["inflow_outflow_ratio"] = f["total_inflow"] / (f["total_outflow"] + 1)
    f["std_transaction_amount"] = f["std_transaction_amount"].fillna(0)  # single-txn customers have no std
    f["amount_cv"]            = f["std_transaction_amount"] / (f["avg_transaction_amount"] + 1)

    f["first_transaction_date"] = pd.to_datetime(f["first_transaction_date"])
    f["last_transaction_date"]  = pd.to_datetime(f["last_transaction_date"])
    f["time_span_days"]         = (f["last_transaction_date"] - f["first_transaction_date"]).dt.days + 1

    f["transactions_per_day"]        = f["transaction_count_total"] / f["time_span_days"]
    f["transactions_per_active_day"] = f["transaction_count_total"] / f["active_days"].clip(lower=1)
    f["volume_per_day"]   = f["total_volume"]  / f["time_span_days"]
    f["inflow_per_day"]   = f["total_inflow"]  / f["time_span_days"]
    f["outflow_per_day"]  = f["total_outflow"] / f["time_span_days"]

    # ATYPICAL-007/008: flow-through ratio — funds enter and immediately leave
    f["flow_through_ratio"] = (
        np.minimum(f["total_inflow"], f["total_outflow"]) / (f["total_volume"] / 2 + 1)
    )

    print(f"   Volume & velocity: {f.shape[1]-1} features")

    # ── Structuring & Layering detection ──────────────────────────────────
    # STRUCT-001: Multiple cash deposits below $10K (ABM only — cash_indicator available)
    # STRUCT-003: Transactions $9,000-$9,999 (near-threshold avoidance)
    # STRUCT-006: Multiple transactions below reporting threshold in short period
    # STRUCT-009: Round sum amounts (PROF-007 also flags this)
    # BEHAV-001:  Multi-location activity (location hopping)
    near_10k  = txn[(txn["amount_cad"] >= 9000) & (txn["amount_cad"] < 10000)].groupby("customer_id").size().reset_index(name="count_near_10k")
    very_near = txn[(txn["amount_cad"] >= 9800) & (txn["amount_cad"] < 10000)].groupby("customer_id").size().reset_index(name="count_very_near_10k")
    below_10k = txn[txn["amount_cad"] < 10000].groupby("customer_id").agg(
        count_below_10k=("amount_cad", "count"), sum_below_10k=("amount_cad", "sum")
    ).reset_index()
    round_100  = txn[txn["amount_cad"] % 100  == 0].groupby("customer_id").size().reset_index(name="count_round_100")
    round_1000 = txn[txn["amount_cad"] % 1000 == 0].groupby("customer_id").size().reset_index(name="count_round_1000")

    for tmp in [near_10k, very_near, below_10k, round_100, round_1000]:
        f = f.merge(tmp, on="customer_id", how="left")

    struct_cols = ["count_near_10k", "count_very_near_10k", "count_below_10k",
                   "sum_below_10k", "count_round_100", "count_round_1000"]
    f[struct_cols] = f[struct_cols].fillna(0)
    f["ratio_below_10k"]  = f["count_below_10k"]  / f["transaction_count_total"]
    f["ratio_round_100"]  = f["count_round_100"]   / f["transaction_count_total"]
    f["ratio_round_1000"] = f["count_round_1000"]  / f["transaction_count_total"]

    # ABM cash structuring (STRUCT-001 — cash_indicator field)
    if "cash_indicator" in txn.columns:
        cash_abm = txn[(txn["channel"] == "ABM") & (txn["cash_indicator"] == 1) &
                       (txn["amount_cad"] < 10000)].groupby("customer_id").size().reset_index(name="cash_struct_count")
        f = f.merge(cash_abm, on="customer_id", how="left")
        f["cash_struct_count"] = f["cash_struct_count"].fillna(0)
    else:
        f["cash_struct_count"] = 0

    # ── Channel features ───────────────────────────────────────────────────
    ch_agg = txn.groupby(["customer_id", "channel"]).agg(
        count=("amount_cad", "count"), sum=("amount_cad", "sum"), mean=("amount_cad", "mean")
    ).reset_index()
    for metric in ["count", "sum", "mean"]:
        pivot = ch_agg.pivot(index="customer_id", columns="channel", values=metric).reset_index()
        pivot.columns = ["customer_id"] + [f"{metric}_{c.lower()}" for c in pivot.columns[1:]]
        f = f.merge(pivot, on="customer_id", how="left")

    ch_cols = [c for c in f.columns if any(x in c for x in ["_abm", "_card", "_cheque", "_eft", "_emt", "_westernunion", "_wire"])]
    f[ch_cols] = f[ch_cols].fillna(0)

    ch_div = txn.groupby("customer_id")["channel"].nunique().reset_index(name="unique_channels")
    f = f.merge(ch_div, on="customer_id", how="left")
    f["channel_diversity"] = f["unique_channels"] / 7  # 7 available channels

    ch_share = txn.groupby(["customer_id", "channel"]).size().reset_index(name="n")
    ch_share["total"]    = ch_share.groupby("customer_id")["n"].transform("sum")
    ch_share["share_sq"] = (ch_share["n"] / ch_share["total"]) ** 2
    hhi = ch_share.groupby("customer_id")["share_sq"].sum().reset_index(name="channel_concentration_hhi")
    f = f.merge(hhi, on="customer_id", how="left")

    f["has_western_union"] = (f.get("count_westernunion", pd.Series(0, index=f.index)) > 0).astype(int)
    f["has_wire"]          = (f.get("count_wire",         pd.Series(0, index=f.index)) > 0).astype(int)
    f["has_eft"]           = (f.get("count_eft",          pd.Series(0, index=f.index)) > 0).astype(int)
    f["has_emt"]           = (f.get("count_emt",          pd.Series(0, index=f.index)) > 0).astype(int)

    # ── Card-specific features (MCC / ecommerce) ───────────────────────────
    # merchant_category contains numeric ISO 18245 MCC codes (e.g. 5411, 5814).
    # Codes mapped from standard MCC taxonomy — verified against data output.
    #
    # Seen in data (top codes): 5814=Fast Food, 5411=Grocery, 5812=Restaurants,
    # 5541=Service Stations, 5310=Discount Stores, 5912=Drug/Pharmacy,
    # 5542=Fuel Dispensers, 5499=Misc Food, 4816=Computer/Network Services,
    # 5921=Liquor Stores, 4121=Taxicabs, 5251=Hardware, 5331=Variety,
    # 5300=Wholesale, 5968=Subscriptions, 5816=Digital Games, 5815=Digital Media
    #
    # MCC groups by AML indicator:
    # HT-SEX-01/02: 5411 Grocery, 5912 Drug/Pharmacy, 5310 Discount, 5331 Variety,
    #               5300 Wholesale, 5499 Misc Food Stores
    # HT-SEX-03:    5094 Jewelry/Watches, 5944 Jewelry Stores, 5661 Shoe Stores (luxury),
    #               7011 Hotels (luxury stay), 5945 Hobby/Toy (luxury goods)
    # HT-SEX-04:    7523 Parking Lots/Garages, 4121 Taxicabs/Limousines
    # HT-SEX-05:    5814 Fast Food, 5812 Restaurants, 5499 Misc Food
    # HT-SEX-06:    4816 Computer Network/Info Services (crypto exchanges),
    #               7995 Gambling Transactions, 5816 Digital Games, 5968 Subscriptions
    # HT-SEX-08:    7011 Hotels/Motels, 7012 Timeshares
    # HT-SEX-10:    4511 Airlines/Air Carriers, 4722 Travel Agencies, 4723 Tour Operators,
    #               3000-3299 Airline MCCs (stored as range — handled separately)
    # HT-SEX-12:    7297 Massage Parlors, 7298 Health/Beauty Spas

    # MCC code sets (as strings to match cast column)
    MCC_RETAIL        = {"5411", "5912", "5310", "5331", "5300", "5499"}
    MCC_LUXURY        = {"5094", "5944", "5945", "5661", "5699"}
    MCC_PARKING       = {"7523", "4121"}   # parking lots + taxicabs (HT-SEX-04)
    MCC_FOOD_DELIVERY = {"5814", "5812", "5499"}
    MCC_DIGITAL       = {"4816", "7995", "5816", "5968", "5815"}
    MCC_ACCOMMODATION = {"7011", "7012"}
    MCC_TRAVEL        = {"4511", "4722", "4723"}
    MCC_ADULT_SPA     = {"7297", "7298"}

    card_txn = txn[txn["channel"] == "Card"].copy()
    if len(card_txn) > 0 and "merchant_category" in card_txn.columns:
        # Cast to string to handle both numeric and text MCC values
        card_txn["mcc"] = card_txn["merchant_category"].astype(str).str.strip().str.split(".").str[0]

        # Print top MCCs so matching can be verified
        print(f"\n   Card MCC sample (top 20):")
        for mcc, cnt in card_txn["mcc"].value_counts().head(20).items():
            print(f"      MCC {mcc}: {cnt:,}")

        # Also handle airline MCCs 3000–3299 (individual carrier codes)
        airline_range = {str(i) for i in range(3000, 3300)}
        MCC_TRAVEL_ALL = MCC_TRAVEL | airline_range

        def mcc_filter(df, codes):
            return df[df["mcc"].isin(codes)]

        # HT-SEX-01/02: Retail / convenience store (gift card proxy)
        retail = mcc_filter(card_txn, MCC_RETAIL).groupby("customer_id").agg(
            card_retail_count=("amount_cad", "count"),
            card_retail_sum=("amount_cad", "sum"),
        ).reset_index()
        f = f.merge(retail, on="customer_id", how="left")
        print(f"   HT-SEX-01/02 retail (MCC {MCC_RETAIL}): {len(retail):,} customers")

        # HT-SEX-03: Luxury merchant categories
        luxury = mcc_filter(card_txn, MCC_LUXURY).groupby("customer_id").agg(
            card_luxury_count=("amount_cad", "count"),
            card_luxury_sum=("amount_cad", "sum"),
        ).reset_index()
        f = f.merge(luxury, on="customer_id", how="left")
        print(f"   HT-SEX-03 luxury (MCC {MCC_LUXURY}): {len(luxury):,} customers")

        # HT-SEX-04: Parking / taxicabs (victim transit indicator)
        parking = mcc_filter(card_txn, MCC_PARKING).groupby("customer_id").agg(
            card_parking_count=("amount_cad", "count"),
        ).reset_index()
        f = f.merge(parking, on="customer_id", how="left")
        print(f"   HT-SEX-04 parking/taxi (MCC {MCC_PARKING}): {len(parking):,} customers")

        # HT-SEX-05: Food delivery / restaurants (victim maintenance)
        delivery = mcc_filter(card_txn, MCC_FOOD_DELIVERY).groupby("customer_id").agg(
            card_delivery_count=("amount_cad", "count"),
        ).reset_index()
        f = f.merge(delivery, on="customer_id", how="left")
        print(f"   HT-SEX-05 food/delivery (MCC {MCC_FOOD_DELIVERY}): {len(delivery):,} customers")

        # HT-SEX-06: Digital / crypto / gambling
        digital = mcc_filter(card_txn, MCC_DIGITAL).groupby("customer_id").agg(
            card_digital_count=("amount_cad", "count"),
            card_digital_sum=("amount_cad", "sum"),
        ).reset_index()
        f = f.merge(digital, on="customer_id", how="left")
        print(f"   HT-SEX-06 digital/crypto (MCC {MCC_DIGITAL}): {len(digital):,} customers")

        # HT-SEX-08: Accommodation (non-residential city spend)
        accommodation = mcc_filter(card_txn, MCC_ACCOMMODATION).groupby("customer_id").agg(
            card_accommodation_count=("amount_cad", "count"),
            card_accommodation_sum=("amount_cad", "sum"),
        ).reset_index()
        f = f.merge(accommodation, on="customer_id", how="left")
        print(f"   HT-SEX-08 accommodation (MCC {MCC_ACCOMMODATION}): {len(accommodation):,} customers")

        # HT-SEX-10: Airfare / travel agencies
        travel = mcc_filter(card_txn, MCC_TRAVEL_ALL).groupby("customer_id").agg(
            card_travel_count=("amount_cad", "count"),
            card_travel_sum=("amount_cad", "sum"),
        ).reset_index()
        f = f.merge(travel, on="customer_id", how="left")
        print(f"   HT-SEX-10 travel/airline (MCC 4511/4722/4723/3000-3299): {len(travel):,} customers")

        # HT-SEX-12: After-hours adult MCC transactions (10 PM – 6 AM)
        adult_txn = mcc_filter(card_txn, MCC_ADULT_SPA)
        after_hours_adult = adult_txn[
            (adult_txn["hour"] >= 22) | (adult_txn["hour"] <= 5)
        ].groupby("customer_id").size().reset_index(name="card_adult_afterhours_count")
        f = f.merge(after_hours_adult, on="customer_id", how="left")

        # All after-hours card transactions (any MCC, 10 PM – 6 AM)
        card_afterhours = card_txn[
            (card_txn["hour"] >= 22) | (card_txn["hour"] <= 5)
        ].groupby("customer_id").size().reset_index(name="card_afterhours_count")
        f = f.merge(card_afterhours, on="customer_id", how="left")
        print(f"   HT-SEX-12 adult MCC after-hours (MCC {MCC_ADULT_SPA}): {len(after_hours_adult):,} customers")

        # Ecommerce ratio (ACCT-003 / HT-SEX-06 digital channel use)
        if "ecommerce_ind" in card_txn.columns:
            ecomm = card_txn[card_txn["ecommerce_ind"] == 1].groupby("customer_id").agg(
                card_ecommerce_count=("amount_cad", "count"),
                card_ecommerce_sum=("amount_cad", "sum"),
            ).reset_index()
            f = f.merge(ecomm, on="customer_id", how="left")

        # Card country diversity — HT-SEX-08 (non-residential city merchant locations)
        card_geo = card_txn.groupby("customer_id").agg(
            card_unique_cities=("city", "nunique"),
            card_unique_countries=("country", "nunique"),
        ).reset_index()
        f = f.merge(card_geo, on="customer_id", how="left")

    # Fill all card feature columns
    card_feature_cols = [c for c in f.columns if c.startswith("card_")]
    f[card_feature_cols] = f[card_feature_cols].fillna(0)

    # ── EFT/EMT-specific features ──────────────────────────────────────────
    # EFT (Electronic Fund Transfers): PML-TBML-03/04 (counterparty risk, volume spike)
    # EMT (e-Transfers): HT-SEX-09 (fund flow pattern), FENT-LAY-01 (pass-through)
    eft_txn = txn[txn["channel"] == "EFT"]
    emt_txn = txn[txn["channel"] == "EMT"]

    if len(eft_txn) > 0:
        # PML-TBML-04: Sudden large inflow of EFT — credits specifically
        eft_credits = eft_txn[eft_txn["debit_credit"] == "C"].groupby("customer_id").agg(
            eft_credit_count=("amount_cad", "count"),
            eft_credit_sum=("amount_cad", "sum"),
        ).reset_index()
        f = f.merge(eft_credits, on="customer_id", how="left")

        eft_debits = eft_txn[eft_txn["debit_credit"] == "D"].groupby("customer_id").agg(
            eft_debit_count=("amount_cad", "count"),
            eft_debit_sum=("amount_cad", "sum"),
        ).reset_index()
        f = f.merge(eft_debits, on="customer_id", how="left")

    if len(emt_txn) > 0:
        # HT-SEX-09 / FENT-LAY-01: EMT pass-through pattern
        emt_credits = emt_txn[emt_txn["debit_credit"] == "C"].groupby("customer_id").agg(
            emt_credit_count=("amount_cad", "count"),
            emt_credit_sum=("amount_cad", "sum"),
        ).reset_index()
        f = f.merge(emt_credits, on="customer_id", how="left")

        emt_debits = emt_txn[emt_txn["debit_credit"] == "D"].groupby("customer_id").agg(
            emt_debit_count=("amount_cad", "count"),
            emt_debit_sum=("amount_cad", "sum"),
        ).reset_index()
        f = f.merge(emt_debits, on="customer_id", how="left")

    eft_emt_cols = [c for c in f.columns if c.startswith("eft_") or c.startswith("emt_")]
    f[eft_emt_cols] = f[eft_emt_cols].fillna(0)

    print(f"   After channel-specific features: {f.shape[1]-1} features")

    # ── Geographic risk ────────────────────────────────────────────────────
    # GEO-001: Drug jurisdictions; GEO-002: FATF high-risk; GEO-003/004: FATF non-cooperative
    # GEO-005: Frequent overseas transfers (proxied via WU use + international ABM)
    # WIRE-008/010: Wire volume mismatch; multiple senders/receivers (proxied via wire count)
    geo = txn.groupby("customer_id").agg(
        unique_countries=("country", "nunique"),
        unique_provinces=("province", "nunique"),
        unique_cities=("city",    "nunique"),
    ).reset_index()
    f = f.merge(geo, on="customer_id", how="left")
    f["avg_txn_per_city"]     = f["transaction_count_total"] / (f["unique_cities"] + 0.1)
    f["avg_txn_per_province"] = f["transaction_count_total"] / (f["unique_provinces"] + 0.1)

    intl = txn[txn["country"] != "CA"].groupby("customer_id").agg(
        international_txn_count=("amount_cad", "count"),
        international_txn_sum=("amount_cad", "sum"),
    ).reset_index()
    f = f.merge(intl, on="customer_id", how="left")
    f["international_txn_count"] = f["international_txn_count"].fillna(0)
    f["international_txn_sum"]   = f["international_txn_sum"].fillna(0)
    f["international_ratio"]     = f["international_txn_count"] / f["transaction_count_total"]

    geo_risk_sets = [
        ("drug_country",        DRUG_COUNTRIES,                "drug_country_txn_count"),
        ("high_risk_fatf",      HIGH_RISK_FATF,                "high_risk_fatf_txn_count"),
        ("greylist",            GREYLIST,                      "greylist_txn_count"),
        ("offshore_center",     OFFSHORE,                      "offshore_center_txn_count"),
        ("underground_banking", UNDERGROUND_BANKING_COUNTRIES, "underground_banking_country_count"),
        ("ht_source_country",   HT_SOURCE_COUNTRIES,           "ht_source_country_txn_count"),  # HT-SEX-10
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
    # HT-SEX-12: Merchant POS after business hours (10 PM – 6 AM) — proxied via all-channel night txns
    # ATYPICAL-006: Same time of day pattern — proxied via night concentration
    night   = txn[(txn["hour"] >= 22) | (txn["hour"] <= 5)].groupby("customer_id").size().reset_index(name="night_transaction_count")
    evening = txn[(txn["hour"] >= 18) & (txn["hour"] <= 21)].groupby("customer_id").size().reset_index(name="evening_transaction_count")
    weekend = txn[txn["day_of_week"].isin([5, 6])].groupby("customer_id").size().reset_index(name="weekend_transaction_count")

    f = f.merge(night,   on="customer_id", how="left")
    f = f.merge(evening, on="customer_id", how="left")
    f = f.merge(weekend, on="customer_id", how="left")
    f["night_transaction_count"]   = f["night_transaction_count"].fillna(0)
    f["evening_transaction_count"] = f["evening_transaction_count"].fillna(0)
    f["weekend_transaction_count"] = f["weekend_transaction_count"].fillna(0)
    f["night_transaction_ratio"]   = f["night_transaction_count"]   / f["transaction_count_total"]
    f["evening_transaction_ratio"] = f["evening_transaction_count"] / f["transaction_count_total"]
    f["weekend_ratio"]             = f["weekend_transaction_count"] / f["transaction_count_total"]

    # Late-night + multi-city ABM combination (HT-SEX-07: accommodation + cash pattern)
    night_abm = txn[(txn["channel"] == "ABM") & ((txn["hour"] >= 22) | (txn["hour"] <= 5))].groupby("customer_id").size().reset_index(name="night_abm_count")
    f = f.merge(night_abm, on="customer_id", how="left")
    f["night_abm_count"] = f["night_abm_count"].fillna(0)

    monthly = txn.groupby(["customer_id", "year_month"]).agg(
        monthly_txn_count=("amount_cad", "count"),
        monthly_volume=("amount_cad", "sum"),
    ).reset_index()
    monthly_stats = monthly.groupby("customer_id").agg(
        monthly_txn_count_std=("monthly_txn_count",  "std"),
        monthly_txn_count_mean=("monthly_txn_count", "mean"),
        monthly_volume_std=("monthly_volume",  "std"),
        monthly_volume_mean=("monthly_volume", "mean"),
    ).reset_index()
    monthly_stats["monthly_txn_cv"]    = monthly_stats["monthly_txn_count_std"] / (monthly_stats["monthly_txn_count_mean"] + 1)
    monthly_stats["monthly_volume_cv"] = monthly_stats["monthly_volume_std"]    / (monthly_stats["monthly_volume_mean"] + 1)
    f = f.merge(monthly_stats, on="customer_id", how="left")
    for col in ["monthly_txn_count_std", "monthly_volume_std", "monthly_txn_cv", "monthly_volume_cv"]:
        f[col] = f[col].fillna(0)

    print(f"   After temporal: {f.shape[1]-1} features")

    # ── KYC features ──────────────────────────────────────────────────────
    f = f.merge(
        df_kyc_ind[["customer_id", "income", "occupation_code", "birth_date", "onboard_date", "occupation_risk_high"]],
        on="customer_id", how="left"
    )
    f = f.merge(
        df_kyc_bus[["customer_id", "sales", "industry_code", "employee_count", "established_date",
                    "industry_risk_high", "ht_accommodation_sector"]],
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

    # Segment-aware ratios
    for col in ["total_outflow", "total_inflow", "total_volume",
                "transactions_per_active_day", "flow_through_ratio"]:
        for seg, mask in [("ind", f["is_individual"] == 1), ("bus", f["is_business"] == 1)]:
            seg_median = f.loc[mask, col].median() if mask.sum() > 0 else 1
            seg_p90    = f.loc[mask, col].quantile(0.90) if mask.sum() > 0 else 1
            f[f"{col}_vs_{seg}_median"] = f[col] / (seg_median + 1)
            f[f"{col}_above_{seg}_p90"] = (f[col] > seg_p90).astype(int)

    f["spending_to_income_ratio"] = (f["total_outflow"] / (f["income"] + 1)).fillna(0)
    f["volume_to_sales_ratio"]    = ((f["total_inflow"] + f["total_outflow"]) / (f["sales"] + 1)).fillna(0)
    f["income_vol_ratio"]         = (f["total_volume"] / (f["income"] + 1)).fillna(0)
    f["occupation_risk_high"]     = f["occupation_risk_high"].fillna(0)
    f["industry_risk_high"]       = f["industry_risk_high"].fillna(0)
    f["ht_accommodation_sector"]  = f["ht_accommodation_sector"].fillna(0)

    # ── Typology composite scores ─────────────────────────────────────────
    # Defensive NaN check — any NaN in these inputs will silently poison composites
    composite_inputs = [
        "spending_to_income_ratio", "volume_to_sales_ratio", "income_vol_ratio",
        "monthly_volume_cv", "amount_cv", "flow_through_ratio",
        "ratio_round_100", "transactions_per_active_day",
    ]
    nan_cols = {c: int(f[c].isna().sum()) for c in composite_inputs if c in f.columns and f[c].isna().any()}
    if nan_cols:
        print(f"\n   WARNING: NaN values in composite inputs (will be filled with 0): {nan_cols}")
        for c in nan_cols:
            f[c] = f[c].fillna(0)

    def pct_norm(col):
        p99 = safe_quantile(f[col], 0.99)
        return (f[col].clip(upper=p99) / p99).clip(0, 1)

    z0 = pd.Series(0.0, index=f.index)

    wire_norm    = pct_norm("sum_wire")         if "sum_wire"         in f.columns else z0
    abm_norm     = pct_norm("count_abm")        if "count_abm"        in f.columns else z0
    wu_norm      = pct_norm("sum_westernunion")  if "sum_westernunion" in f.columns else z0
    cheque_norm  = pct_norm("count_cheque")     if "count_cheque"     in f.columns else z0
    eft_norm     = pct_norm("count_eft")        if "count_eft"        in f.columns else z0
    emt_norm     = pct_norm("count_emt")        if "count_emt"        in f.columns else z0
    card_norm    = pct_norm("count_card")       if "count_card"       in f.columns else z0

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

    near_10k_norm = pct_norm("count_near_10k") if "count_near_10k" in f.columns else z0

    # ── Typology 1: Structuring & Layering ────────────────────────────────
    # Library indicators: STRUCT-001, STRUCT-006, ATYPICAL-007, ATYPICAL-008,
    #                     BEHAV-001, PROF-006, PROF-007
    f["structuring_layering_risk"] = clip01(
        near_10k_norm                                   * 0.25 +  # STRUCT-003: threshold avoidance
        f["flow_through_ratio"].clip(0, 1)              * 0.25 +  # ATYPICAL-007/008: pass-through / same-day
        f["ratio_round_100"].clip(0, 1)                 * 0.15 +  # PROF-007: round sums
        f["unique_cities"].clip(upper=20) / 20          * 0.10 +  # BEHAV-001: location hopping
        (f["transactions_per_active_day"].clip(
            upper=safe_quantile(f["transactions_per_active_day"], 0.99)) /
         safe_quantile(f["transactions_per_active_day"], 0.99)) * 0.10 +  # STRUCT-006: velocity
        (eft_norm + emt_norm).clip(0, 1)                * 0.10 +  # ATYPICAL-007: rapid EFT/EMT pass-through
        wire_norm                                       * 0.05    # PROF-006: rapid large wire movement
    )

    # ── Typology 2: Behavioural & Profile Anomalies ───────────────────────
    # Library indicators: ACCT-001..004 (dormant, periodic, abrupt change),
    #                     PROF-001..010 (activity vs expectation, financial standing,
    #                     living beyond means, sudden change),
    #                     ATYPICAL-006..008 (temporal pattern, quick in-out, same-day)
    monthly_vol_cv_norm = (
        f["monthly_volume_cv"].clip(upper=safe_quantile(f["monthly_volume_cv"], 0.99)) /
        safe_quantile(f["monthly_volume_cv"], 0.99)
    ).clip(0, 1)

    amount_cv_norm = (
        f["amount_cv"].clip(upper=safe_quantile(f["amount_cv"], 0.99)) /
        safe_quantile(f["amount_cv"], 0.99)
    ).clip(0, 1)

    f["behavioural_profile_risk"] = clip01(
        spending_income_norm                * 0.30 +  # PROF-002/005: financial standing / living beyond means
        income_vol_norm                     * 0.20 +  # PROF-001: activity vs expectation
        monthly_vol_cv_norm                 * 0.20 +  # PROF-010 / ACCT-004: sudden change
        amount_cv_norm                      * 0.15 +  # PROF-008 / ATYPICAL: transaction size atypical
        vol_to_sales_norm * f["is_business"]* 0.10 +  # PROF-004: business activity mismatch
        f["channel_diversity"]              * 0.05    # ACCT-002 / ATYPICAL: channel mixing
    )

    # ── Typology 3: Trade-Based ML & Shell Entities ───────────────────────
    # Library indicators: PML-TBML-02 (sector deviation), PML-TBML-03 (counterparty risk),
    #                     PML-TBML-04 (volume spike via EFT), PML-TBML-08 (round sums),
    #                     PROF-004 (business activity), GATE-001 (gatekeeper / pass-through)
    f["trade_shell_risk"] = clip01(
        f["industry_risk_high"].astype(float)           * 0.25 +  # PML-TBML-02: sector deviation
        vol_to_sales_norm                               * 0.25 +  # PROF-004 / PML-TBML-05: profile mismatch
        f["flow_through_ratio"].clip(0, 1)              * 0.20 +  # GATE-001: pass-through / gatekeeper
        eft_norm                                        * 0.15 +  # PML-TBML-04: large EFT volume spike
        f["ratio_round_1000"].clip(0, 1)                * 0.10 +  # PML-TBML-08: round sum invoices
        abm_norm                                        * 0.05    # PML-TBML-03: multi-location counterparty
    )

    # ── Typology 4: Cross-Border & Geographic Risk ────────────────────────
    # Library indicators: GEO-001 (drug jurisdictions), GEO-002 (ML/TF risk),
    #                     GEO-003 (locations of concern), GEO-004 (FATF non-cooperative),
    #                     GEO-005 (frequent overseas transfers), WIRE-008 (volume mismatch),
    #                     WIRE-010 (multiple senders/receivers)
    f["cross_border_geo_risk"] = clip01(
        (f["high_risk_fatf_txn_count"] > 0).astype(float) * 0.35 +  # GEO-002/004: FATF blacklist (hard flag)
        f["international_ratio"].clip(0, 1)               * 0.20 +  # GEO-005: frequent overseas transfers
        wu_norm                                            * 0.15 +  # GEO-005: WU as overseas transfer proxy
        (f["drug_country_txn_count"] > 0).astype(float)   * 0.10 +  # GEO-001: drug jurisdictions
        f["greylist_txn_count"].clip(
            upper=safe_quantile(f["greylist_txn_count"], 0.99)) /
        (safe_quantile(f["greylist_txn_count"], 0.99) + 1) * 0.10 + # GEO-003: greylist exposure
        wire_norm                                           * 0.10   # WIRE-008/010: wire volume mismatch
    )

    # ── Typology 5: Human Trafficking ────────────────────────────────────
    # Library indicators: HT-SEX-01..14 (full set — card.csv available)
    # HT-SEX-01/02: Rounded grocery / convenience purchases (gift card proxy)
    # HT-SEX-03:    Luxury spend inconsistent with income/occupation
    # HT-SEX-04/05: Frequent parking + food delivery (victim maintenance)
    # HT-SEX-06:    Virtual currency / online gambling transfers
    # HT-SEX-07:    Multi-city ABM + accommodation booking pattern
    # HT-SEX-08:    Accommodation / travel card spend in non-residential cities
    # HT-SEX-10:    International airfare to high-risk source countries
    # HT-SEX-12:    After-hours card transactions (spa, massage, escort — 10 PM to 6 AM)
    # HT-SEX-13:    High-value transfers disproportionate to income
    # HT-SEX-14:    Monthly rental/maintenance payments via EMT/EFT
    multi_city_flag = (f["unique_cities"].clip(upper=20) / 20).clip(0, 1)
    ht_source_norm  = (f["ht_source_country_txn_count"].clip(
        upper=safe_quantile(f["ht_source_country_txn_count"], 0.99)) /
        (safe_quantile(f["ht_source_country_txn_count"], 0.99) + 1)).clip(0, 1)
    night_abm_norm  = (f["night_abm_count"].clip(
        upper=safe_quantile(f["night_abm_count"], 0.99)) /
        (safe_quantile(f["night_abm_count"], 0.99) + 1)).clip(0, 1)

    # Card MCC signals (direct from card.csv)
    luxury_norm        = pct_norm("card_luxury_sum")            if "card_luxury_sum"            in f.columns else z0
    accommodation_norm = pct_norm("card_accommodation_count")   if "card_accommodation_count"   in f.columns else z0
    adult_ah_norm      = pct_norm("card_adult_afterhours_count")if "card_adult_afterhours_count" in f.columns else z0
    parking_norm       = pct_norm("card_parking_count")         if "card_parking_count"         in f.columns else z0
    delivery_norm      = pct_norm("card_delivery_count")        if "card_delivery_count"        in f.columns else z0
    travel_norm        = pct_norm("card_travel_sum")            if "card_travel_sum"            in f.columns else z0
    emt_debit_norm     = pct_norm("emt_debit_count")            if "emt_debit_count"            in f.columns else z0
    eft_debit_norm     = pct_norm("eft_debit_count")            if "eft_debit_count"            in f.columns else z0

    f["human_trafficking_risk"] = clip01(
        adult_ah_norm                                  * 0.20 +  # HT-SEX-12: after-hours spa/massage/escort card txns
        f["night_transaction_ratio"].clip(0, 1)        * 0.10 +  # HT-SEX-12: general after-hours pattern
        multi_city_flag                                * 0.15 +  # HT-SEX-07/08: multi-city travel pattern
        luxury_norm                                    * 0.10 +  # HT-SEX-03: luxury spend vs income/occupation
        accommodation_norm                             * 0.10 +  # HT-SEX-08: accommodation in non-residential cities
        ht_source_norm                                 * 0.10 +  # HT-SEX-10: txns to high-risk source countries
        travel_norm                                    * 0.05 +  # HT-SEX-10: airfare to source countries
        (parking_norm + delivery_norm).clip(0, 1)      * 0.05 +  # HT-SEX-04/05: victim maintenance (parking + food)
        (emt_debit_norm + eft_debit_norm).clip(0, 1)   * 0.10 +  # HT-SEX-14: rental/maintenance payments
        night_abm_norm                                 * 0.05    # HT-SEX-07: night ABM across multiple cities
    )

    typology_cols = [
        "structuring_layering_risk",
        "behavioural_profile_risk",
        "trade_shell_risk",
        "cross_border_geo_risk",
        "human_trafficking_risk",
    ]
    f["overall_typology_max_risk"] = f[typology_cols].max(axis=1)
    f["typology_breadth"]          = (f[typology_cols] > 0.3).sum(axis=1)

    print(f"   After typology composites: {f.shape[1]-1} features")
    print(f"   Typologies (5): {typology_cols}")

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
    parser.add_argument("--base_dir", type=str, default="/content/gdrive/MyDrive/AML_Competition")
    args = parser.parse_args()

    base_dir   = Path(args.base_dir)
    data_dir   = base_dir / "data"
    output_dir = base_dir / "features"
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("AML FEATURE ENGINEERING — 5 Knowledge Library Typologies")
    print("=" * 70)

    df_kyc_ind    = pd.read_csv(data_dir / "kyc_individual.csv")
    df_kyc_bus    = pd.read_csv(data_dir / "kyc_smallbusiness.csv")
    df_occupation = pd.read_csv(data_dir / "kyc_occupation_codes.csv")
    df_industry   = pd.read_csv(data_dir / "kyc_industry_codes.csv")
    df_labels     = pd.read_csv(data_dir / "labels.csv")

    print("\nLoading transactions...")
    txn = load_transactions(data_dir)

    print("\nAttaching KYC risk tiers...")
    df_kyc_ind, df_kyc_bus = attach_kyc_risk_tiers(df_kyc_ind, df_kyc_bus, df_occupation, df_industry)

    df = build_customer_features(txn, df_kyc_ind, df_kyc_bus)
    df = df.merge(df_labels, on="customer_id", how="left")

    inf_cols = [c for c in df.columns if df[c].isin([np.inf, -np.inf]).any()]
    if inf_cols:
        for col in inf_cols:
            df[col] = df[col].replace([np.inf, -np.inf], df[col].median())

    dupes = df["customer_id"].duplicated().sum()
    if dupes:
        df = df.drop_duplicates(subset="customer_id", keep="first")

    out_main = output_dir / "customer_features_enhanced.csv"
    df.to_csv(out_main, index=False)

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)
    print(f"   Customers:  {len(df):,}")
    print(f"   Features:   {df.shape[1]-2}  (excl. customer_id, label)")
    print(f"   Labeled:    {df['label'].notna().sum():,}  |  Suspicious: {df['label'].sum():.0f}")
    print(f"\n   Typology composites:")
    for t in ["structuring_layering_risk", "behavioural_profile_risk",
              "trade_shell_risk", "cross_border_geo_risk", "human_trafficking_risk"]:
        print(f"      {t:<35}  mean={df[t].mean():.3f}  p95={df[t].quantile(0.95):.3f}")
    print(f"\n   Output: {out_main}")


if __name__ == "__main__":
    main()