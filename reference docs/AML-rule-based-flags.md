# AML Rule-Based Flags — Reference Document

This document formalizes the deterministic rule-based scoring layer of the hybrid AML detection pipeline. Each rule corresponds directly to one or more indicators from the `AML-indicator-DB.csv` catalogue and is implemented as a vectorised scoring block in `03-hybrid-model.py`.

## Architecture Overview

The rule-based layer is **Layer 2** of the three-layer ensemble:

```
Layer 1 (60%) — Typology Isolation Forests
Layer 2 (30%) — Rule-Based Flags
Layer 3 (10%) — K-Means Cluster Risk Tier
```

Rules are organized into five typology functions, each returning a score in **[0.0, 1.0]** for every customer simultaneously (fully vectorised — no row-by-row iteration). Within each function, indicator sub-blocks produce raw contribution values that are summed then clamped to 1.0.

The five typology rule scores are weighted using the same typology weights as the Isolation Forest layer and combined into a single `rule_score_weighted` value per customer.

### Scoring Design Principles

- **Additive within a typology.** Multiple indicators firing for the same customer accumulate score, reflecting corroborating evidence. The cap at 1.0 prevents any single typology from dominating unchecked.
- **Tiered thresholds.** Each indicator has two or three tiers (low, medium, high severity), so that borderline cases receive a lower contribution rather than triggering a binary flag.
- **Guard conditions.** Income, sales, and volume guards (`income > 0`, `sales > 0`) prevent rules from misfiring on customers with missing or zero baseline values.
- **No double-counting across channels.** Where two channels capture the same economic signal (e.g., EMT vs EFT for rental payments), the maximum of the two is used rather than summing both.

---

## 1. Structuring & Layering

**Typology weight: 0.30** (highest of the five typologies)

This typology targets deliberate avoidance of regulatory reporting thresholds and the rapid, multi-channel movement of funds through accounts used as conduits.

---

### STRUCT-001 — Cash Structuring

**Indicator:** Multiple cash deposits below $10,000 to avoid Currency Transaction Report (CTR) filing requirements.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `count_below_10k` | ≥ 20 | +0.20 | High volume of sub-threshold transactions is a textbook structuring pattern |
| `count_below_10k` | ≥ 10 | +0.10 | Moderate volume — still significant when combined with other signals |
| `ratio_below_10k` | > 0.90 AND `count_below_10k` ≥ 10 | +0.10 | When *most* of a customer's transactions are sub-threshold, it suggests systematic behavior rather than coincidence |
| `cash_struct_count` | ≥ 10 | +0.15 | ABM-specific: cash deposits are the primary mechanism for STRUCT-001; this feature isolates physical currency |
| `cash_struct_count` | ≥ 5 | +0.07 | Softer signal — fewer ABM deposits but still anomalous |

**Note:** `count_below_10k` covers all channels while `cash_struct_count` is ABM-specific. Both contribute independently because the indicator explicitly targets *cash* deposits — high `cash_struct_count` on its own is more specific than a high general count.

---

### STRUCT-006 — Short Period Multiple Transactions

**Indicator:** Multiple transactions conducted below the reporting threshold within a short time period (burst velocity).

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `transactions_per_active_day` | > 20 | +0.20 | Very high burst rate on active days — not consistent with legitimate retail banking |
| `transactions_per_active_day` | > 10 | +0.10 | Elevated but less severe burst rate |
| `volume_per_day` | > $5,000/day | +0.10 | Sustained high daily throughput amplifies the velocity signal regardless of individual transaction count |
| `volume_per_day` | > $2,000/day | +0.05 | Moderate daily volume |

**Note:** `transactions_per_active_day` measures intensity on days the account is actually used, not across the full observation window. This prevents diluting the signal for customers who transact in short concentrated bursts.

---

### ATYPICAL-006 — Suspicious Temporal Pattern

**Indicator:** Suspicious pattern in *when* transactions occur (e.g., same time of day, suggesting automated/programmatic activity).

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `night_transaction_ratio` | > 0.60 AND `transaction_count_total` ≥ 20 | +0.12 | A majority of transactions occurring at night, consistently, is anomalous for most retail customers |
| `night_transaction_ratio` | > 0.40 | +0.06 | Elevated night activity — softer signal |
| `weekend_ratio` | > 0.70 AND `transaction_count_total` ≥ 20 | +0.10 | Weekend concentration can indicate scheduled/scripted activity or trafficking operations |

**Limitation:** Actual detection of "same time of day" programmatic behavior would require per-transaction hour data aggregated into a concentration measure (e.g., entropy of transaction hour). Due to feasibility constraints, `night_transaction_ratio` and `weekend_ratio` serve as temporal pattern proxies. The transaction count guard (`≥ 20`) is added to ensure the ratios are statistically meaningful.

---

### ATYPICAL-007 — Quick In-Out / Pass-Through

**Indicator:** Atypical transfers on an in-and-out basis — funds deposited and immediately withdrawn or transferred. Classic pass-through / layering pattern.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `flow_through_ratio` | > 0.95 AND `total_volume` > $100K | +0.40 | Near-total pass-through at high volume — the strongest available signal for a conduit account |
| `flow_through_ratio` | > 0.90 AND `total_volume` > $50K | +0.25 | Still very high pass-through at significant volume |
| `flow_through_ratio` | > 0.80 AND `total_volume` > $20K | +0.12 | Moderate pass-through — meaningful but could have legitimate explanations |
| `count_eft` + `count_emt` | > 20 AND `flow_through_ratio` > 0.70 | +0.10 | High EFT/EMT counts alongside pass-through behavior identifies the electronic channel mechanism |

**Definition of `flow_through_ratio`:** `min(total_inflow, total_outflow) / total_inflow`. A value near 1.0 means almost everything that arrives in the account is then sent out — the account retains no economic value. The volume guard prevents this from firing on genuinely low-activity accounts where the ratio may be artificially high.

> **Note:** ATYPICAL-008 (same-day turnover) is intentionally excluded from the rule layer. True same-day detection requires matching individual inflow and outflow timestamps within a 24-hour window, which was deemed out of scope due to feasibility concerns. ATYPICAL-007's `flow_through_ratio` captures the same economic signal at the aggregate level.

---

### BEHAV-001 — Location Hopping

**Indicator:** Client conducts transactions at different physical locations (ABM, card terminals) across multiple cities and provinces.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `unique_cities` | ≥ 5 | +0.20 | Transactions spread across 5+ cities strongly suggest deliberate geographic dispersal |
| `unique_cities` | ≥ 3 | +0.10 | Multi-city activity — notable but could reflect legitimate travel |
| `unique_provinces` | ≥ 3 | +0.10 | Cross-provincial adds to the geographic risk beyond the city count |
| `unique_provinces` | ≥ 2 | +0.05 | Two provinces — softer signal |

**Note:** City and province counts should be additive because they capture different scales of geographic dispersal. A customer in 3 cities within one province is different from a customer in 3 cities across 3 provinces.

---

### PROF-006 — Large/Rapid Fund Movement

**Indicator:** Large and/or rapid movement of funds not commensurate with the client's financial profile.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `total_volume` | > $500K AND `income` > 0 AND `spending_to_income_ratio` > 5 | +0.30 | Very large volume that cannot be explained by declared income — strong mismatch |
| `total_volume` | > $200K AND `income` > 0 AND `spending_to_income_ratio` > 3 | +0.18 | Significant volume with moderate income mismatch |
| `volume_per_day` | > $10,000/day | +0.15 | Rapid sustained throughput regardless of total volume |
| `volume_per_day` | > $3,000/day | +0.08 | Elevated daily flow — softer signal |

---

### PROF-007 — Round Sums

**Indicator:** Rounded sum transactions atypical of what would be expected from a client engaging in organic commerce.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `ratio_round_100` | > 0.80 AND `transaction_count_total` ≥ 20 | +0.15 | Over 80% of transactions are exact $100 multiples — extremely unlikely in legitimate retail activity |
| `ratio_round_100` | > 0.60 AND `transaction_count_total` ≥ 20 | +0.08 | Majority round — moderate signal |
| `ratio_round_1000` | > 0.40 AND `transaction_count_total` ≥ 10 | +0.10 | $1,000-exact transactions at high proportion suggest controlled fund movement |
| `count_round_1000` | ≥ 5 | +0.05 | Absolute count floor — ensures even low-frequency $1K-round patterns are noted |

---

## 2. Behavioural & Profile Anomalies

**Typology weight: 0.20**

This typology detects activity that is inconsistent with a customer's declared KYC profile — their occupation, income level, account history, and prior behavioral baseline.

---

### ACCT-001 — Dormant Activation

**Indicator:** A previously inactive account begins to see financial activity — classic money mule activation pattern.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `customer_tenure_days` | > 365 AND `time_span_days` < 90 AND `total_volume` > $10K | +0.25 | Old account with all activity compressed into a short recent window — strong dormancy-to-active signal |
| `customer_tenure_days` | > 180 AND `active_days` < 14 AND `total_volume` > $5K | +0.15 | Mature account that transacted on fewer than 14 distinct days despite significant volume |
| `monthly_volume_cv` | > 3.0 AND `customer_tenure_days` > 180 | +0.12 | Very high month-to-month volume variability on a mature account suggests sudden activation after dormancy |

**Note:** True dormancy requires a prior-period activity flag (e.g., zero transactions in months 1–6 followed by high activity in month 7). This approximation uses `time_span_days` (length of observed activity window) relative to `customer_tenure_days` (total account age) as a proxy.

---

### ACCT-002 — Periodic Patterns

**Indicator:** Accounts receive periodic deposits and are inactive at other times without logical explanation.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `monthly_txn_count_std` | > 50 | +0.12 | Very high standard deviation in monthly transaction count — extreme variability between months |
| `monthly_volume_cv` | > 2.0 AND `active_days` < `time_span_days` × 0.30 | +0.15 | High volume variability combined with low activity density — concentrated bursts with long quiet periods |
| `channel_concentration_hhi` | > 0.85 AND `monthly_txn_count_std` > 20 | +0.08 | Erratic frequency concentrated in a single channel amplifies the periodic pattern signal |

---

### ACCT-003 — Credit Surge

**Indicator:** Sudden increase in card credit usage or applications for new credit.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `count_card` | > 50 AND `channel_concentration_hhi` > 0.60 | +0.15 | High card transaction count with heavy channel concentration — card-dominant activity atypical of diverse banking use |
| `sum_card` | > $20K AND `income` > 0 AND `sum_card` > `income` × 0.70 | +0.15 | Card spend materially exceeding income — financial standing inconsistency via card channel |
| `amount_cv` | > 2.0 AND `count_card` > 20 | +0.08 | High variability in transaction amounts alongside frequent card use — consistent with surging, erratic card activity |

**Note:** A "sudden increase" ideally requires time-split features (e.g., recent 30-day card spend vs. prior 90-day baseline). These are not in the current feature matrix, so the rule approximates the signal using card volume and income ratios.

---

### ACCT-004 — Abrupt Change

**Indicator:** Abrupt change in account activity patterns — key investigation trigger.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `monthly_volume_cv` | > 2.5 | +0.15 | High coefficient of variation on monthly volume — dramatic swings between months |
| `monthly_txn_count_std` | > 30 | +0.10 | High standard deviation in monthly transaction frequency |
| `amount_cv` | > 3.0 | +0.12 | Very high variability in individual transaction amounts — inconsistent sizing across the observation period |
| `std_transaction_amount` | > $5,000 | +0.08 | Large absolute spread in transaction amounts — complements the ratio-based CV with scale context |

---

### PROD-008 — Credit Card Abuse

**Indicator:** Credit card transactions and payments exceptionally high, including excessive cash advances, balance transfers, or luxury items.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `sum_card` | > $50K | +0.20 | Extremely high total card spend — few legitimate retail customers reach this level |
| `sum_card` | > $20K | +0.10 | Elevated card spend — notable but could be legitimate high-income spending |
| `card_luxury_sum` | > $15K AND (`income` == 0 OR `spi` > 3) | +0.20 | Luxury card spend disproportionate to declared income — strongest PROD-008 signal |
| `card_luxury_count` | ≥ 5 AND `income` > 0 AND `card_luxury_sum` > `income` × 0.5 | +0.10 | Frequent luxury transactions consuming more than half of declared annual income |

---

### PROF-001 — Activity vs Expectation

**Indicator:** Transactional activity far exceeds projected activity established at account opening or relationship beginning.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `income_vol_ratio` | > 50 AND `income` > 0 | +0.20 | Transaction volume is 50× declared income — extreme projection mismatch |
| `income_vol_ratio` | > 20 AND `income` > 0 | +0.10 | Volume is 20× income — significant but slightly less extreme |
| `total_volume` | > $100K AND `customer_tenure_days` < 180 | +0.15 | New account processing very high volume — onboarding projections would not anticipate this |

---

### PROF-002 — Financial Standing

**Indicator:** Transactional activity inconsistent with the client's apparent financial standing, usual pattern, or occupation.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `spending_to_income_ratio` | > 10 AND `income` > 0 AND `total_outflow` > $10K | +0.35 | Spending is 10× income — the highest-confidence financial standing mismatch |
| `spending_to_income_ratio` | > 5 AND `income` > 0 AND `total_outflow` > $10K | +0.18 | Spending is 5× income — significant, especially for lower-income profiles |

**Note:** The `total_outflow > $10K` guard ensures rules don't fire on customers with near-zero activity where ratios are mathematically large but economically insignificant.

---

### PROF-005 — Living Beyond Means

**Indicator:** Client appears to be living significantly beyond their means — more spending than income can explain.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `spending_to_income_ratio` | > 5 AND `income` > 0 | +0.20 | Clear lifestyle spending mismatch |
| `spending_to_income_ratio` | > 3 AND `income` > 0 | +0.10 | Moderate mismatch — may indicate proceeds of crime supplementing legitimate income |
| `card_luxury_sum` | > $5K AND `spi` > 2 | +0.10 | Luxury purchases on top of general spending mismatch — lifestyle corroboration |

---

### PROF-008 — Transaction Type/Size Atypical

**Indicator:** Size or type of transactions atypical of what is expected from the client.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `amount_cv` | > 3.0 AND `transaction_count_total` ≥ 20 | +0.15 | Highly irregular transaction sizes across a statistically meaningful sample |
| `max_transaction_amount` | > $50K AND (`income` == 0 OR `max_amount` > `income` × 2) | +0.20 | Single transaction exceeding twice declared annual income — strong outlier |
| `avg_transaction_amount` | > $10K AND `is_individual` == 1 | +0.12 | Average transaction of $10K+ is very atypical for an individual retail banking customer |

---

### PROF-010 — Sudden Change

**Indicator:** Sudden change in the client's financial profile, pattern of activity, or transactions.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `monthly_volume_cv` | > 3.0 | +0.20 | Extreme month-to-month volume swings — the primary sudden-change signal |
| `monthly_volume_cv` | > 2.0 | +0.10 | Significant but less extreme volume variability |
| `monthly_txn_count_std` | > 40 AND `monthly_volume_cv` > 1.5 | +0.12 | Both frequency and volume are erratic together — corroborating a genuine behavioral shift |
| `amount_cv` | > 4.0 | +0.10 | Very high individual transaction amount variability — sizing pattern has also changed |

---

## 3. Trade-Based ML & Shell Entities

**Typology weight: 0.25**

This typology targets businesses whose financial flows are inconsistent with their declared commercial activity, and professional accounts used as conduits for client funds.

---

### GATE-001 — Gatekeeper Atypical Account Use

**Indicator:** A gatekeeper (lawyer, accountant, notary) utilizing their account for transactions not typical of their business — pass-through, excessive cash, payments to non-clients.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `flow_through_ratio` | > 0.92 AND `total_volume` > $200K AND `is_business` == 1 | +0.35 | Near-complete pass-through at high volume for a business account — the clearest gatekeeper signal |
| `flow_through_ratio` | > 0.85 AND `total_volume` > $100K AND `is_business` == 1 | +0.20 | Strong pass-through at significant volume |
| `net_flow` | < 2% of `total_volume` AND `total_volume` > $100K | +0.15 | Net economic retention is less than 2% of all funds processed — the account adds no economic value, it only moves money |

**Note:** `net_flow = total_inflow − total_outflow`. A legitimate business retains profit; a pure conduit retains almost nothing. The 2% threshold accounts for minor timing differences between inflows and corresponding outflows.

---

### PML-TBML-02 — Sector Deviation

**Indicator:** Entity has business activities outside the norm for its declared sector.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `industry_risk_high` | == 1 AND `volume_to_sales_ratio` > 5 | +0.25 | High-risk industry with volume 5× declared sales — sector deviation compounded by inherent risk |
| `industry_risk_high` | == 1 AND `volume_to_sales_ratio` > 2 | +0.12 | Moderate deviation in a risk-elevated sector |
| `is_business` | == 1 AND `industry_risk_high` == 1 AND `count_wire` > 5 | +0.10 | Wire activity on top of sector and volume risk — cross-border dimension of TBML |

---

### PML-TBML-03 — Counterparty Risk

**Indicator:** Entity transacts with a large number of entities in high-volume, high-demand, or unrelated sectors — particularly via wire and EFT.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `count_wire` | > 20 AND `is_business` == 1 | +0.25 | Many wires to different counterparties — the "large number of entities" signal |
| `count_wire` | > 10 AND `is_business` == 1 | +0.12 | Elevated wire count |
| `count_eft` | > 50 AND `is_business` == 1 | +0.15 | Frequent EFT transactions complement wire count for identifying wide counterparty networks |
| `sum_wire` | > $500K | +0.20 | Very high total wire value regardless of count — large-value counterparty exposure |

---

### PML-TBML-04 — Volume Spike (Sudden EFT Inflow)

**Indicator:** Entity receives a sudden inflow of large-value electronic funds transfers.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `eft_credit_sum` | > $500K | +0.35 | Very large cumulative EFT inflow — primary signal |
| `eft_credit_sum` | > $100K | +0.18 | Significant EFT inflow |
| `monthly_txn_cv` | > 2.0 AND `eft_credit_sum` > $50K | +0.10 | High monthly variability in transaction count alongside material EFT inflow — the "spike" temporal pattern |
| `inflow_per_day` | > $5,000/day AND `is_business` == 1 | +0.10 | Sustained high daily inflow for a business — not consistent with small declared sales |

---

### PML-TBML-08 — Round Sum Invoices

**Indicator:** Orders or receives payments for goods in round figures — indicative of fictitious or padded invoicing rather than genuine commercial transactions.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `ratio_round_1000` | > 0.60 AND `is_business` == 1 | +0.20 | Over 60% of business transactions in exact $1,000 increments — fictitious invoicing pattern |
| `ratio_round_1000` | > 0.40 AND `is_business` == 1 | +0.10 | Elevated round proportion for a business |
| `count_round_1000` | ≥ 10 AND `is_business` == 1 | +0.10 | Absolute count — ensures the pattern isn't driven by a handful of coincidental transactions |

---

### PROF-004 — Business Activity Mismatch

**Indicator:** Transactional activity inconsistent with declared business (e.g., no payrolls, no invoices, no normal operating transactions) — strong shell/front company indicator.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `volume_to_sales_ratio` | > 20 AND `sales` > 0 AND `total_volume` > $50K | +0.30 | Volume is 20× declared annual sales — the account moves far more money than the business claims to generate |
| `volume_to_sales_ratio` | > 10 AND `sales` > 0 AND `total_volume` > $50K | +0.15 | 10× declared sales with significant volume |
| `employee_count` | ≤ 2 AND `volume_to_sales_ratio` > 20 AND `is_business` == 1 | +0.20 | Maximum revenue with minimal operating footprint — the classic shell company signature |

---

## 4. Cross-Border & Geographic Risk

**Typology weight: 0.13**

This typology captures transactions with jurisdictions carrying elevated ML/TF risk and wire/transfer patterns inconsistent with the customer's profile.

---

### GEO-001 — Drug Jurisdictions

**Indicator:** Transactions with jurisdictions known to produce or transit drugs or precursor chemicals.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `drug_country_txn_count` | ≥ 5 | +0.25 | Multiple transactions with drug-source countries (Mexico, Colombia, Peru, Bolivia, etc.) |
| `drug_country_txn_count` | ≥ 1 | +0.12 | Even a single transaction warrants a softer flag |

---

### GEO-002 — High ML/TF Risk (FATF Blacklist)

**Indicator:** Transactions with jurisdictions known to be at the highest risk of ML/TF — FATF blacklisted countries (North Korea, Iran, Myanmar).

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `high_risk_fatf_txn_count` | ≥ 3 | +0.55 | Multiple blacklist transactions — the **highest-weighted single rule in the entire system** |
| `high_risk_fatf_txn_count` | ≥ 1 | +0.40 | A single FATF blacklist transaction is a severe regulatory event |

**Rationale for high weights:** Transacting with FATF blacklisted jurisdictions is basically a near-certain compliance violation. As such, the weights should reflect the regulatory significance of blacklisted jurisdictions.

---

### GEO-003 — Locations of Concern (Greylist / Offshore / Underground Banking)

**Indicator:** Activity involving FATF greylisted countries, known offshore financial centres, or jurisdictions with underground banking networks.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `greylist_txn_count` | ≥ 10 | +0.18 | Frequent greylist activity — jurisdictions under increased FATF monitoring |
| `greylist_txn_count` | ≥ 3 | +0.10 | Some greylist exposure |
| `offshore_center_txn_count` | ≥ 3 | +0.18 | Multiple offshore centre transactions (Cayman Islands, BVI, Panama) — secretive banking, layering risk |
| `offshore_center_txn_count` | ≥ 1 | +0.10 | Any offshore exposure |
| `underground_banking_country_count` | ≥ 2 | +0.10 | Transactions with 2+ jurisdictions known for hawala/underground networks (China, Hong Kong, Philippines) |

---

### GEO-004 — FATF Non-Cooperative (Floor Signal)

**Indicator:** Transactions involving countries deemed high risk or non-cooperative by FATF (overlaps GEO-002/003 but serves as a minimum floor).

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `has_any_high_risk_txn` | == 1 | +0.10 | Binary floor: if any FATF-flagged transaction exists (blacklist or greylist), a minimum contribution is always present |

**Note:** GEO-004 is merged into the cross-border rule function as a floor signal rather than a standalone block. It ensures that any customer appearing in GEO-002 or GEO-003 also carries this minimum baseline contribution, and that customers with edge-case greylist exposure (e.g., only 1–2 transactions, below GEO-003 tier thresholds) still receive a small signal - although this is not ideal.

---

### GEO-005 — Frequent Overseas Transfers

**Indicator:** Client makes frequent overseas transfers not in line with their financial profile.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `international_ratio` | > 0.80 AND `international_txn_count` ≥ 10 | +0.18 | Most of the customer's activity is international and frequent — unusual for domestic retail customers |
| `international_ratio` | > 0.50 AND `international_txn_count` ≥ 5 | +0.10 | Majority international with moderate frequency |
| `unique_countries` | ≥ 5 | +0.10 | Wide geographic scatter — harder to explain through legitimate activity than a few recurring partners |
| `sum_westernunion` | > $10K | +0.12 | High WU/MSB spend — inherently international, limited AML oversight in many corridors |
| `sum_westernunion` | > $1K | +0.06 | Any notable WU usage |
| `count_westernunion` | ≥ 5 | +0.08 | Frequent WU transactions — regularity amplifies the risk beyond total amount |
| `card_unique_countries` | ≥ 4 | +0.08 | Card transactions across 4+ countries — geographic footprint beyond wire/EFT channels |

---

### WIRE-008 — Wire Transfer Volume Mismatch

**Indicator:** Large wire transfers or high volume through an account that doesn't fit the expected pattern for that customer type.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `sum_wire` | > $500K | +0.30 | Very large total wire value — well beyond what most retail or small business customers generate |
| `sum_wire` | > $100K | +0.15 | Significant wire volume |
| `sum_wire` | > `income` AND `income` > 0 | +0.10 | Wire volume exceeds declared annual income — profile mismatch regardless of absolute amount |

---

### WIRE-010 — Multiple Wire Senders/Receivers

**Indicator:** Client sending to or receiving wire transfers from multiple clients — possible ML intermediary.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `count_wire` | ≥ 15 | +0.18 | Large number of distinct wire transactions implies many counterparties |
| `count_wire` | ≥ 8 | +0.10 | Elevated wire count |

---

## 5. Human Trafficking

**Typology weight: 0.12**

This typology operationalises the FINTRAC HT-SEX indicator set using Merchant Category Code (MCC) level card features and EMT/EFT transaction patterns. It is the most feature-rich typology, relying heavily on card MCC data for precision.

---

### HT-SEX-01 — Retail & Gift Card (Rounded Purchases)

**Indicator:** Rounded sum purchases at grocery stores and other retailers that sell gift cards and/or prepaid credit cards.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `card_retail_count` | ≥ 10 AND `ratio_round_100` > 0.50 | +0.18 | High retail card frequency with majority round amounts — gift card bulk purchase pattern |
| `card_retail_count` | ≥ 5 AND `ratio_round_100` > 0.50 | +0.10 | Moderate retail frequency with rounds |

**MCC coverage:** `card_retail_count` aggregates transactions at grocery (5411), pharmacy (5912), discount (5310), and convenience store MCCs.

---

### HT-SEX-02 — High-Value Convenience Store Purchases

**Indicator:** Atypically high-value purchases at convenience stores — likely for gift cards or money transfer products.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `card_retail_count` | ≥ 10 AND `avg_transaction_amount` > $200 | +0.12 | High volume retail activity with unusually high average amount — consistent with gift card or prepaid instrument purchases rather than personal shopping |

**Note:** Without a `card_retail_avg_amount` feature (average specifically for retail MCCs), this rule uses the overall `avg_transaction_amount` as a proxy. A retail-specific average would improve precision, but would require far too many additional features for diminishing returns.

---

### HT-SEX-03 — Lifestyle & Luxury Spend

**Indicator:** Luxury purchases (dealerships, high-end restaurants, jewelry) inconsistent with reported income or occupation.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `card_luxury_sum` | > $10K AND (`income` == 0 OR `spi` > 3) | +0.28 | Significant luxury spend with no declared income or spending ratio > 3× — primary proceeds-integration signal |
| `card_luxury_sum` | > $5K AND (`income` == 0 OR `spi` > 2) | +0.15 | Moderate luxury spend with income mismatch |
| `card_luxury_count` | ≥ 5 | +0.08 | Frequency distinguishes lifestyle spending from a single isolated purchase |

**MCC coverage:** `card_luxury_sum` and `card_luxury_count` aggregate jewellers (5094, 5944) and related luxury MCCs.

---

### HT-SEX-04 — Victim Transit: Parking Spend

**Indicator:** Frequent low-value payments for parking — indicative of victim transit or client-facing activity.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `card_parking_count` | ≥ 20 AND `unique_cities` ≥ 3 | +0.15 | Frequent parking across multiple cities — not consistent with personal commuting, consistent with victim transit management |
| `card_parking_count` | ≥ 10 | +0.08 | Elevated parking frequency — softer signal without multi-city corroboration |

**MCC coverage:** MCC 7523 (Parking Lots and Garages).

---

### HT-SEX-05 — Victim Maintenance: Food Delivery

**Indicator:** Frequent purchases from food delivery services — victims kept in controlled locations are often fed via delivery.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `card_delivery_count` | ≥ 20 AND `card_accommodation_count` ≥ 3 | +0.15 | Frequent delivery alongside hotel/accommodation spend — consistent with victims maintained at venues |
| `card_delivery_count` | ≥ 10 | +0.08 | Elevated delivery frequency |

**MCC coverage:** MCC 5814 and food delivery app MCCs.

---

### HT-SEX-06 — Digital Integration: Crypto, Gambling, Online

**Indicator:** Transfers to virtual currency, online gambling, or digital investment platforms — used to launder trafficking proceeds and pay for online advertising.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `card_digital_count` | ≥ 5 | +0.15 | Frequent digital/crypto/gambling card transactions |
| `card_digital_count` | ≥ 2 | +0.08 | Some digital activity — smaller signal |

**MCC coverage:** `card_digital_count` captures MCC 4816 (Computer Network/Information Services) and MCC 7995 (Gambling Transactions).

---

### HT-SEX-07 — Geographic & Travel: Multi-City Cash

**Indicator:** Cash deposit city matching hotel booking city across multiple locations — high-confidence trafficking mobility indicator.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `unique_cities` | ≥ 5 AND `night_abm_count` ≥ 5 | +0.35 | Cash withdrawals at night across 5+ cities — the strongest available proxy for multi-city victim management |
| `unique_cities` | ≥ 3 AND `night_abm_count` ≥ 3 | +0.22 | Multi-city night cash pattern |
| `unique_cities` | ≥ 3 AND `night_abm_count` ≥ 1 | +0.12 | Multi-city presence with any night cash |

**Limitation:** Full HT-SEX-07 requires matching the city of each ABM withdrawal with the city of each hotel booking on the same or adjacent date. This would be a transaction-level join and was unfeasible for this project. This rule is the best aggregate-level approximation; a combination of night ABM withdrawals and multi-city presence is a fairly strong proxy.

---

### HT-SEX-08 — Accommodation Spend (Non-Residential Venues)

**Indicator:** Payments to online accommodation or travel websites in non-residential cities — hotels used as working venues.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `card_accommodation_count` | ≥ 5 | +0.18 | Frequent hotel transactions |
| `card_accommodation_count` | ≥ 3 | +0.12 | Moderate hotel transaction count |
| `card_accommodation_sum` | > $2K | +0.08 | Total accommodation spend adds value context |

**MCC coverage:** MCC 7011 (Lodging Services / Hotels, Motels, Resorts).

---

### HT-SEX-10 — International Recruitment & Travel

**Indicator:** Large international travel purchases targeting known trafficking source countries.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `ht_source_country_txn_count` | ≥ 5 | +0.25 | Multiple transactions with known trafficking origin countries — strong recruitment/transport signal |
| `ht_source_country_txn_count` | ≥ 3 | +0.15 | Moderate exposure to source countries |
| `card_travel_sum` | > $5K | +0.22 | Large travel spend — international airfare at the scale typical of recurring victim transport |
| `card_travel_sum` | > $2K | +0.12 | Elevated travel spend |
| `sum_westernunion` | > $2K AND `ht_source_country_txn_count` ≥ 1 | +0.10 | WU remittances to source countries — common mechanism for paying recruiters |

**MCC coverage:** `card_travel_sum` aggregates MCCs 4511 (Airlines) and 4722 (Travel Agencies).

---

### HT-SEX-12 — After-Hours Operations

**Indicator:** Merchant POS transactions at spa, massage, and escort establishments after business hours (10 PM – 6 AM).

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `card_adult_afterhours_count` | ≥ 3 | +0.45 | **Highest-weighted single rule in the human trafficking typology.** Direct MCC evidence of after-hours spa/massage transactions from a flagged merchant category |
| `card_adult_afterhours_count` | ≥ 1 | +0.28 | Even one after-hours adult MCC transaction is a strong signal |
| `card_afterhours_count` | ≥ 10 AND `night_transaction_ratio` > 0.40 AND `transaction_count_total` ≥ 10 | +0.12 | Fallback when adult-MCC-specific data is absent: general after-hours card activity at scale |

**MCC coverage:** `card_adult_afterhours_count` is specifically filtered to MCC 7297 (Massage Parlors) and 7298 (Health and Beauty Spas) between 22:00 and 06:00. Legitimate businesses in these categories rarely process client payments past midnight.

---

### HT-SEX-13 — Asset Procurement: High-Value Transfers vs Income

**Indicator:** High-value transfers disproportionate to reported income — real estate and asset acquisition funded by trafficking proceeds.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `spending_to_income_ratio` | > 10 AND `income` > 0 AND `total_volume` > $50K | +0.18 | Very high spending ratio at significant volume — unexplained wealth in HT context |
| `total_volume` | > $200K AND `income` > 0 AND `spi` > 5 | +0.12 | Large absolute volume with material income mismatch |

**Note:** PROF-002 uses the same features to flag general financial standing mismatch. In the HT typology, the identical signal is interpreted as proceeds of trafficking rather than generic financial crime, and contributes specifically to the HT indicator trace for the explanation model.

---

### HT-SEX-14 — Rental Payments via EMT/EFT

**Indicator:** Monthly payments to multiple individuals or entities involved in residential rentals — financial footprint of victim accommodation sites.

| Feature | Condition | Contribution | Rationale |
|---|---|---|---|
| `max(emt_debit_count, eft_debit_count)` | ≥ 8 | +0.18 | Frequent recurring outbound transfers — consistent with multiple venue rent payments |
| `max(emt_debit_count, eft_debit_count)` | ≥ 4 | +0.10 | Moderate recurring outbound transfers |

**Implementation note:** The rule takes the maximum of `emt_debit_count` and `eft_debit_count` rather than summing them. This prevents double-counting when the same rental payment is structured as either an EMT or EFT in different months.

---

## Excluded Indicators

| Indicator | Reason |
|---|---|
| **PROF-003** (Geographic Volume) | Requires a geographic peer-group baseline — the mean and standard deviation of transaction volume by postal code or city. No such peer-group aggregation exists in the current feature matrix. The Isolation Forest model captures geographic outliers implicitly through `transaction_count_total`, but a deterministic rule cannot fire without the peer reference. |
| **ATYPICAL-008** (Same-Day Turnover) | True detection requires matching individual inflow and outflow timestamps within a 24-hour window at the transaction level. Customer-level aggregate features cannot resolve same-day pairing. ATYPICAL-007's `flow_through_ratio` captures the same economic signal at the aggregate level, making a separate rule redundant and potentially misleading. |

**Note:** The above rules were excluded primarily due to time constraints - they have been kept in the practical AML indicator database for reference purposes.