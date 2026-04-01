# AML Hybrid Model Architecture

The core philosophy, or rather forced conclusion behind this detection pipeline is that neither machine learning nor deterministic rules are sufficient on their own - but combining them is difficult. Rule-based systems lack granularity and can be fairly easily bypassed (presumably) by sophisticated bad actors. Conversely, pure unsupervised Machine Learning (in our case, Isolation Forests) flags anomalous behavior, but has a high false-positive rate because "unusual" doesn't necessarily mean "illegal".

The Hybrid Model resolves this by fusing both approaches (plus a little extra) into a **3-Pillar Ensemble**.

---

## Pillar 1: The Dynamic Score (Corroborated Risk)
**Weight: 50%**

This pillar broadly catches bad actors. It requires both the rule-based system and the ML model to "agree" on the suspiciousness of a customer (i.e. we multiply the scores).

For each of the five illicit typologies (Human Trafficking, Trade Based Shell, etc.), the pipeline calculates a **Dynamic Multiplier**:
```text
Dynamic Score = Rule Engine Score × Isolation Forest Probability
```
*   If a customer breaks a FINTRAC regulatory threshold, their rule score jumps.
*   If the Isolation Forest also determines their behavior shape matches an anomaly, their IF probability jumps.
*   Because the two are multiplied together, a customer would trip *both* the deterministic rule and the probabilistic ML boundary anomaly to score highly here.

This combination is greater than the sum of its parts. The final Dynamic Pillar is the normalized sum of these 5 multipliers.

---

## Pillar 2: Coverage-Based Fallback (Zero-Day Anomalies)
**Weight: 20%**

When we came up with it (after a long and grueling process), we realized the dynamic score was flawed - it was entirely dependent on the rule systen. If no rules were triggered, the dynamic score would be 0, even if the Isolation Forest detected highly anomalous behavior. Hence, this pillar (in theory) catches the sophisticated bad actors that perfectly evade all internal rule thresholds but behave highly suspiciously. In perhaps less flattering terms, it is a fallback. 

When a customer evades all rules for a given typology, their dynamic score is `0`. To offset this, we calculate `coverage`—the fraction of the 5 rule typologies that triggered. We then build the fallback equation:
```text
Fallback Score = (1.0 - coverage) × IF_Score_Weighted
```
*   **Coverage = 1.0 (All rules fired):** The IF score contributes `0` points here because it is already working perfectly as a multiplier in Pillar 1.
*   **Coverage = 0.0 (No rules fired):** The rules are blind to this customer. The equation pivots and gives 100% of the scoring weight to the Isolation Forest anomaly detection, acting as an unsanctioned net to catch zero-day typologies.

---

## Pillar 3: KMeans Behavioral Peer-Grouping
**Weight: 30%**

We had another glaring problem - the Isolation Forest model could be (and likely is) skewed by massive dataset outliers. This pillar provides "relative severity". KMeans allows us to group similar customers into behavioral cohorts, determining how severely anomalous a customer is compared strictly to people who act like them.

### 1. Robust Scaled Feature Space
We feed all 61,000 customers into the KMeans algorithm using *only* their 5 pure Isolation Forest anomaly probabilities. Because anomalies are highly left-skewed, we need to aggressively flatten them using a `RobustScaler()` (using the Interquartile Range rather than strict variance). This prevents ultra-anomalies from dragging cluster centroids artificially outward.

### 2. The Thin-Cluster Refit Loop
KMeans begins by slicing the 61,000 customers into `K=8` behavioral archetypes. 
We must also institute a strict statistical safeguard: a percentile rank is useless if the sample size is insignificant. If any cluster contains fewer than `200` members, we reject the fit and dynamically drop `K` to `7`, and refit. This should iterate downwards until stable, macro-level cohorts are achieved.

### 3. Within-Cluster Percentile (`kmeans_score`)
Once clustered, the pipeline evaluates each customer against the rest of their cluster members using their raw dynamic cross-corroboration sum.
If the customer has a worse dynamic score than 95% of the people in their cluster, their `kmeans_score` becomes `0.950`.

### 4. Risk-Tier Adjustment (`adjusted_kmeans_score`)
As we found out from quite poor results, a pure percentile is dangerous: being the "most anomalous" customer inside a cluster of 50,000 completely legitimate retail users means nothing. You are the "worst of the best."
Thankfully, we discovered we could prevent these false positives from skyrocketing by scaling the percentile by the historical baseline risk of the cluster itself (`cluster_risk_tier`). 
```text
Adjusted KMeans Score = kmeans_score × cluster_risk_tier
```
Now, being the 99th percentile in a safe cluster earns a fraction of the points compared to being the 50th percentile in a high-risk trafficking ring cluster.

---

## The Final Ensemble Equation

The final fusion combines all three pillars into a single `0.0` to `1.0` continuous feature.

> **Hybrid Risk =** 
> `0.50` × (Corroborated Dynamic Mix) + 
> `0.20` × (Zero-Day IF Fallback) + 
> `0.30` × (Adjusted Peer-Group Percentile)

This ensures that the top 1% of flagged customers have either tripped hard FINTRAC rules while behaving anomalously, circumvented rules while behaving irregularly, or are the single worst offending entity inside an already-suspicious network.

### Normalization & Compliance Hierarchy
Note that the raw weights (`0.50`, `0.20`, `0.30`) do not dictate the absolute ceiling of the final score. After the raw sums are calculated, the entire population distribution is min-max normalized from 0.0 to 1.0. 

The weights have been structured primarily to ensure regulatory compliance, and reflect to our best ability the relative impartial importance of each pillar:
* **Regulatory Primacy:** Basic FINTRAC/FinCEN compliance should take precedence. If Customer A breaks explicit $10k reporting thresholds, the model should score a non-definite ML anomaly higher than them. The dynamic score is weighed (`0.50`) to ensure explicit rule-breakers loosely comprise a 'straightforward' half of the final score.
* **Throttling the Safety Net:** The fallback is necessary, but pure ML anomalies should have a high false-positive rate. By throttling the fallback to `0.20`, a 'maximum undetected anomaly' (Coverage=0) scores `0.50` raw points (`0.2 IF + 0.3 KMeans`). A 'maximum rule-breaker' (Coverage=1) scores `0.80` raw points (`0.5 Dynamic + 0.3 KMeans`). 
---

## Sanity Check & Intuitn

We can to a limited capacity test the model against known labels to validate: the Hybrid Ensemble's ROC-AUC (`0.852`) is higher than any of its individual constituents:
- Hybrid (weighted ensemble):  0.852
- IF weighted alone:           0.840
- IF max alone:                0.843
- Rules alone (weighted):      0.734
- Dynamic score (IF×Rule):     0.818
- KMeans adjusted percentile:  0.724

The hybrid model out-performs its components because of orthogonal signals (at least we think as much). The three pillars (are intended to) cover most aspects of illicit financial activity with minimal overlap.

Because the components fail on *different* types of customers (their errors are ideally uncorrelated), blending them dilutes the false positives (since they rarely agree on a false positive) while stacking true positives. The resulting ensemble creates an umbrella that no single methodology could provide alone.
