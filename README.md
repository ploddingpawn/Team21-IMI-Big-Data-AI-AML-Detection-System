# IMI Big Data & AI AML Detection System

This repository contains a comprehensive Anti-Money Laundering (AML) detection pipeline. The architecture leverages a hybrid modeling approach combining statistical anomaly detection, deterministic rule checks, and behavior-based clustering to accurately score and explain customer risk.

## Pipeline Architecture

The detection and explanation pipeline is structured into three main phases: Scoring, Fusion, and Explanation.

### 1. Multi-Layer Scoring Engine
The system analyzes customer behavior through multiple distinct lenses, generating independent risk signals:

- **Typology-Specific Anomaly Detection (Isolation Forests):** Instead of a single generalized model, the pipeline trains independent models—each mapped to a specific AML typology (e.g., structuring, layering, profile mismatch). This effectively isolates illicit behaviors and scores the severity of the anomalies independently.
- **Rule-Based Hard Flags:** Deterministic checks evaluate customer transactions against rigid regulatory thresholds and industry-standard limits. This provides strict, non-negotiable compliance signals.
- **Behavioral Clustering (K-Means):** Evaluates the statistical scoring dimensions to group customers into behaviorally similar risk tiers, further validating and mapping typological profiles.

### 2. Customer-Level Score Fusion & Indicator Tracking
The independent signals must be aggregated and interpreted:

- **Customer-Level Weighted Score:** The distinct signals (anomaly scores, clustering tier, and rule triggers) are weighted and combined into a final, unified AML risk score per customer.
- **Indicator Tracking & Contribution:** During the scoring stage, for each underlying technique, the system actively tracks all significant contributing indicators. Every driving factor is recorded alongside its associated red flag ID and its exact quantitative contribution to the final score.

### 3. Automated Narrative Generation Engine
For investigation and auditability, the finalized scores and tracking data are synthesized into professional, evidence-based narratives:

- **Contextual Data Ingestion:** The engine receives the unified risk scores and the associated indicator traces as a structured payload. This ensures the model bridges the gap between Layer 1 anomalies, Layer 2 regulatory flags, and Layer 3 peer-group context.
- **Synthesized Investigative Summaries:** The engine interprets these mathematical signals to produce a plain-language explanation that leads with the dominant risk concern. It explicitly cites the mapped indicator IDs (e.g., ATYPICAL-007) and applies logic-driven disclaimers for low-coverage profiles, ensuring the final output is a defensible and transparent record of the model's decision-making process.