# Solution Steps

1. Open `analysis/analyze.py` and implement `build_member_features(conn)` to create a leakage-free member feature table: aggregate transactions only from the scoring window (`2024-01-01` inclusive to `2024-10-01` exclusive). Use only member attributes + transaction-derived aggregates in that window (frequency, recency, points earned/redeemed, redemption intensity, and channel mix). Do not touch any transactions in the held-out window.

2. In `score_and_validate_churn(conn, member_features)`, compute an interpretable formula-based churn-risk score using only the scoring-period features (e.g., higher risk for larger recency, fewer score-period transactions, lower redemption intensity, and lower points earned per transaction). Bucket members by quantiles of the computed score (e.g., 5 buckets).

3. For held-out validation, query later inactivity from the transactions table for the held-out window (`2024-10-01` to `2025-01-01`). Merge held-out transaction counts into the scored members, then compute per-bucket inactivity rate and average held-out transactions. Confirm and print that inactivity increases monotonically from low- to high-risk buckets (and optionally compute a Spearman correlation with bucket order).

4. In `detect_dealer_anomalies(conn)`, build dealer-level point behavior features using ONLY the scoring window: total earned points, total redeemed points, redemption rate, etc. Then, within each dealer region, compute robust peer-relative metrics (median/MAD based robust z-scores) for (a) log earned points and (b) redemption rate. Define a simple anomaly score such as `z_earn - z_redeem_rate` so that high-earning/low-redemption dealers rise to the top.

5. Select a small set of anomalous dealers per region using rule-based thresholds (e.g., robust z of earned above a cutoff and robust z of redemption below a cutoff; otherwise fall back to the top N by anomaly score). Return a table of those dealers with their peer z-scores and anomaly score.

6. Estimate each implicated dealer’s recent point transaction exposure as a defensible range: compute weekly totals of points earned/redeemed in the held-out/recent window (`2024-10-01` to `2025-01-01`), include zero-activity weeks in the weekly distribution, take P10 and P90 of weekly totals, and scale by the number of weeks to form a range for total exposure. Also compute the realized totals for reference.

7. Update `main()` to call the three functions in order and print finance-facing outputs: (1) held-out inactivity-by-risk-bucket table, (2) the anomalous dealer list with earned/redemption behavior, exposure ranges, and realized recent points, and (3) a concise conclusion tying the numbers back to the computed risk scores and dealer exposure estimates.

