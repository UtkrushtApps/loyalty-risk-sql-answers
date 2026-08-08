import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import psycopg2


def get_connection() -> Any:
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "loyalty"),
        user=os.getenv("DB_USER", "loyalty_user"),
        password=os.getenv("DB_PASSWORD", "loyalty_pass"),
    )


def build_member_features(conn: Any) -> pd.DataFrame:
    """Build one row per member using ONLY the scoring-period behavior.

    Scoring period: 2024-01-01 (inclusive) to 2024-10-01 (exclusive).

    Leakage-free requirement: features must not use any transactions from the
    later validation/held-out window (>= 2024-10-01).
    """

    score_start = "2024-01-01"  # inclusive
    score_end_exclusive = "2024-10-01"  # exclusive
    score_end_inclusive = "2024-09-30"  # for recency math

    sql = f"""
    WITH score_tx AS (
        SELECT
            t.member_id,
            t.dealer_id,
            t.transaction_date,
            t.amount,
            t.points_earned,
            t.points_redeemed,
            t.channel
        FROM transactions t
        WHERE t.transaction_date >= DATE %%(score_start)s
          AND t.transaction_date <  DATE %%(score_end_exclusive)s
    )
    SELECT
        m.member_id,
        m.tier,
        m.region,

        COUNT(st.transaction_date) AS score_txn_count,

        COALESCE(SUM(st.points_earned), 0)::double precision AS points_earned_score,
        COALESCE(SUM(st.points_redeemed), 0)::double precision AS points_redeemed_score,
        COALESCE(SUM(st.points_earned - st.points_redeemed), 0)::double precision AS net_points_score,

        -- Redemption intensity: redeemed as share of earned points
        COALESCE(
            SUM(st.points_redeemed)::double precision / NULLIF(SUM(st.points_earned), 0),
            0
        ) AS redemption_rate_score,

        -- Average monetary/points intensity during the score window
        COALESCE(AVG(st.amount), 0)::double precision AS avg_amount_score,
        COALESCE(SUM(st.points_earned)::double precision / NULLIF(COUNT(st.transaction_date), 0), 0)
            AS avg_points_earned_per_txn_score,

        -- Recency: days between last txn in score window and the score window end
        CASE
            WHEN MAX(st.transaction_date) IS NULL THEN NULL
            ELSE (DATE %%(score_end_inclusive)s - MAX(st.transaction_date))::int
        END AS recency_days_score,

        CASE
            WHEN MIN(st.transaction_date) IS NULL THEN NULL
            ELSE (DATE %%(score_end_inclusive)s - MIN(st.transaction_date))::int
        END AS tenure_days_score,

        -- Channel engagement
        COALESCE(SUM(CASE WHEN st.channel = 'store' THEN st.points_earned ELSE 0 END)::double precision
            / NULLIF(SUM(st.points_earned), 0), 0) AS store_points_share_score,
        COALESCE(SUM(CASE WHEN st.channel = 'app' THEN st.points_earned ELSE 0 END)::double precision
            / NULLIF(SUM(st.points_earned), 0), 0) AS app_points_share_score,
        COALESCE(SUM(CASE WHEN st.channel = 'web' THEN st.points_earned ELSE 0 END)::double precision
            / NULLIF(SUM(st.points_earned), 0), 0) AS web_points_share_score,
        COALESCE(SUM(CASE WHEN st.channel = 'call_center' THEN st.points_earned ELSE 0 END)::double precision
            / NULLIF(SUM(st.points_earned), 0), 0) AS call_center_points_share_score,

        COUNT(DISTINCT st.channel) AS channel_diversity_score,

        -- Redeem frequency (txn-level): fraction of score txns where some points were redeemed
        COALESCE(
            SUM(CASE WHEN st.points_redeemed > 0 THEN 1 ELSE 0 END)::double precision
            / NULLIF(COUNT(st.transaction_date), 0),
            0
        ) AS redeem_txn_freq_score,

        -- Member age at score end (not an outcome)
        (DATE %%(score_end_inclusive)s - m.enrollment_date)::int AS member_age_days_at_score_end

    FROM loyalty_members m
    LEFT JOIN score_tx st
        ON st.member_id = m.member_id
    GROUP BY m.member_id, m.tier, m.region, m.enrollment_date
    """

    df = pd.read_sql_query(
        sql,
        conn,
        params={
            "score_start": score_start,
            "score_end_exclusive": score_end_exclusive,
            "score_end_inclusive": score_end_inclusive,
        },
    )

    # If a member truly has no score txns (unlikely in this synthetic data), set
    # recency to the maximum observed window to avoid NaNs in scoring.
    if df["recency_days_score"].isna().any():
        max_recency = int(df["recency_days_score"].max(skipna=True))
        df["recency_days_score"] = df["recency_days_score"].fillna(max_recency)

    # Replace possible missing txn-derived fields (should be 0 because of COALESCE)
    numeric_cols = [
        "score_txn_count",
        "points_earned_score",
        "points_redeemed_score",
        "net_points_score",
        "avg_amount_score",
        "avg_points_earned_per_txn_score",
        "redemption_rate_score",
        "store_points_share_score",
        "app_points_share_score",
        "web_points_share_score",
        "call_center_points_share_score",
        "channel_diversity_score",
        "redeem_txn_freq_score",
        "member_age_days_at_score_end",
        "tenure_days_score",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    return df


def _robust_z(x: pd.Series) -> pd.Series:
    """Robust z-score using median and MAD."""
    median = x.median()
    mad = (x - median).abs().median()
    if mad == 0 or np.isnan(mad):
        # Fallback: use std
        std = x.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - median) / std
    return 0.6745 * (x - median) / mad


def score_and_validate_churn(conn: Any, member_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a formula-based churn-risk score.

    Risk score is based ONLY on scoring-period features.
    Held-out validation uses later transactions (>= 2024-10-01) and must not
    feed back into the score.
    """

    df = member_features.copy()

    # --- Formula components (interpretable, monotonic in risk direction) ---
    # 1) Recency: more days since last score txn => higher churn risk
    df["recency_norm"] = df["recency_days_score"] / max(df["recency_days_score"].max(), 1)

    # 2) Frequency: fewer score txns => higher risk
    # Use log scaling to reduce effect of heavy users.
    freq = np.log1p(df["score_txn_count"].astype(float))
    fmin, fmax = float(freq.min()), float(freq.max())
    if fmax - fmin < 1e-9:
        df["freq_inverted"] = 0.5
    else:
        freq_norm = (freq - fmin) / (fmax - fmin)
        df["freq_inverted"] = 1.0 - freq_norm

    # 3) Redemption intensity: lower redemption engagement => higher risk
    # Redeemed-share is in [0, ~]; normalize within observed min/max.
    rr = df["redemption_rate_score"].astype(float)
    rr_min, rr_max = float(rr.min()), float(rr.max())
    if rr_max - rr_min < 1e-9:
        df["redeem_inverted"] = 0.5
    else:
        rr_norm = (rr - rr_min) / (rr_max - rr_min)
        df["redeem_inverted"] = 1.0 - rr_norm

    # 4) Points intensity: lower earned per txn => higher risk
    earn_per_txn = np.log1p(df["avg_points_earned_per_txn_score"].astype(float))
    emin, emax = float(earn_per_txn.min()), float(earn_per_txn.max())
    if emax - emin < 1e-9:
        df["earn_inverted"] = 0.5
    else:
        earn_norm = (earn_per_txn - emin) / (emax - emin)
        df["earn_inverted"] = 1.0 - earn_norm

    # Weighted risk score (weights sum to 1)
    # Justification (finance-facing): inactivity risk increases with
    # (a) larger recency, (b) lower frequency, and (c) weaker point engagement.
    w_recency = 0.50
    w_freq = 0.30
    w_redeem = 0.10
    w_earn = 0.10

    df["risk_score"] = 100.0 * (
        w_recency * df["recency_norm"]
        + w_freq * df["freq_inverted"]
        + w_redeem * df["redeem_inverted"]
        + w_earn * df["earn_inverted"]
    )

    # --- Bucketization ---
    # Use quantiles so each bucket has a similar size.
    n_buckets = 5
    try:
        df["risk_bucket"] = pd.qcut(df["risk_score"], q=n_buckets, labels=list(range(1, n_buckets + 1)))
    except ValueError:
        # If qcut fails due to duplicates, fall back to rank-based equal width.
        df["risk_bucket"] = pd.cut(df["risk_score"].rank(method="first"), bins=n_buckets, labels=list(range(1, n_buckets + 1)))

    df["risk_bucket"] = df["risk_bucket"].astype(int)

    # --- Held-out validation: later inactivity outside the scoring window ---
    heldout_sql = """
    SELECT
        t.member_id,
        COUNT(*) AS heldout_txn_count
    FROM transactions t
    WHERE t.transaction_date >= DATE '2024-10-01'
      AND t.transaction_date <  DATE '2025-01-01'
    GROUP BY t.member_id
    """
    heldout = pd.read_sql_query(heldout_sql, conn)

    df = df.merge(heldout, on="member_id", how="left")
    df["heldout_txn_count"] = df["heldout_txn_count"].fillna(0).astype(int)
    df["heldout_inactive_flag"] = (df["heldout_txn_count"] == 0).astype(int)

    bucket_summary = (
        df.groupby(["risk_bucket"], as_index=False)
        .agg(
            members=("member_id", "count"),
            inactive_rate=("heldout_inactive_flag", "mean"),
            avg_heldout_txns=("heldout_txn_count", "mean"),
            median_heldout_txns=("heldout_txn_count", "median"),
        )
        .sort_values("risk_bucket")
    )

    # Add a simple monotonic trend metric: Spearman correlation between bucket index and inactivity
    # (bucket index increases with risk by construction).
    bucket_order = bucket_summary["risk_bucket"].values
    inactive = bucket_summary["inactive_rate"].values
    if len(bucket_summary) >= 2:
        spearman = pd.Series(inactive).corr(pd.Series(bucket_order), method="spearman")
    else:
        spearman = np.nan
    bucket_summary["spearman_with_bucket_order"] = spearman

    # Store key evidence columns for downstream summary
    keep_cols = [
        "member_id",
        "tier",
        "region",
        "score_txn_count",
        "recency_days_score",
        "redemption_rate_score",
        "risk_score",
        "risk_bucket",
        "heldout_txn_count",
        "heldout_inactive_flag",
    ]
    scored = df[keep_cols].copy()

    return scored, bucket_summary


def detect_dealer_anomalies(conn: Any) -> pd.DataFrame:
    """Detect anomalous dealers within each region peer group.

    We compare dealers to their regional peers using robust (median/MAD) z-scores.
    No ML training and no outcome/held-out data is used to flag anomalies.

    Anomalies are driven by (high point earning) + (low redemption intensity),
    which matches the synthetic generation logic for a subset of dealers.

    Finally, we estimate their recent transaction-point exposure as a defensible
    range using the variability of weekly earned/redemption totals in the recent
    (held-out) observation window.
    """

    score_start = "2024-01-01"
    score_end_exclusive = "2024-10-01"

    dealer_score_sql = """
    WITH score_dealer_tx AS (
        SELECT
            t.dealer_id,
            d.region,
            t.transaction_date,
            t.points_earned,
            t.points_redeemed,
            t.channel
        FROM transactions t
        JOIN dealers d ON d.dealer_id = t.dealer_id
        WHERE t.transaction_date >= DATE %%(score_start)s
          AND t.transaction_date <  DATE %%(score_end_exclusive)s
    )
    SELECT
        dealer_id,
        region,
        COUNT(*) AS score_txn_count,
        SUM(points_earned)::double precision AS points_earned_score,
        SUM(points_redeemed)::double precision AS points_redeemed_score,
        (SUM(points_earned - points_redeemed))::double precision AS net_points_score,
        COALESCE(SUM(points_redeemed)::double precision / NULLIF(SUM(points_earned),0), 0) AS redemption_rate_score,
        SUM(points_earned)::double precision / NULLIF(SUM(points_redeemed + 0.0), 0) AS earned_to_redeemed_ratio_score,
        AVG(points_earned)::double precision AS avg_points_earned_per_txn_score,
        AVG(points_redeemed)::double precision AS avg_points_redeemed_per_txn_score,
        COUNT(DISTINCT channel) AS channel_diversity_score
    FROM score_dealer_tx
    GROUP BY dealer_id, region
    """

    dealer_score = pd.read_sql_query(
        dealer_score_sql,
        conn,
        params={"score_start": score_start, "score_end_exclusive": score_end_exclusive},
    )

    # Join dealer names for reporting
    names = pd.read_sql_query(
        """
        SELECT dealer_id, dealer_name
        FROM dealers
        """,
        conn,
    )
    dealer_score = dealer_score.merge(names, on="dealer_id", how="left")

    # Robust scoring within each region
    dealer_score["log_points_earned_score"] = np.log1p(dealer_score["points_earned_score"].astype(float))

    dealer_score["z_earn"] = np.nan
    dealer_score["z_redeem_rate"] = np.nan
    dealer_score["anomaly_score"] = np.nan

    for region, grp in dealer_score.groupby("region"):
        idx = grp.index
        z_earn = _robust_z(grp["log_points_earned_score"].astype(float))
        z_redeem = _robust_z(grp["redemption_rate_score"].astype(float))
        # High earning + low redemption => positive anomaly score.
        anomaly = z_earn - z_redeem

        dealer_score.loc[idx, "z_earn"] = z_earn.values
        dealer_score.loc[idx, "z_redeem_rate"] = z_redeem.values
        dealer_score.loc[idx, "anomaly_score"] = anomaly.values

    # Select a small set per region
    selected_rows = []
    for region, grp in dealer_score.groupby("region"):
        grp = grp.sort_values("anomaly_score", ascending=False).copy()

        # Primary rule: earn is meaningfully high and redemption meaningfully low
        cand = grp[(grp["z_earn"] > 1.5) & (grp["z_redeem_rate"] < -0.8)]
        if len(cand) == 0:
            cand = grp.head(3)  # fallback
        else:
            cand = cand.head(min(3, len(cand)))

        selected_rows.append(cand)

    anomalies = pd.concat(selected_rows, ignore_index=True)

    # --- Estimate recent exposure range in the held-out window ---
    # Recent window: 2024-10-01 to 2025-01-01 (the period we used for member validation).
    heldout_dealer_sql = """
    SELECT
        t.dealer_id,
        date_trunc('week', t.transaction_date)::date AS week_start,
        SUM(t.points_earned)::double precision AS points_earned_week,
        SUM(t.points_redeemed)::double precision AS points_redeemed_week,
        COUNT(*) AS txn_count_week
    FROM transactions t
    WHERE t.transaction_date >= DATE '2024-10-01'
      AND t.transaction_date <  DATE '2025-01-01'
    GROUP BY t.dealer_id, week_start
    """
    dealer_weekly = pd.read_sql_query(heldout_dealer_sql, conn)

    # Build full week grid (week_start aligned to Postgres date_trunc('week') which is Monday)
    start = pd.Timestamp("2024-10-01")
    end = pd.Timestamp("2024-12-31")
    # Shift start backward to Monday
    start_monday = start - pd.Timedelta(days=start.weekday())
    week_starts = pd.date_range(start_monday, end, freq="7D").date
    n_weeks = len(week_starts)

    anomalies = anomalies.copy()
    anomalies["realized_points_earned_recent"] = 0.0
    anomalies["realized_points_redeemed_recent"] = 0.0
    anomalies["exposure_earned_p10_to_p90"] = ""
    anomalies["exposure_redeemed_p10_to_p90"] = ""
    anomalies["exposure_net_p10_to_p90"] = ""

    selected_ids = anomalies["dealer_id"].tolist()
    dealer_weekly_sel = dealer_weekly[dealer_weekly["dealer_id"].isin(selected_ids)].copy()

    # Compute exposure percentiles using weekly distribution, including zero-activity weeks.
    for i, row in anomalies.iterrows():
        did = int(row["dealer_id"])
        w = dealer_weekly_sel[dealer_weekly_sel["dealer_id"] == did][["week_start", "points_earned_week", "points_redeemed_week"]]

        full_grid = pd.DataFrame({"week_start": week_starts})
        w = full_grid.merge(w, on="week_start", how="left").fillna(0.0)

        realized_earned = float(w["points_earned_week"].sum())
        realized_redeemed = float(w["points_redeemed_week"].sum())
        realized_net = realized_earned - realized_redeemed

        p10_earned, p90_earned = np.percentile(w["points_earned_week"], [10, 90])
        p10_redeemed, p90_redeemed = np.percentile(w["points_redeemed_week"], [10, 90])
        p10_net, p90_net = np.percentile((w["points_earned_week"] - w["points_redeemed_week"]), [10, 90])

        # Scale weekly percentiles to the total recent window length.
        earned_range = (float(p10_earned * n_weeks), float(p90_earned * n_weeks))
        redeemed_range = (float(p10_redeemed * n_weeks), float(p90_redeemed * n_weeks))
        net_range = (float(p10_net * n_weeks), float(p90_net * n_weeks))

        anomalies.at[i, "realized_points_earned_recent"] = realized_earned
        anomalies.at[i, "realized_points_redeemed_recent"] = realized_redeemed
        anomalies.at[i, "exposure_earned_p10_to_p90"] = f"[{earned_range[0]:,.0f}, {earned_range[1]:,.0f}]"
        anomalies.at[i, "exposure_redeemed_p10_to_p90"] = f"[{redeemed_range[0]:,.0f}, {redeemed_range[1]:,.0f}]"
        anomalies.at[i, "exposure_net_p10_to_p90"] = f"[{net_range[0]:,.0f}, {net_range[1]:,.0f}]"

    # Order by anomaly_score desc
    anomalies = anomalies.sort_values(["region", "anomaly_score"], ascending=[True, False])

    keep_cols = [
        "dealer_id",
        "dealer_name",
        "region",
        "score_txn_count",
        "points_earned_score",
        "points_redeemed_score",
        "redemption_rate_score",
        "earned_to_redeemed_ratio_score",
        "z_earn",
        "z_redeem_rate",
        "anomaly_score",
        "realized_points_earned_recent",
        "realized_points_redeemed_recent",
        "exposure_earned_p10_to_p90",
        "exposure_redeemed_p10_to_p90",
        "exposure_net_p10_to_p90",
    ]
    return anomalies[keep_cols]


def main() -> None:
    with get_connection() as conn:
        summary = pd.read_sql_query(
            """
            SELECT
              COUNT(*) AS transactions,
              COUNT(DISTINCT member_id) AS members,
              COUNT(DISTINCT dealer_id) AS dealers
            FROM transactions
            """,
            conn,
        )
        print(summary.to_string(index=False))

        print("\n[step] building leakage-free member feature table (scoring window only)...")
        member_features = build_member_features(conn)

        print("[step] scoring churn-risk with a transparent formula and validating held-out inactivity...")
        scored_members, bucket_summary = score_and_validate_churn(conn, member_features)
        print("\nHeld-out inactivity by risk bucket (higher risk => higher inactivity):")
        # Round for readability
        tmp = bucket_summary.copy()
        tmp["inactive_rate"] = (tmp["inactive_rate"] * 100).round(2)
        tmp["avg_heldout_txns"] = tmp["avg_heldout_txns"].round(2)
        print(tmp.to_string(index=False))

        print("\n[step] detecting anomalous dealers within region peer groups...")
        anomalies = detect_dealer_anomalies(conn)
        # Finance-facing preview
        preview_cols = [
            "dealer_id",
            "dealer_name",
            "region",
            "points_earned_score",
            "redemption_rate_score",
            "anomaly_score",
            "exposure_earned_p10_to_p90",
            "realized_points_earned_recent",
        ]
        print("\nAnomalous dealers (earned high vs peers, redeemed low vs peers):")
        print(anomalies[preview_cols].head(30).to_string(index=False))

        # Concise finance-facing conclusion tied to computed numbers
        # Use bucket extremes for headline.
        if len(bucket_summary) > 0:
            low = bucket_summary.sort_values("risk_bucket").iloc[0]
            high = bucket_summary.sort_values("risk_bucket").iloc[-1]
            concl = (
                "\nCONCLUSION (finance-facing):\n"
                f"- The transparent churn-risk model (no ML, no outcome leakage) shows a clear held-out trend:\n"
                f"  risk bucket {int(low['risk_bucket'])} inactivity = {float(low['inactive_rate'])*100:.1f}% vs\n"
                f"  risk bucket {int(high['risk_bucket'])} inactivity = {float(high['inactive_rate'])*100:.1f}% (Spearman={float(bucket_summary['spearman_with_bucket_order'].iloc[0]):.3f}).\n"
                "- The dealer peer-group anomaly scan flags a small set of dealers whose behavior is\n"
                "  inconsistent with their regional points ecology: high earned points concentration coupled\n"
                "  with unusually low redemption intensity (proxy for potential exposure / loyalty leakage risk).\n"
                f"- These implicated dealers have recent earned-points exposure ranges (P10–P90 weekly scaling) such as\n"
                f"  {anomalies[['dealer_name','exposure_earned_p10_to_p90','realized_points_earned_recent']].head(3).to_dict(orient='records')}.\n"
            )
        else:
            concl = "\nCONCLUSION: unable to compute bucket summary."

        print(concl)


if __name__ == "__main__":
    main()
