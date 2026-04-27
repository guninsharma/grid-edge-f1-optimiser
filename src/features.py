"""
Stage 3: Feature Engineering
Computes 5 core features per driver×race for speed and robustness.
Designed for recent-data-only training (captures current season form).
Supports 2022-2026 data across season boundaries.
"""
import pandas as pd
import numpy as np
import os

RAW_FEATURES_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")

STREET_CIRCUITS = {
    "Monaco Grand Prix", "Singapore Grand Prix", "Azerbaijan Grand Prix",
    "Saudi Arabian Grand Prix", "Las Vegas Grand Prix", "Miami Grand Prix",
    "Australian Grand Prix"
}
HIGH_SPEED_CIRCUITS = {
    "Italian Grand Prix", "Belgian Grand Prix", "British Grand Prix",
    "Austrian Grand Prix", "Dutch Grand Prix", "Bahrain Grand Prix",
    "Spanish Grand Prix", "Abu Dhabi Grand Prix"
}

def classify_circuit(name: str) -> str:
    if name in STREET_CIRCUITS:
        return "street"
    if name in HIGH_SPEED_CIRCUITS:
        return "high_speed"
    return "technical"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build 5 core features from race data across all seasons."""
    df = df.copy()
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df["qual_position"] = pd.to_numeric(df["qual_position"], errors="coerce")
    df["fantasy_pts"] = pd.to_numeric(df["fantasy_pts"], errors="coerce").fillna(0)
    df["price_m"] = pd.to_numeric(df["price_m"], errors="coerce")
    df["circuit_type"] = df["circuit"].apply(classify_circuit)

    # Sort by driver then chronologically (year, round) to ensure
    # rolling windows work across season boundaries
    df = df.sort_values(["FullName", "year", "round"]).reset_index(drop=True)

    # Feature 1: Recent form (last 3 races average points)
    def recent_form(group):
        forms = []
        for i in range(len(group)):
            past = group.iloc[max(0, i-3):i]
            avg = past["fantasy_pts"].mean() if len(past) > 0 else 5.0
            forms.append(avg)
        return pd.Series(forms, index=group.index)

    df["recent_form_pts"] = df.groupby("FullName", group_keys=False).apply(recent_form)

    # Feature 2: Position consistency (std of last 5 finishes)
    def position_consistency(group):
        consistency = []
        for i in range(len(group)):
            past = group.iloc[max(0, i-5):i]
            std = past["Position"].std() if len(past) > 1 else 5.0
            consistency.append(std)
        return pd.Series(consistency, index=group.index)

    df["position_std"] = df.groupby("FullName", group_keys=False).apply(position_consistency)

    # Feature 3: Qualifying skill (qual_position, lower is better)
    df["qual_position"] = df["qual_position"].fillna(15.0)

    # Feature 4: DNF rate in last 10 races
    def dnf_rate(group):
        rates = []
        for i in range(len(group)):
            past = group.iloc[max(0, i-10):i]
            if len(past) == 0:
                rates.append(0)
            else:
                dnfs = (past["fantasy_pts"] == 0).sum()
                rates.append(dnfs / len(past))
        return pd.Series(rates, index=group.index)

    df["dnf_rate"] = df.groupby("FullName", group_keys=False).apply(dnf_rate)

    # Feature 5: Team momentum (team's recent form)
    def team_momentum(group):
        momentum = []
        for i in range(len(group)):
            past = group.iloc[max(0, i-3):i]
            avg = past["fantasy_pts"].mean() if len(past) > 0 else 5.0
            momentum.append(avg)
        return pd.Series(momentum, index=group.index)

    df["team_momentum"] = df.groupby("TeamName", group_keys=False).apply(team_momentum)

    # Select final features
    feature_cols = [
        "FullName", "year", "round", "circuit", "circuit_type",
        "Position", "fantasy_pts", "price_m", "TeamName",
        "recent_form_pts", "position_std", "qual_position", "dnf_rate", "team_momentum"
    ]

    feat_df = df[[c for c in feature_cols if c in df.columns]].copy()
    feat_df.to_csv(RAW_FEATURES_CSV, index=False)
    print(f"[features] Saved {len(feat_df)} rows with 5 core features "
          f"(years: {sorted(feat_df['year'].unique())})")
    return feat_df

if __name__ == "__main__":
    merged = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "data", "merged_data.csv")
    )
    feat = build_features(merged)
    print(feat[["FullName", "recent_form_pts", "dnf_rate", "qual_position"]].head(10))
