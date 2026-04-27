"""
Stage 4: Model Training (Rolling Window)
Trains on recent races to avoid distribution shift.
Uses rolling window across season boundaries for best recency.
"""
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

FEATURES_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "lgbm_model.pkl")

FEATURE_COLS = ["recent_form_pts", "position_std", "qual_position", "dnf_rate", "team_momentum"]
TARGET = "fantasy_pts"
BUDGET = 100.0

def train_model(feat_df: pd.DataFrame, lookback_races: int = 30):
    """Train on recent data using a rolling window approach.

    Uses the most recent N races across season boundaries, prioritizing
    the latest season's data to capture current form.

    Args:
        feat_df: Feature DataFrame with all seasons.
        lookback_races: Number of most recent races to train on (default 30 ~1.5 seasons).
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Sort chronologically and get all unique year-round combos
    feat_df = feat_df.sort_values(["year", "round"]).copy()
    race_keys = feat_df.groupby(["year", "round"]).size().reset_index()[["year", "round"]]
    race_keys = race_keys.sort_values(["year", "round"]).reset_index(drop=True)

    # Take the most recent N races
    recent_race_keys = race_keys.tail(lookback_races)
    recent_pairs = set(zip(recent_race_keys["year"], recent_race_keys["round"]))

    df_recent = feat_df[
        feat_df.apply(lambda r: (r["year"], r["round"]) in recent_pairs, axis=1)
    ].copy()

    if df_recent.empty:
        print("[model] No recent data found, using all data")
        df_recent = feat_df.copy()

    df_recent = df_recent.dropna(subset=FEATURE_COLS + [TARGET])

    # Determine year range for logging
    years = sorted(df_recent["year"].unique())
    total_rounds = df_recent.groupby(["year", "round"]).ngroups

    X = df_recent[FEATURE_COLS]
    y = df_recent[TARGET]

    print(f"[model] Training on {len(X)} records from {total_rounds} races "
          f"(years: {years})")

    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=15,
        min_child_samples=5,
        random_state=42,
        verbose=-1
    )
    model.fit(X, y, callbacks=[lgb.log_evaluation(-1)])

    # Save
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    test_mae = mean_absolute_error(y, model.predict(X))
    print(f"[model] Train MAE: {test_mae:.3f}")
    print(f"[model] Saved to {MODEL_PATH}")

    return model, df_recent, df_recent

def naive_team(race_df):
    """5 most expensive drivers within $100M budget."""
    race_df = race_df.dropna(subset=["price_m"]).copy()
    race_df = race_df.sort_values("price_m", ascending=False)
    selected, total = [], 0.0
    for _, row in race_df.iterrows():
        if total + row["price_m"] <= BUDGET and len(selected) < 5:
            selected.append(row)
            total += row["price_m"]
        if len(selected) == 5:
            break
    return pd.DataFrame(selected) if selected else pd.DataFrame()

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    feat_df = pd.read_csv(FEATURES_CSV)
    train_model(feat_df)
