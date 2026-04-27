#!/usr/bin/env python3
"""
Weekly Retraining System for GridEdge
Automatically fetches latest race data, rebuilds features, retrains model,
and generates predictions for the next race.

Designed for scheduled execution (e.g., every Tuesday after race weekends).

Usage:
  python weekly_retrain.py                  # Full retrain with auto-fetch
  python weekly_retrain.py --skip-fetch     # Retrain on existing data only
  python weekly_retrain.py --lookback 12    # Custom lookback window
"""
import os
import sys
import pickle
import argparse
import warnings
import logging
from datetime import datetime
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

BASE = os.path.dirname(__file__)
FEATURES_CSV = os.path.join(BASE, "data", "features.csv")
PRICES_CSV   = os.path.join(BASE, "data", "prices.csv")
MODEL_PATH   = os.path.join(BASE, "models", "lgbm_model.pkl")
LOG_DIR      = os.path.join(BASE, "logs")
LOG_FILE     = os.path.join(LOG_DIR, "weekly_retrain.log")

FEATURE_COLS = ["recent_form_pts", "position_std", "qual_position", "dnf_rate", "team_momentum"]
TARGET = "fantasy_pts"
BUDGET = 100.0

# ── Logging ──────────────────────────────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("weekly_retrain")


def fetch_latest_data():
    """Fetch any new race data from FastF1 (incremental)."""
    from ingest import fetch_race_data, merge_prices

    logger.info("Fetching latest race data from FastF1...")
    race_df = fetch_race_data(use_cache=True, force_refresh=False)

    logger.info("Merging with prices.csv...")
    prices_df = pd.read_csv(PRICES_CSV)
    merged = merge_prices(race_df, prices_df)

    logger.info("Rebuilding features...")
    from features import build_features
    feat_df = build_features(merged)

    return feat_df


def train_recent_only(feat_df, lookback_races=12):
    """
    Train model on the most recent N races across all seasons.
    Uses a rolling window approach to capture current form while
    avoiding distribution shift from old data.
    """
    # Sort chronologically
    feat_df = feat_df.sort_values(["year", "round"]).copy()

    # Get all unique race identifiers
    race_keys = feat_df.groupby(["year", "round"]).size().reset_index()[["year", "round"]]
    race_keys = race_keys.sort_values(["year", "round"]).reset_index(drop=True)

    # Take the most recent N races
    recent_keys = race_keys.tail(lookback_races)
    recent_pairs = set(zip(recent_keys["year"], recent_keys["round"]))

    df_recent = feat_df[
        feat_df.apply(lambda r: (r["year"], r["round"]) in recent_pairs, axis=1)
    ].copy()

    if df_recent.empty:
        logger.warning("No recent data found, using all data")
        df_recent = feat_df.copy()

    # Log training data summary
    years = sorted(df_recent["year"].unique())
    n_races = df_recent.groupby(["year", "round"]).ngroups
    logger.info(f"Training on {len(df_recent)} records from {n_races} races "
                f"(years: {years}, lookback: {lookback_races})")

    # Drop NAs
    df_recent = df_recent.dropna(subset=FEATURE_COLS + [TARGET])

    if len(df_recent) < 20:
        logger.warning(f"Only {len(df_recent)} records, may be underfitting")

    X = df_recent[FEATURE_COLS]
    y = df_recent[TARGET]

    # Train LightGBM
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
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Eval
    train_mae = mean_absolute_error(y, model.predict(X))
    logger.info(f"Model trained. Train MAE: {train_mae:.3f}")

    return model


def get_driver_features(driver_name, team_name, circuit_type, feat_df):
    """Extract latest features for a driver from historical data."""
    drv = feat_df[feat_df["FullName"] == driver_name]
    team = feat_df[feat_df["TeamName"] == team_name]

    if drv.empty:
        return {
            "recent_form_pts": 5.0,
            "position_std": 5.0,
            "qual_position": 12.0,
            "dnf_rate": 0.2,
            "team_momentum": 5.0
        }

    last_driver = drv.iloc[-1]
    last_team = team.iloc[-1] if not team.empty else last_driver

    return {
        "recent_form_pts": float(last_driver.get("recent_form_pts", 5.0)),
        "position_std": float(last_driver.get("position_std", 5.0)),
        "qual_position": float(last_driver.get("qual_position", 12.0)),
        "dnf_rate": float(last_driver.get("dnf_rate", 0.2)),
        "team_momentum": float(last_team.get("team_momentum", 5.0))
    }


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_next_race(model, race_drivers, feat_df, prices_df):
    """Generate predictions + ensemble for next race."""
    predictions = []

    for driver_name in race_drivers:
        driver_row = prices_df[prices_df["driver_name"] == driver_name]
        if driver_row.empty:
            continue

        team = driver_row["team"].values[0]
        price = driver_row["price_m"].values[0]

        # Get features
        features = get_driver_features(driver_name, team, "technical", feat_df)
        X = pd.DataFrame([features])[FEATURE_COLS]

        # Model prediction (40% weight)
        model_pred = float(model.predict(X)[0])

        # Baseline: recent form (30% weight)
        baseline_pred = features["recent_form_pts"]

        # Price efficiency (30% weight)
        efficiency_pred = (features["recent_form_pts"] / price) * 10 if price > 0 else 0

        # Ensemble
        ensemble_pred = (model_pred * 0.4) + (baseline_pred * 0.3) + (efficiency_pred * 0.3)

        predictions.append({
            "driver_name": driver_name,
            "team": team,
            "price_m": price,
            "model_pred": model_pred,
            "ensemble_pred": ensemble_pred,
            "recent_form": baseline_pred
        })

    return pd.DataFrame(predictions)


def select_team(pred_df, budget=BUDGET):
    """Optimize team selection using ensemble scores."""
    pred_df = pred_df.sort_values("ensemble_pred", ascending=False).copy()

    selected = []
    total_price = 0.0

    for _, row in pred_df.iterrows():
        if total_price + row["price_m"] <= budget and len(selected) < 5:
            selected.append(row)
            total_price += row["price_m"]
        if len(selected) == 5:
            break

    team = pd.DataFrame(selected)
    return team, total_price


def sanity_check(model, feat_df, prices_df):
    """Post-training sanity check: verify model produces reasonable predictions."""
    drivers = prices_df["driver_name"].unique()
    preds = []
    for d in drivers:
        feats = get_driver_features(d, "", "technical", feat_df)
        X = pd.DataFrame([feats])[FEATURE_COLS]
        pred = float(model.predict(X)[0])
        preds.append(pred)

    preds = np.array(preds)
    logger.info(f"Sanity check: {len(preds)} drivers, "
                f"pred range [{preds.min():.1f}, {preds.max():.1f}], "
                f"mean={preds.mean():.1f}, std={preds.std():.1f}")

    if preds.max() < 0.1:
        logger.error("FAIL: All predictions near zero — model may be broken")
        return False
    if preds.std() < 0.01:
        logger.error("FAIL: No variance in predictions — model may be degenerate")
        return False

    logger.info("Sanity check PASSED ✓")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridEdge Weekly Retraining")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip fetching new data, retrain on existing features.csv")
    parser.add_argument("--lookback", type=int, default=12,
                        help="Number of recent races to train on (default: 12)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("  GridEdge Weekly Retraining System")
    logger.info(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    # Stage 0: Fetch latest data
    if not args.skip_fetch:
        logger.info("\n[Stage 0] Fetching latest race data...")
        try:
            feat_df = fetch_latest_data()
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            logger.info("Falling back to existing features.csv")
            feat_df = pd.read_csv(FEATURES_CSV)
    else:
        logger.info("\n[Stage 0] Skipping fetch, loading existing data...")
        feat_df = pd.read_csv(FEATURES_CSV)

    prices_df = pd.read_csv(PRICES_CSV)

    # Log data summary
    years = sorted(feat_df["year"].unique())
    logger.info(f"Data loaded: {len(feat_df)} records, years: {years}")

    # Stage 1: Retrain on recent races
    logger.info(f"\n[Stage 1] Training on recent races (lookback={args.lookback})...")
    model = train_recent_only(feat_df, lookback_races=args.lookback)

    # Stage 1.5: Sanity check
    logger.info("\n[Stage 1.5] Running sanity check...")
    if not sanity_check(model, feat_df, prices_df):
        logger.error("Sanity check failed! Model may be unreliable.")

    # Stage 2: Get next race drivers
    next_race_drivers = prices_df["driver_name"].unique()

    # Predict
    logger.info(f"\n[Stage 2] Predicting for {len(next_race_drivers)} drivers...")
    preds = predict_next_race(model, next_race_drivers, feat_df, prices_df)

    # Stage 3: Select team
    logger.info(f"\n[Stage 3] Selecting optimal team...")
    team, total_price = select_team(preds, budget=BUDGET)

    logger.info(f"\nRecommended Team (${total_price:.1f}M budget):")
    logger.info("=" * 70)
    for _, row in team.iterrows():
        logger.info(f"  {row['driver_name']:<20} ${row['price_m']:>6.1f}M  "
                     f"Pred: {row['ensemble_pred']:>5.1f} pts")
    logger.info("=" * 70)
    logger.info(f"Total: ${total_price:.1f}M | "
                f"Predicted: {team['ensemble_pred'].sum():.1f} pts")
    logger.info("=" * 70)
    logger.info(f"Retrain complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
