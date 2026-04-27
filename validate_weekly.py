#!/usr/bin/env python3
"""
Performance Tracker: Validates weekly retraining approach on historical races.
Tests: Does recent-only training + ensemble beat naive baseline?
Supports dynamic year detection — works across 2024, 2025, 2026 data.
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

BASE = os.path.dirname(__file__)
FEATURES_CSV = os.path.join(BASE, "data", "features.csv")
PRICES_CSV = os.path.join(BASE, "data", "prices.csv")

FEATURE_COLS = ["recent_form_pts", "position_std", "qual_position", "dnf_rate", "team_momentum"]
TARGET = "fantasy_pts"
BUDGET = 100.0

def naive_team(race_df):
    """5 most expensive drivers within budget."""
    race_df = race_df.dropna(subset=["price_m"])
    race_df = race_df.sort_values("price_m", ascending=False)
    selected, total = [], 0.0
    for _, row in race_df.iterrows():
        if total + row["price_m"] <= BUDGET and len(selected) < 5:
            selected.append(row)
            total += row["price_m"]
        if len(selected) == 5:
            break
    return pd.DataFrame(selected) if selected else pd.DataFrame()

def get_driver_features(driver_name, feat_df):
    """Get latest features for a driver."""
    drv = feat_df[feat_df["FullName"] == driver_name].sort_values(["year", "round"])
    if drv.empty:
        return {col: 0.0 for col in FEATURE_COLS}
    last = drv.iloc[-1]
    return {col: float(last.get(col, 0.0)) for col in FEATURE_COLS}

def train_model_recent_window(feat_df, test_year, test_round, lookback=6):
    """Train on races BEFORE test_round (lookback window across seasons)."""
    # Get all races before this test race (chronological)
    train_df = feat_df[
        (feat_df["year"] < test_year) |
        ((feat_df["year"] == test_year) & (feat_df["round"] < test_round))
    ].copy()

    # Get unique race keys and keep only recent ones
    race_keys = train_df.groupby(["year", "round"]).size().reset_index()[["year", "round"]]
    race_keys = race_keys.sort_values(["year", "round"]).reset_index(drop=True)
    recent_keys = race_keys.tail(lookback)
    recent_pairs = set(zip(recent_keys["year"], recent_keys["round"]))

    train_df = train_df[
        train_df.apply(lambda r: (r["year"], r["round"]) in recent_pairs, axis=1)
    ].copy()
    train_df = train_df.dropna(subset=FEATURE_COLS + [TARGET])

    if len(train_df) < 10:
        return None, f"Insufficient data ({len(train_df)} rows)"

    X = train_df[FEATURE_COLS]
    y = train_df[TARGET]

    model = lgb.LGBMRegressor(
        n_estimators=150,
        learning_rate=0.1,
        num_leaves=15,
        min_child_samples=3,
        random_state=42,
        verbose=-1
    )
    model.fit(X, y, callbacks=[lgb.log_evaluation(-1)])

    return model, None

def predict_and_ensemble(model, race_df, prices_df):
    """Generate ensemble predictions for a race."""
    preds = []

    for driver_name in race_df["FullName"].unique():
        driver_row = race_df[race_df["FullName"] == driver_name].iloc[0]
        price_row = prices_df[prices_df["driver_name"] == driver_name]
        if price_row.empty:
            continue

        price = float(price_row["price_m"].values[0])

        # Model prediction (40%)
        features = get_driver_features(driver_name, race_df)
        X = pd.DataFrame([{col: features[col] for col in FEATURE_COLS}])
        model_pred = float(model.predict(X)[0])

        # Baseline: recent form (30%)
        baseline_pred = float(driver_row.get("recent_form_pts", 5.0))

        # Efficiency (30%)
        efficiency_pred = (baseline_pred / price * 10) if price > 0 else 0

        # Ensemble
        ensemble_pred = model_pred * 0.4 + baseline_pred * 0.3 + efficiency_pred * 0.3

        preds.append({
            "driver": driver_name,
            "price": price,
            "actual_pts": float(driver_row.get("fantasy_pts", 0)),
            "ensemble_pred": ensemble_pred
        })

    return pd.DataFrame(preds)

def validate_on_recent_races():
    """Backtest recent-only approach on the most recent season's last 8 races."""
    feat_df = pd.read_csv(FEATURES_CSV)
    prices_df = pd.read_csv(PRICES_CSV)

    # Dynamically detect the latest full season with enough data
    available_years = sorted(feat_df["year"].unique())
    print(f"Available years in data: {available_years}")

    # Find the latest year with at least 8 races for meaningful validation
    target_year = None
    for year in reversed(available_years):
        n_rounds = feat_df[feat_df["year"] == year]["round"].nunique()
        if n_rounds >= 10:
            target_year = year
            break

    if target_year is None:
        # Fall back to the latest year with any data
        target_year = available_years[-1]
        print(f"No year with 10+ races, using latest: {target_year}")

    feat_year = feat_df[feat_df["year"] == target_year].copy()
    all_rounds = sorted(feat_year["round"].unique())
    test_rounds = all_rounds[-8:] if len(all_rounds) > 8 else all_rounds

    print("=" * 80)
    print("  GridEdge Weekly Retraining Validation")
    print("=" * 80)
    print(f"\nValidation year: {target_year}")
    print(f"Testing on {len(test_rounds)} races: R{min(test_rounds)}-R{max(test_rounds)}")
    print(f"Using lookback window of 6 races for training\n")

    results = []

    for test_round in test_rounds:
        race_data = feat_year[feat_year["round"] == test_round]
        if len(race_data) < 5:
            continue

        # Train on recent races before this race (across all seasons)
        model, err = train_model_recent_window(feat_df, target_year, test_round, lookback=6)
        if model is None:
            print(f"  R{test_round:2d}: SKIP ({err})")
            continue

        # Predict
        preds = predict_and_ensemble(model, race_data, prices_df)
        if preds.empty:
            continue

        # Get model team
        preds_sorted = preds.sort_values("ensemble_pred", ascending=False)
        model_team, total = [], 0.0
        for _, row in preds_sorted.iterrows():
            if total + row["price"] <= BUDGET and len(model_team) < 5:
                model_team.append(row)
                total += row["price"]
            if len(model_team) == 5:
                break

        if not model_team:
            continue

        model_team_df = pd.DataFrame(model_team)
        model_score = model_team_df["actual_pts"].sum()

        # Get naive team
        naive = naive_team(race_data)
        naive_score = naive["fantasy_pts"].sum() if not naive.empty else 0

        delta = model_score - naive_score
        win = "WIN" if delta > 0 else "LOSS"

        results.append({
            "round": test_round,
            "model_score": model_score,
            "naive_score": naive_score,
            "delta": delta,
            "win": delta > 0
        })

        print(f"  R{test_round:2d}: Model {model_score:>5.0f} | Baseline {naive_score:>5.0f} | {delta:>+6.1f} pts | {win}")

    if not results:
        print("\nNo results to evaluate")
        return

    results_df = pd.DataFrame(results)
    win_rate = (results_df["win"].sum() / len(results_df) * 100) if len(results_df) > 0 else 0
    avg_delta = results_df["delta"].mean()

    print("\n" + "=" * 80)
    print(f"  SUMMARY: {len(results)} races tested (year {target_year})")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Avg Improvement: {avg_delta:+.1f} pts/race")
    print(f"  Total Delta: {results_df['delta'].sum():+.1f} pts")
    print("=" * 80)

    if win_rate >= 50:
        print("\n✓ PASS: Recent-only approach beats baseline")
    else:
        print("\n✗ FAIL: Recent-only approach underperforms baseline")

    # Save results to log
    log_path = os.path.join(BASE, "logs", "validation_results.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    results_df.to_csv(log_path, index=False)
    print(f"\nResults saved to {log_path}")

    return results_df

if __name__ == "__main__":
    validate_on_recent_races()
