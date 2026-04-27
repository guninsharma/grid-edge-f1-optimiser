import os
import pickle
import pandas as pd
from constants import MODEL_PATH, FEAT_CSV, PRICES_CSV, FEATURE_COLS

_model = None
_feat_df = None
_prices = None

def get_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model

def get_features():
    global _feat_df
    if _feat_df is None and os.path.exists(FEAT_CSV):
        _feat_df = pd.read_csv(FEAT_CSV)
    return _feat_df

def get_prices():
    global _prices
    if _prices is None:
        df = pd.read_csv(PRICES_CSV)
        df["price_m"] = df["price_m"].astype(float)
        _prices = df
    return _prices

def get_driver_features(driver_name, feat_df, race_name=None):
    if feat_df is None: 
        return {f: 5.0 for f in FEATURE_COLS}
    
    drv = feat_df[feat_df["FullName"] == driver_name].sort_values(["year", "round"])
    if drv.empty:
        return {"recent_form_pts": 5.0, "position_std": 5.0, "qual_position": 12.0, "dnf_rate": 0.2, "team_momentum": 5.0}
    
    # Filter by circuit/race name for race-specific features
    if race_name:
        race_rows = drv[drv.get("circuit", "") == race_name]
        if not race_rows.empty:
            drv = race_rows
    
    last = drv.iloc[-1]
    return {f: float(last.get(f, 5.0)) for f in FEATURE_COLS}

def build_rationale(features, importance):
    top_feat = sorted(importance.items(), key=lambda x: -x[1])[:2]
    blurbs = []
    for feat, _ in top_feat:
        val = features.get(feat, 0)
        if feat == "recent_form_pts": 
            blurbs.append(f"{val:.1f} pts avg")
        elif feat == "position_std": 
            blurbs.append("consistent" if val < 3 else f"±{val:.1f} variance")
        elif feat == "qual_position": 
            blurbs.append(f"P{int(val)} quali")
        elif feat == "dnf_rate": 
            blurbs.append("reliable" if val < 0.1 else f"{val*100:.0f}% DNF risk")
        elif feat == "team_momentum": 
            blurbs.append(f"{val:.1f} team pts")
    return " · ".join(blurbs) if blurbs else "Strong predicted performance"
