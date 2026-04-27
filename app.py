"""
GridEdge — F1 Fantasy Team Optimizer
Modular Flask application (2026 Season).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd
from flask import Flask, jsonify, render_template, request

# Add src to path for optimizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from optimizer import optimize

# Import constants and utilities
from constants import (
    RACE_CALENDAR_2026, FEATURE_COLS, BUDGET, 
    CIRCUIT_TYPES, TEAM_COLORS, FEATURE_LABELS,
    DRIVER_IMAGES, DRIVER_NUMBERS
)
from utils import (
    get_model, get_features, get_prices, 
    get_driver_features, build_rationale
)

app = Flask(__name__)

@app.route("/")
def index():
    """Render the main dashboard."""
    return render_template("index.html", races=RACE_CALENDAR_2026)

@app.route("/api/optimize")
def api_optimize():
    """API endpoint to run the team optimizer."""
    race_name = request.args.get("race", "Miami Grand Prix")
    try:
        model   = get_model()
        feat_df = get_features()
        prices  = get_prices()
        importance = dict(zip(FEATURE_COLS, model.feature_importances_))

        driver_rows = []
        for _, row in prices.iterrows():
            feats = get_driver_features(row["driver_name"], feat_df, race_name)
            X     = pd.DataFrame([feats])[FEATURE_COLS]
            pred  = float(model.predict(X)[0])
            price = float(row["price_m"])
            name  = row["driver_name"]
            driver_rows.append({
                "name": name,
                "team": row["team"],
                "price": price,
                "predicted_pts": round(max(0.0, pred), 1),
                "value_score": round(max(0.0, pred) / price, 3) if price > 0 else 0,
                "team_color": TEAM_COLORS.get(row["team"], "#888"),
                "image_url": DRIVER_IMAGES.get(name, ""),
                "number": DRIVER_NUMBERS.get(name, ""),
                **{k: round(float(v), 2) for k, v in feats.items()},
            })

        pred_df = pd.DataFrame(driver_rows).rename(columns={"name": "FullName", "price": "price_m"})
        team_df = optimize(pred_df)
        selected_names = set(team_df["FullName"].tolist())

        for d in driver_rows:
            d["selected"] = d["name"] in selected_names
            if d["selected"]:
                feats = {f: d.get(f, 0) for f in FEATURE_COLS}
                d["rationale"] = build_rationale(feats, importance)

        total_cost = round(team_df["price_m"].sum(), 1)
        total_pts  = round(team_df["predicted_pts"].sum(), 1)
        
        # Normalize feature importances to 0-1 range
        imp_values = list(importance.values())
        imp_sum = sum(imp_values) if sum(imp_values) > 0 else 1.0
        normalized_imp = {k: v / imp_sum for k, v in importance.items()}
        
        feat_imp   = [
            {"feature": FEATURE_LABELS.get(f, f), "importance": round(float(normalized_imp[f]), 4)}
            for f in sorted(normalized_imp.keys(), key=lambda x: -normalized_imp[x])
        ]

        return jsonify({
            "drivers": driver_rows,
            "total_cost": total_cost,
            "total_pts": total_pts,
            "budget_remaining": round(BUDGET - total_cost, 1),
            "circuit_type": CIRCUIT_TYPES.get(race_name, "technical"),
            "race_name": race_name,
            "feature_importance": feat_imp,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("GridEdge — http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)