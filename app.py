"""
GridEdge — F1 Fantasy Team Optimizer
Production-grade Flask + HTML frontend (2026 Season).
"""
import os, sys, pickle, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template_string, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from optimizer import optimize

BASE       = os.path.dirname(__file__)
PRICES_CSV = os.path.join(BASE, "data", "prices.csv")
FEAT_CSV   = os.path.join(BASE, "data", "features.csv")
MODEL_PATH = os.path.join(BASE, "models", "lgbm_model.pkl")

FEATURE_COLS = ["recent_form_pts", "position_std", "qual_position", "dnf_rate", "team_momentum"]
BUDGET       = 100.0

# Official F1 CDN image URL pattern discovered from formula1.com
# Format: https://media.formula1.com/image/upload/{transforms}/v1740000001/common/f1/2026/{team}/{drivercode}/{year}{team}{drivercode}right.webp
F1_CDN = "https://media.formula1.com/image/upload/c_lfill,w_320/q_auto/d_common:f1:2026:fallback:driver:2026fallbackdriverright.webp/v1740000001/common/f1/2026"

DRIVER_IMAGES = {
    "Alexander Albon":      f"{F1_CDN}/williams/alealb01/2026williamsalealb01right.webp",
    "Fernando Alonso":      f"{F1_CDN}/astonmartin/feralo01/2026astonmartinferalo01right.webp",
    "Kimi Antonelli":       f"{F1_CDN}/mercedes/andant01/2026mercedesandant01right.webp",
    "Oliver Bearman":       f"{F1_CDN}/haasf1team/olibea01/2026haasf1teamolibea01right.webp",
    "Gabriel Bortoleto":    f"{F1_CDN}/audi/gabbor01/2026audigabbor01right.webp",
    "Valtteri Bottas":      f"{F1_CDN}/cadillac/valbot01/2026cadillacvalbot01right.webp",
    "Franco Colapinto":     f"{F1_CDN}/alpine/fracol01/2026alpinefracol01right.webp",
    "Pierre Gasly":         f"{F1_CDN}/alpine/piegas01/2026alpinepiegas01right.webp",
    "Isack Hadjar":         f"{F1_CDN}/redbull/isahad01/2026redbullisahad01right.webp",
    "Lewis Hamilton":       f"{F1_CDN}/ferrari/lewham01/2026ferrarilewham01right.webp",
    "Nico Hulkenberg":      f"{F1_CDN}/audi/nichul01/2026audinichul01right.webp",
    "Liam Lawson":          f"{F1_CDN}/racingbulls/lialaw01/2026racingbullslialaw01right.webp",
    "Charles Leclerc":      f"{F1_CDN}/ferrari/chalec01/2026ferrarichalec01right.webp",
    "Arvid Lindblad":       f"{F1_CDN}/racingbulls/arvlin01/2026racingbullsarvlin01right.webp",
    "Lando Norris":         f"{F1_CDN}/mclaren/lannor01/2026mclarenlannor01right.webp",
    "Esteban Ocon":         f"{F1_CDN}/haasf1team/esteoc01/2026haasf1teamesteoc01right.webp",
    "Sergio Perez":         f"{F1_CDN}/cadillac/serper01/2026cadillacserper01right.webp",
    "Oscar Piastri":        f"{F1_CDN}/mclaren/oscpia01/2026mclarenoscpia01right.webp",
    "George Russell":       f"{F1_CDN}/mercedes/georus01/2026mercedesgeorust01right.webp",
    "Carlos Sainz":         f"{F1_CDN}/williams/carsai01/2026williamscarsai01right.webp",
    "Lance Stroll":         f"{F1_CDN}/astonmartin/lanstr01/2026astonmartinlanstr01right.webp",
    "Max Verstappen":       f"{F1_CDN}/redbull/maxver01/2026redbullmaxver01right.webp",
}

DRIVER_NUMBERS = {
    "Max Verstappen": 1, "Lando Norris": 4, "Charles Leclerc": 16,
    "Oscar Piastri": 81, "Lewis Hamilton": 44, "George Russell": 63,
    "Carlos Sainz": 55, "Kimi Antonelli": 12, "Fernando Alonso": 14,
    "Alexander Albon": 23, "Pierre Gasly": 10, "Liam Lawson": 30,
    "Isack Hadjar": 6, "Nico Hulkenberg": 27, "Esteban Ocon": 31,
    "Oliver Bearman": 87, "Franco Colapinto": 43, "Lance Stroll": 18,
    "Gabriel Bortoleto": 5, "Arvid Lindblad": 7, "Sergio Perez": 11,
    "Valtteri Bottas": 77,
}

RACE_CALENDAR_2026 = [
    {"id": "R01", "name": "Australian Grand Prix",    "completed": True},
    {"id": "R02", "name": "Chinese Grand Prix",       "completed": True},
    {"id": "R03", "name": "Japanese Grand Prix",      "completed": True},
    {"id": "R04", "name": "Miami Grand Prix",         "completed": False},
    {"id": "R05", "name": "Canadian Grand Prix",      "completed": False},
    {"id": "R06", "name": "Monaco Grand Prix",        "completed": False},
    {"id": "R07", "name": "Spanish Grand Prix",       "completed": False},
    {"id": "R08", "name": "Austrian Grand Prix",      "completed": False},
    {"id": "R09", "name": "British Grand Prix",       "completed": False},
    {"id": "R10", "name": "Belgian Grand Prix",       "completed": False},
    {"id": "R11", "name": "Hungarian Grand Prix",     "completed": False},
    {"id": "R12", "name": "Dutch Grand Prix",         "completed": False},
    {"id": "R13", "name": "Italian Grand Prix",       "completed": False},
    {"id": "R14", "name": "Madrid Grand Prix",        "completed": False},
    {"id": "R15", "name": "Azerbaijan Grand Prix",    "completed": False},
    {"id": "R16", "name": "Singapore Grand Prix",     "completed": False},
    {"id": "R17", "name": "United States Grand Prix", "completed": False},
    {"id": "R18", "name": "Mexico City Grand Prix",   "completed": False},
    {"id": "R19", "name": "Sao Paulo Grand Prix",     "completed": False},
    {"id": "R20", "name": "Las Vegas Grand Prix",     "completed": False},
    {"id": "R21", "name": "Qatar Grand Prix",         "completed": False},
    {"id": "R22", "name": "Abu Dhabi Grand Prix",     "completed": False},
]

CIRCUIT_TYPES = {
    "Australian Grand Prix": "street", "Chinese Grand Prix": "technical",
    "Japanese Grand Prix": "technical", "Miami Grand Prix": "street",
    "Canadian Grand Prix": "technical", "Monaco Grand Prix": "street",
    "Spanish Grand Prix": "high speed", "Austrian Grand Prix": "high speed",
    "British Grand Prix": "high speed", "Belgian Grand Prix": "high speed",
    "Hungarian Grand Prix": "technical", "Dutch Grand Prix": "technical",
    "Italian Grand Prix": "high speed", "Madrid Grand Prix": "technical",
    "Azerbaijan Grand Prix": "street", "Singapore Grand Prix": "street",
    "United States Grand Prix": "technical", "Mexico City Grand Prix": "technical",
    "Sao Paulo Grand Prix": "technical", "Las Vegas Grand Prix": "street",
    "Qatar Grand Prix": "technical", "Abu Dhabi Grand Prix": "high speed",
}

TEAM_COLORS = {
    "Red Bull": "#3671C6", "McLaren": "#FF8000", "Ferrari": "#E8002D",
    "Mercedes": "#27F4D2", "Williams": "#64C4FF", "Aston Martin": "#229971",
    "Alpine": "#FF87BC", "Racing Bulls": "#6692FF", "Audi": "#D0D0D0",
    "Haas": "#B6BABD", "Cadillac": "#C0C0C0",
}

FEATURE_LABELS = {
    "recent_form_pts": "Recent Form",
    "position_std": "Consistency",
    "qual_position": "Qualifying",
    "dnf_rate": "Reliability",
    "team_momentum": "Team Momentum",
}

app = Flask(__name__)
_model = None; _feat_df = None; _prices = None

def get_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f: _model = pickle.load(f)
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
    if feat_df is None: return {f: 5.0 for f in FEATURE_COLS}
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
        if feat == "recent_form_pts": blurbs.append(f"{val:.1f} pts avg")
        elif feat == "position_std": blurbs.append("consistent" if val < 3 else f"±{val:.1f} variance")
        elif feat == "qual_position": blurbs.append(f"P{int(val)} quali")
        elif feat == "dnf_rate": blurbs.append("reliable" if val < 0.1 else f"{val*100:.0f}% DNF risk")
        elif feat == "team_momentum": blurbs.append(f"{val:.1f} team pts")
    return " · ".join(blurbs) if blurbs else "Strong predicted performance"


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, races=RACE_CALENDAR_2026)

@app.route("/api/optimize")
def api_optimize():
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


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GridEdge — F1 Fantasy Optimizer 2026</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700;900&family=Barlow+Condensed:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
:root {
  --red:#E10600; --red-dim:rgba(225,6,0,0.12); --red-glow:rgba(225,6,0,0.07);
  --black:#080808; --dark:#111; --card:#161616; --card2:#1e1e1e;
  --border:rgba(255,255,255,0.07); --border2:rgba(255,255,255,0.14);
  --text:#F0F0F0; --muted:#666; --muted2:#3a3a3a;
  --font:'Titillium Web',sans-serif; --cond:'Barlow Condensed',sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--black);color:var(--text);font-family:var(--font);font-size:15px;line-height:1.5;overflow-x:hidden}

/* ── NAV ── */
.nav{position:sticky;top:0;z-index:100;background:rgba(8,8,8,0.97);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border-bottom:1px solid var(--border);padding:0 2.5rem;display:flex;align-items:center;gap:1.5rem;height:58px}
.logo{font-family:var(--cond);font-weight:800;font-size:1.45rem;letter-spacing:0.1em;text-transform:uppercase;display:flex;align-items:center;gap:10px;color:var(--text)}
.logo-hex{width:28px;height:28px;background:var(--red);clip-path:polygon(0 25%,50% 0,100% 25%,100% 75%,50% 100%,0 75%);flex-shrink:0}
.nav-sep{width:1px;height:22px;background:var(--border2)}
.nav-label{font-family:var(--cond);font-size:11px;letter-spacing:0.18em;text-transform:uppercase;color:var(--muted)}
.nav-pill{margin-left:auto;font-family:var(--cond);font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;background:var(--red);color:#fff;padding:3px 10px;border-radius:2px}

/* ── HERO with F1 logo watermark ── */
.hero{position:relative;padding:4rem 2.5rem 3rem;border-bottom:1px solid var(--border);overflow:hidden;min-height:280px}
.hero-gridlines{position:absolute;inset:0;background:
  repeating-linear-gradient(90deg,rgba(255,255,255,0.013) 0,transparent 1px,transparent 70px,rgba(255,255,255,0.013) 71px),
  repeating-linear-gradient(0deg,rgba(255,255,255,0.013) 0,transparent 1px,transparent 70px,rgba(255,255,255,0.013) 71px);pointer-events:none}
.hero-f1-logo{
  position:absolute; right:-40px; top:50%; transform:translateY(-50%);
  width:620px; height:auto;
  opacity:0.04;
  filter:grayscale(1) brightness(10);
  pointer-events:none;
  user-select:none;
}
.hero-red-glow{position:absolute;top:-100px;left:-80px;width:600px;height:600px;background:radial-gradient(ellipse,rgba(225,6,0,0.05) 0%,transparent 65%);pointer-events:none}
.eyebrow{font-family:var(--cond);font-size:11px;font-weight:700;letter-spacing:0.22em;text-transform:uppercase;color:var(--red);margin-bottom:0.9rem;display:flex;align-items:center;gap:10px;position:relative}
.eyebrow::before{content:'';width:0px;height:2px;background:var(--red)}
h1{font-family:var(--cond);font-size:clamp(2.8rem,5.5vw,5rem);font-weight:900;line-height:0.9;letter-spacing:-0.01em;text-transform:uppercase;position:relative;margin-top:0.5rem}
h1 em{color:var(--red);font-style:normal}
.hero-desc{font-size:13px;color:var(--muted);letter-spacing:0.03em;margin-top:1.1rem;position:relative}
.hero-stats{display:flex;border:1px solid var(--border);border-radius:3px;overflow:hidden;width:fit-content;margin-top:1.8rem;position:relative}
.hs{padding:0.85rem 1.5rem;border-right:1px solid var(--border);display:flex;flex-direction:column;gap:2px}
.hs:last-child{border-right:none}
.hs-v{font-family:var(--cond);font-size:1.8rem;font-weight:800;line-height:1}
.hs-l{font-size:9px;text-transform:uppercase;letter-spacing:0.18em;color:var(--muted)}

/* ── MAIN WRAP ── */
.wrap{max-width:1440px;margin:0 auto;padding:0 2.5rem 5rem}

/* ── RACE SELECTOR ── */
.race-sec{padding:2.5rem 0 2rem;border-bottom:1px solid var(--border)}
.sec-hdr{font-family:var(--cond);font-size:10px;font-weight:700;letter-spacing:0.22em;text-transform:uppercase;color:var(--red);margin-bottom:1.25rem;display:flex;align-items:center;gap:10px}
.sec-hdr::after{content:'';flex:1;height:1px;background:var(--border)}
.race-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:5px}
.race-btn{background:var(--card);border:1px solid var(--border);color:var(--text);padding:9px 13px;font-family:var(--cond);font-size:13px;font-weight:600;letter-spacing:0.03em;cursor:pointer;border-radius:2px;transition:border-color 0.12s,background 0.12s;text-align:left;display:flex;align-items:center;justify-content:space-between;gap:8px}
.race-btn:hover:not([disabled]){border-color:var(--red);background:var(--red-glow)}
.race-btn.active{border-color:var(--red);background:var(--red-dim);color:#fff}
.race-btn[disabled]{color:var(--muted2);cursor:default}
.race-id{font-size:10px;color:var(--muted);font-family:var(--cond);letter-spacing:0.1em;flex-shrink:0}
.race-btn.active .race-id{color:var(--red)}

/* ── LOADER ── */
.loader{display:none;position:fixed;inset:0;background:rgba(8,8,8,0.9);backdrop-filter:blur(10px);z-index:200;flex-direction:column;align-items:center;justify-content:center;gap:1.5rem}
.loader.on{display:flex}
.loader-logo{font-family:var(--cond);font-weight:800;font-size:1.3rem;letter-spacing:0.1em;text-transform:uppercase;display:flex;align-items:center;gap:10px}
.loader-word{font-family:var(--cond);font-size:11px;letter-spacing:0.22em;color:var(--muted);text-transform:uppercase}
.loader-track{width:200px;height:2px;background:var(--border2);border-radius:1px;overflow:hidden}
.loader-fill{height:100%;width:0;background:var(--red);animation:la 1s ease-in-out infinite}
@keyframes la{0%{width:0;margin-left:0}55%{width:60%;margin-left:15%}100%{width:0;margin-left:100%}}

/* ── RESULTS ── */
.results{padding:2rem 0;display:none}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:2rem}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:3px;padding:1.2rem 1.5rem;position:relative;overflow:hidden}
.kpi::before{content:'';position:absolute;top:0;left:0;width:3px;height:100%;background:var(--red)}
.kpi-l{font-family:var(--cond);font-size:9px;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;color:var(--muted);margin-bottom:0.45rem}
.kpi-v{font-family:var(--cond);font-size:2rem;font-weight:800;line-height:1}
.kpi-s{font-size:11px;color:var(--muted);margin-top:5px}
.ctag{display:inline-flex;align-items:center;font-family:var(--cond);font-size:10px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;padding:2px 8px;border-radius:2px}
.ctag-s{background:rgba(255,135,0,0.12);color:#FF8700;border:1px solid rgba(255,135,0,0.25)}
.ctag-t{background:rgba(55,138,221,0.12);color:#378ADD;border:1px solid rgba(55,138,221,0.25)}
.ctag-h{background:rgba(34,153,113,0.12);color:#1D9E75;border:1px solid rgba(34,153,113,0.25)}

/* ── COLUMNS ── */
.cols{display:grid;grid-template-columns:1fr 400px;gap:18px;align-items:start}
.panel{background:var(--card);border:1px solid var(--border);border-radius:3px;overflow:hidden;margin-bottom:16px}
.pnl-hdr{padding:0.85rem 1.25rem;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.pnl-t{font-family:var(--cond);font-size:12px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase}
.pnl-s{font-size:11px;color:var(--muted)}
.c-tabs{display:flex;gap:3px}
.c-tab{font-family:var(--cond);font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;padding:4px 11px;border-radius:2px;cursor:pointer;background:transparent;border:1px solid var(--border);color:var(--muted);transition:all 0.12s}
.c-tab.on{background:var(--red);border-color:var(--red);color:#fff}
.c-tab:hover:not(.on){border-color:var(--border2);color:var(--text)}
.chart-body{padding:1.25rem}
.chart-wrap{position:relative;width:100%;height:280px}

/* ── DRIVER TEAM CARDS ── */
/* Card grid for the optimal team — large F1-style cards */
.team-grid{display:flex;flex-direction:column;gap:0}

.drv-card{
  position:relative;overflow:hidden;
  border-bottom:1px solid var(--border);
  display:grid;
  grid-template-columns:100px 1fr;
  min-height:110px;
  transition:background 0.12s;
}
.drv-card:last-child{border-bottom:none}
.drv-card:hover{background:rgba(255,255,255,0.02)}

/* Coloured left stripe for team colour */
.drv-team-stripe{
  position:absolute;left:0;top:0;bottom:0;width:3px;
}

/* F1 logo watermark inside each card */
.drv-card-f1bg{
  position:absolute;right:8px;top:50%;transform:translateY(-50%);
  width:80px;opacity:0;filter:grayscale(1) brightness(10);
  pointer-events:none;user-select:none;display:none;
}

/* Driver headshot container */
.drv-img-wrap{
  position:relative;
  background:linear-gradient(135deg,#1a1a1a 0%,#0f0f0f 100%);
  overflow:hidden;display:flex;align-items:flex-end;justify-content:center;box-shadow:inset 0 0 20px rgba(225,6,0,0.1);
}
.drv-img-wrap::after{
  content:'';position:absolute;bottom:0;left:0;right:0;height:40%;
  background:linear-gradient(to top,rgba(22,22,22,1) 0%,transparent 100%);
}
.drv-number-bg{
  position:absolute;top:4px;left:6px;
  font-family:var(--cond);font-size:1.6rem;font-weight:800;
  color:rgba(255,255,255,0.07);line-height:1;pointer-events:none;
  letter-spacing:-0.02em;
}
.drv-photo{
  width:100%;height:100%;object-fit:cover;object-position:top center;
  display:block;
}
.drv-photo-placeholder{
  width:70px;height:90px;
  display:flex;align-items:center;justify-content:center;
  font-family:var(--cond);font-size:1.6rem;font-weight:900;
  color:rgba(225,6,0,0.3);
  padding-bottom:6px;background:linear-gradient(135deg,rgba(225,6,0,0.08) 0%,rgba(225,6,0,0.02) 100%);
}

/* Info section */
.drv-info{
  padding:12px 14px 10px;
  display:flex;flex-direction:column;justify-content:space-between;
  position:relative;z-index:1;
}
.drv-name{font-family:var(--cond);font-size:1.1rem;font-weight:800;letter-spacing:0.03em;text-transform:uppercase;line-height:1.1}
.drv-team-name{font-size:11px;color:var(--muted);margin-top:2px;letter-spacing:0.03em}
.drv-stats-row{display:flex;gap:14px;margin-top:8px}
.drv-stat{display:flex;flex-direction:column;gap:1px}
.drv-stat-v{font-family:var(--cond);font-size:1.1rem;font-weight:700;line-height:1}
.drv-stat-l{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:0.12em}
.drv-note{font-size:11px;color:var(--muted);margin-top:6px;line-height:1.4}
.sel-dot{position:absolute;top:10px;right:10px;width:6px;height:6px;border-radius:50%;background:var(--red)}

/* ── ALL DRIVERS TABLE ── */
.tbl-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
thead tr{border-bottom:2px solid var(--red)}
th{font-family:var(--cond);font-size:9px;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;color:var(--muted);text-align:left;padding:9px 12px;white-space:nowrap}
th:not(:first-child){text-align:right}
tbody tr{border-bottom:1px solid var(--border);transition:background 0.1s}
tbody tr:hover{background:rgba(255,255,255,0.025)}
tbody tr.sel-row{background:rgba(225,6,0,0.04)}
td{padding:9px 12px;white-space:nowrap}
td:not(:first-child){text-align:right;font-size:13px}
.td-drv{display:flex;align-items:center;gap:10px}
.td-thumb{width:36px;height:36px;border-radius:2px;overflow:hidden;background:#1a1a1a;flex-shrink:0;position:relative}
.td-thumb img{width:100%;height:100%;object-fit:cover;object-position:top center}
.td-thumb-dot{position:absolute;bottom:0;left:0;right:0;height:3px}
.td-n{font-weight:600;font-size:13px}
.td-tm{font-size:11px;color:var(--muted)}
.vbar-wrap{display:flex;align-items:center;gap:6px;justify-content:flex-end}
.vbar{height:3px;border-radius:2px;background:var(--red);min-width:2px}
.spip{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--red);margin-left:4px;vertical-align:middle}

/* ── FEATURE IMPORTANCE ── */
.imp-bars{padding:1.1rem 1.25rem}
.imp-r{margin-bottom:11px}
.imp-r:last-child{margin-bottom:0}
.imp-top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px}
.imp-lbl{font-family:var(--cond);font-size:11px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase}
.imp-pct{font-size:11px;color:var(--muted)}
.imp-track{height:3px;background:var(--border2);border-radius:2px;overflow:hidden}
.imp-fill{height:100%;background:var(--red);border-radius:2px;transition:width 0.5s ease}

/* ── EMPTY / FOOTER ── */
.empty{padding:4rem 2rem;text-align:center;font-family:var(--cond);font-size:13px;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted)}
footer{border-top:1px solid var(--border);padding:1.2rem 2.5rem;display:flex;align-items:center;justify-content:space-between;font-family:var(--cond);font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:var(--muted)}
footer strong{color:var(--text);font-weight:700}

/* ── ANIMATIONS ── */
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.fu{animation:fadeUp 0.3s ease both}
.fu1{animation-delay:.04s}.fu2{animation-delay:.08s}.fu3{animation-delay:.13s}.fu4{animation-delay:.18s}

/* ── RESPONSIVE ── */
@media(max-width:1000px){.cols{grid-template-columns:1fr}.kpis{grid-template-columns:1fr 1fr}h1{font-size:2.8rem}}
@media(max-width:600px){.wrap,.hero,.nav{padding-left:1rem;padding-right:1rem}.kpis{grid-template-columns:1fr 1fr}.race-grid{grid-template-columns:1fr 1fr}.hero-stats{flex-wrap:wrap}.drv-card{grid-template-columns:80px 1fr}}
</style>
</head>
<body>

<!-- NAV -->
<nav class="nav">
  <div class="logo"><div class="logo-hex"></div>GridEdge</div>
  <div class="nav-sep"></div>
  <span class="nav-label">F1 Fantasy Optimizer</span>
  <span class="nav-pill">2026 Season</span>
</nav>

<!-- HERO -->
<header class="hero">
  <div class="hero-gridlines"></div>
  <div class="hero-red-glow"></div>
  <!-- F1 Logo Watermark -->
  <img src="{{ url_for('static', filename='f1-logo.png') }}" 
       alt="F1 Logo" 
       onerror="this.src='https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg'"
       style="width:80px;height:auto;flex-shrink:0;object-fit:contain;filter:brightness(0) invert(1);opacity:0.8;">
  <div class="wrap" style="padding-top:0;padding-bottom:0">
    <div class="eyebrow">2026 Season</div>
    <h1 style="margin-bottom:1.2rem;">Build Your<br><em>Dream Team</em></h1>
    <p class="hero-desc">Discover the highest-scoring $100M squad for every race with data-driven analysis.</p>
    <div class="hero-stats">
      <div class="hs"><div class="hs-v">22</div><div class="hs-l">Drivers</div></div>
      <div class="hs"><div class="hs-v">11</div><div class="hs-l">Teams</div></div>
      <div class="hs"><div class="hs-v">$100M</div><div class="hs-l">Budget</div></div>
      <div class="hs"><div class="hs-v">5</div><div class="hs-l">Picks</div></div>
      <div class="hs"><div class="hs-v">22</div><div class="hs-l">Races</div></div>
    </div>
  </div>
</header>

<!-- LOADER -->
<div class="loader" id="loader">
  <div class="loader-logo"><div class="logo-hex"></div>GridEdge</div>
  <div class="loader-word">Running optimizer</div>
  <div class="loader-track"><div class="loader-fill"></div></div>
</div>

<div class="wrap">

  <!-- RACE SELECTOR -->
  <section class="race-sec">
    <div class="sec-hdr">Select Race Weekend</div>
    <div class="race-grid">
      {% for race in races %}
      <button
        class="race-btn{% if not race.completed and loop.index == 4 %} active{% endif %}{% if race.completed %} completed{% endif %}"
        data-race="{{ race.name }}"
        {% if race.completed %}disabled{% endif %}
        onclick="selectRace(this)"
      >
        <span>{{ race.name }}</span>
        <span class="race-id">{% if race.completed %}✓{% else %}{{ race.id }}{% endif %}</span>
      </button>
      {% endfor %}
    </div>
  </section>

  <!-- RESULTS SECTION -->
  <section class="results" id="results">

    <!-- KPIs -->
    <div class="kpis">
      <div class="kpi fu fu1">
        <div class="kpi-l">Race</div>
        <div class="kpi-v" id="kRace" style="font-size:1.05rem;padding-top:6px">—</div>
        <div class="kpi-s" id="kCircuit"></div>
      </div>
      <div class="kpi fu fu2">
        <div class="kpi-l">Total Cost</div>
        <div class="kpi-v" id="kCost">—</div>
        <div class="kpi-s" id="kRemain"></div>
      </div>
      <div class="kpi fu fu3">
        <div class="kpi-l">Predicted Points</div>
        <div class="kpi-v" id="kPts">—</div>
        <div class="kpi-s">5-driver total</div>
      </div>
      <div class="kpi fu fu4">
        <div class="kpi-l">Avg Value Score</div>
        <div class="kpi-v" id="kVal">—</div>
        <div class="kpi-s">pts per $M spent</div>
      </div>
    </div>

    <!-- TWO COLUMNS -->
    <div class="cols">

      <!-- LEFT: chart + table -->
      <div>
        <div class="panel fu fu2">
          <div class="pnl-hdr">
            <span class="pnl-t">Driver Analysis</span>
            <div class="c-tabs">
              <button class="c-tab on" onclick="switchChart('pts',this)">Predicted Pts</button>
              <button class="c-tab" onclick="switchChart('value',this)">Value</button>
              <button class="c-tab" onclick="switchChart('price',this)">Price</button>
            </div>
          </div>
          <div class="chart-body">
            <div class="chart-wrap">
              <canvas id="driverChart" role="img" aria-label="Driver fantasy points bar chart"></canvas>
            </div>
          </div>
        </div>

        <div class="panel fu fu3">
          <div class="pnl-hdr">
            <span class="pnl-t">All Drivers</span>
            <span class="pnl-s">Highlighted = selected in team</span>
          </div>
          <div class="tbl-wrap">
            <table>
              <thead>
                <tr>
                  <th>Driver</th>
                  <th>Price</th>
                  <th>Pred Pts</th>
                  <th>Value</th>
                  <th>Form</th>
                  <th>Reliability</th>
                </tr>
              </thead>
              <tbody id="tblBody"></tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- RIGHT: team + importance -->
      <div>
        <div class="panel fu fu1">
          <div class="pnl-hdr">
            <span class="pnl-t">Optimal Team</span>
            <span class="pnl-s" id="teamCostLbl">5 drivers</span>
          </div>
          <div class="team-grid" id="teamList"></div>
        </div>

        <div class="panel fu fu4">
          <div class="pnl-hdr">
            <span class="pnl-t">Model Signals</span>
            <span class="pnl-s">LightGBM importances</span>
          </div>
          <div class="imp-bars" id="impBars"></div>
        </div>
      </div>
    </div>
  </section>

  <div id="emptyMsg" class="empty">Select an upcoming race to generate your optimal team</div>
</div>

<footer>
  <strong>GridEdge v3.0</strong>
  <span>FastF1 · LightGBM · PuLP ILP · 2026 Season · 22 Races</span>
</footer>

<script>
let chart = null, allDrivers = [], metric = 'pts';

const circTag = t => ({
  street:'<span class="ctag ctag-s">Street Circuit</span>',
  technical:'<span class="ctag ctag-t">Technical Circuit</span>',
  'high speed':'<span class="ctag ctag-h">High Speed Circuit</span>',
}[t] || '<span class="ctag ctag-t">Technical Circuit</span>');

function selectRace(btn) {
  document.querySelectorAll('.race-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  runOptimizer(btn.dataset.race);
}

async function runOptimizer(raceName) {
  document.getElementById('loader').classList.add('on');
  document.getElementById('results').style.display = 'none';
  document.getElementById('emptyMsg').style.display = 'none';
  try {
    const r = await fetch('/api/optimize?race=' + encodeURIComponent(raceName));
    const d = await r.json();
    if (d.error) { alert('Optimizer error: ' + d.error); return; }
    renderResults(d);
  } catch(e) { alert('Network error: ' + e.message); }
  finally { document.getElementById('loader').classList.remove('on'); }
}

// Inline SVG F1-style logo for card background watermark
const F1_MARK = `<svg viewBox="0 0 120 60" xmlns="http://www.w3.org/2000/svg"><text x="0" y="52" font-family="'Barlow Condensed',sans-serif" font-weight="800" font-size="60" fill="white" letter-spacing="-3">F1</text></svg>`;

function driverCardHTML(drv) {
  const initials = drv.name.split(' ').slice(0, 2).map(w => w[0]).join('').toUpperCase();
  const imgSection = drv.image_url
    ? `<img class="drv-photo" src="${drv.image_url}" alt="${drv.name}" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'drv-photo-placeholder\\'>${initials}</div>'">`
    : `<div class="drv-photo-placeholder">${initials}</div>`;

  return `
  <div class="drv-card">
    <div class="drv-team-stripe" style="background:${drv.team_color}"></div>
    <div class="drv-img-wrap">
      <div class="drv-number-bg">${drv.number || ''}</div>
      ${imgSection}
    </div>
    <div class="drv-info">
      <div>
        <div class="drv-name">${drv.name}</div>
        <div class="drv-team-name" style="color:${drv.team_color}">${drv.team}</div>
      </div>
      <div class="drv-stats-row">
        <div class="drv-stat">
          <div class="drv-stat-v">$${drv.price}M</div>
          <div class="drv-stat-l">Price</div>
        </div>
        <div class="drv-stat">
          <div class="drv-stat-v" style="color:var(--red)">${drv.predicted_pts.toFixed(1)}</div>
          <div class="drv-stat-l">Pred Pts</div>
        </div>
        <div class="drv-stat">
          <div class="drv-stat-v">${drv.value_score.toFixed(2)}</div>
          <div class="drv-stat-l">Value</div>
        </div>
      </div>
      ${drv.rationale ? `<div class="drv-note">${drv.rationale}</div>` : ''}
    </div>
    <div class="sel-dot"></div>
  </div>`;
}

function renderResults(d) {
  allDrivers = d.drivers;
  const sel = d.drivers.filter(x => x.selected);

  document.getElementById('kRace').textContent = d.race_name;
  document.getElementById('kCircuit').innerHTML = circTag(d.circuit_type);
  document.getElementById('kCost').textContent = '$' + d.total_cost + 'M';
  document.getElementById('kRemain').textContent = '$' + d.budget_remaining + 'M remaining';
  document.getElementById('kPts').textContent = d.total_pts.toFixed(1);
  const avgV = sel.reduce((s,x) => s + x.value_score, 0) / sel.length;
  document.getElementById('kVal').textContent = avgV.toFixed(2);
  document.getElementById('teamCostLbl').textContent = '$' + d.total_cost + 'M / $100M';

  // Team cards with headshots
  document.getElementById('teamList').innerHTML = sel.map(driverCardHTML).join('');

  // Feature importance
  const maxImp = Math.max(...d.feature_importance.map(f => f.importance));
  document.getElementById('impBars').innerHTML = d.feature_importance.map(f => `
    <div class="imp-r">
      <div class="imp-top">
        <span class="imp-lbl">${f.feature}</span>
        <span class="imp-pct">${(f.importance * 100).toFixed(1)}%</span>
      </div>
      <div class="imp-track"><div class="imp-fill" style="width:${(f.importance/maxImp*100).toFixed(1)}%"></div></div>
    </div>`).join('');

  // Table with mini thumbnails
  const sorted = [...d.drivers].sort((a,b) => b.predicted_pts - a.predicted_pts);
  const maxVal = Math.max(...sorted.map(x => x.value_score));
  document.getElementById('tblBody').innerHTML = sorted.map(drv => `
    <tr class="${drv.selected ? 'sel-row' : ''}">
      <td>
        <div class="td-drv">
          <div class="td-thumb">
            ${drv.image_url ? `<img src="${drv.image_url}" alt="${drv.name}" loading="lazy" onerror="this.style.display='none'">` : ''}
            <div class="td-thumb-dot" style="background:${drv.team_color}"></div>
          </div>
          <div>
            <div class="td-n">${drv.name}${drv.selected ? '<span class="spip"></span>' : ''}</div>
            <div class="td-tm">${drv.team}</div>
          </div>
        </div>
      </td>
      <td>$${drv.price}M</td>
      <td>${drv.predicted_pts.toFixed(1)}</td>
      <td>
        <div class="vbar-wrap">
          <div class="vbar" style="width:${Math.max(2,Math.round(drv.value_score/maxVal*48))}px"></div>
          <span>${drv.value_score.toFixed(2)}</span>
        </div>
      </td>
      <td>${drv.recent_form_pts.toFixed(1)}</td>
      <td>${(100 - drv.dnf_rate * 100).toFixed(0)}%</td>
    </tr>`).join('');

  renderChart(d.drivers, metric);
  document.getElementById('results').style.display = 'block';
}

function switchChart(m, btn) {
  metric = m;
  document.querySelectorAll('.c-tab').forEach(b => b.classList.remove('on'));
  btn.classList.add('on');
  renderChart(allDrivers, metric);
}

function renderChart(drivers, m) {
  const s = [...drivers].sort((a,b) => {
    const va = m==='pts'?b.predicted_pts:m==='value'?b.value_score:b.price;
    const vb = m==='pts'?a.predicted_pts:m==='value'?a.value_score:a.price;
    return va-vb;
  });
  const labels = s.map(d => d.name.split(' ').slice(-1)[0]);
  const vals   = s.map(d => m==='pts'?d.predicted_pts:m==='value'?d.value_score:d.price);
  const colors = s.map(d => d.selected ? '#E10600' : 'rgba(255,255,255,0.1)');
  const bords  = s.map(d => d.selected ? '#E10600' : 'rgba(255,255,255,0.18)');
  if (chart) { chart.destroy(); chart = null; }
  chart = new Chart(document.getElementById('driverChart').getContext('2d'), {
    type:'bar',
    data:{labels,datasets:[{data:vals,backgroundColor:colors,borderColor:bords,borderWidth:1,borderRadius:2}]},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{
        legend:{display:false},
        tooltip:{
          backgroundColor:'#161616',borderColor:'rgba(255,255,255,0.1)',borderWidth:1,
          titleColor:'#f0f0f0',bodyColor:'#666',
          titleFont:{family:"'Barlow Condensed',sans-serif",size:13,weight:'700'},
          callbacks:{
            title:items=>s[items[0].dataIndex].name,
            label:item=>{
              const d=s[item.dataIndex];
              return[`${m==='pts'?'Pred pts':m==='value'?'Value score':'Price'}: ${item.raw.toFixed(2)}`,
                     `Team: ${d.team}`,d.selected?'★ In optimal team':''].filter(Boolean);
            }
          }
        }
      },
      scales:{
        x:{ticks:{color:'#444',font:{family:"'Barlow Condensed',sans-serif",size:10},maxRotation:45,autoSkip:false},grid:{color:'rgba(255,255,255,0.03)'},border:{color:'rgba(255,255,255,0.08)'}},
        y:{ticks:{color:'#444',font:{size:11}},grid:{color:'rgba(255,255,255,0.03)'},border:{color:'rgba(255,255,255,0.08)'}}
      }
    }
  });
}

window.addEventListener('DOMContentLoaded', () => {
  const def = document.querySelector('.race-btn.active');
  if (def) runOptimizer(def.dataset.race);
});
</script>
</body>
</html>"""

if __name__ == "__main__":
    print("GridEdge — http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)