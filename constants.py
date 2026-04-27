import os

BASE       = os.path.dirname(__file__)
PRICES_CSV = os.path.join(BASE, "data", "prices.csv")
FEAT_CSV   = os.path.join(BASE, "data", "features.csv")
MODEL_PATH = os.path.join(BASE, "models", "lgbm_model.pkl")

FEATURE_COLS = ["recent_form_pts", "position_std", "qual_position", "dnf_rate", "team_momentum"]
BUDGET       = 100.0

# Official F1 CDN image URL pattern discovered from formula1.com
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
