"""
Stage 1 & 2: FastF1 data ingestion + price CSV merge.
Fetches race results for 2022-2026, caches to CSV, then merges with prices.csv.
Supports incremental fetching for weekly retraining.
"""
import os
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

import pandas as pd
import fastf1
from rapidfuzz import process, fuzz

CACHE_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "f1_cache")
RAW_CSV    = os.path.join(os.path.dirname(__file__), "..", "data", "race_data.csv")
PRICES_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "prices.csv")
MERGED_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "merged_data.csv")

# Seasons to fetch — includes 2025 full season and 2026 (in progress)
SEASONS = [2022, 2023, 2024, 2025, 2026]

# Static normalisation map for known edge-case name mismatches
NAME_MAP = {
    "Zhou Guanyu": "Guanyu Zhou",
    "Magnussen": "Kevin Magnussen",
    "Hulkenberg": "Nico Hulkenberg",
    "Hülkenberg": "Nico Hulkenberg",
    "Andrea Kimi Antonelli": "Kimi Antonelli",
}

REQUIRED_COLS   = {"driver_name", "price_m", "team", "race_week"}
REQUIRED_DTYPES = {"price_m": float}


def validate_prices(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"prices.csv missing columns: {missing}")
    for col, dtype in REQUIRED_DTYPES.items():
        if not pd.api.types.is_float_dtype(df[col]):
            try:
                df[col] = df[col].astype(dtype)
            except Exception:
                raise ValueError(f"prices.csv column '{col}' cannot be cast to {dtype}")


def _get_completed_rounds(year: int) -> list:
    """Get rounds that have already been completed for a given year."""
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        events = schedule[schedule["EventFormat"] != "testing"]
        today = datetime.now()

        completed = []
        for _, event in events.iterrows():
            # Check if the race date has passed
            event_date = pd.to_datetime(event.get("EventDate", event.get("Session5Date", None)))
            if event_date is not None and event_date.date() < today.date():
                completed.append(int(event["RoundNumber"]))

        return completed
    except Exception as e:
        print(f"  [ingest] Could not determine completed rounds for {year}: {e}")
        return []


def fetch_race_data(use_cache: bool = True, force_refresh: bool = False,
                    seasons: list = None) -> pd.DataFrame:
    """Fetch finishing positions, lap times, tyre, pit stops, qualifying from FastF1.

    Args:
        use_cache: If True and CSV exists, load from cache (unless force_refresh).
        force_refresh: If True, re-fetch all data even if cache exists.
        seasons: List of years to fetch. Defaults to SEASONS.
    """
    if seasons is None:
        seasons = SEASONS

    existing_df = None
    if use_cache and os.path.exists(RAW_CSV) and not force_refresh:
        existing_df = pd.read_csv(RAW_CSV)
        # Check if we already have all the data we need
        existing_keys = set()
        if not existing_df.empty:
            existing_keys = set(
                zip(existing_df["year"].astype(int), existing_df["round"].astype(int))
            )
        print(f"[ingest] Loaded {len(existing_df)} cached records "
              f"({len(existing_keys)} year-round combos)")

    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    records = []
    current_year = datetime.now().year

    for year in seasons:
        try:
            if year > current_year:
                print(f"[ingest] Skipping future year {year}")
                continue

            schedule = fastf1.get_event_schedule(year, include_testing=False)
            all_rounds = schedule[schedule["EventFormat"] != "testing"]["RoundNumber"].tolist()

            # For the current year, only fetch completed rounds
            if year == current_year:
                completed_rounds = _get_completed_rounds(year)
                rounds_to_fetch = [r for r in all_rounds if r in completed_rounds]
                print(f"[ingest] Year {year} (current): {len(rounds_to_fetch)}/{len(all_rounds)} "
                      f"rounds completed")
            else:
                rounds_to_fetch = all_rounds
                print(f"[ingest] Year {year}: {len(rounds_to_fetch)} races")

            for rnd in rounds_to_fetch:
                # Skip if already cached (incremental fetch)
                if existing_df is not None and (year, rnd) in existing_keys:
                    continue

                try:
                    race = fastf1.get_session(year, rnd, "R")
                    race.load(laps=True, telemetry=False, weather=False, messages=False)
                    qual = fastf1.get_session(year, rnd, "Q")
                    qual.load(laps=False, telemetry=False, weather=False, messages=False)

                    results = race.results[["DriverNumber", "Abbreviation", "FullName",
                                            "TeamName", "Position", "Points"]].copy()
                    results["year"]        = year
                    results["round"]       = rnd
                    results["circuit"]     = race.event["EventName"]
                    results["country"]     = race.event["Country"]
                    results["fantasy_pts"] = results["Points"]  # use championship pts as proxy

                    # Pit stop count per driver
                    laps = race.laps[["DriverNumber", "PitInTime"]].dropna(subset=["PitInTime"])
                    pit_counts = laps.groupby("DriverNumber").size().reset_index(name="pit_stops")
                    results = results.merge(pit_counts, on="DriverNumber", how="left")
                    results["pit_stops"] = results["pit_stops"].fillna(0).astype(int)

                    # Qualifying position
                    q_res = qual.results[["DriverNumber", "Position"]].rename(
                        columns={"Position": "qual_position"})
                    results = results.merge(q_res, on="DriverNumber", how="left")

                    records.append(results)
                    print(f"  [ingest] {year} R{rnd}: {race.event['EventName']} "
                          f"— {len(results)} drivers")

                except Exception as e:
                    print(f"  [ingest] SKIP {year} R{rnd}: {e}")

        except Exception as e:
            print(f"[ingest] Could not load schedule for {year}: {e}")
            continue

    # Combine new records with existing data
    if records:
        new_df = pd.concat(records, ignore_index=True)
        if existing_df is not None and not existing_df.empty:
            df = pd.concat([existing_df, new_df], ignore_index=True)
            # Remove duplicates (same year + round + driver)
            df = df.drop_duplicates(
                subset=["year", "round", "FullName"], keep="last"
            ).reset_index(drop=True)
        else:
            df = new_df
        print(f"[ingest] Fetched {len(new_df)} new records")
    elif existing_df is not None:
        df = existing_df
        print(f"[ingest] No new data to fetch, using {len(df)} cached records")
    else:
        raise RuntimeError("No race data fetched. Check network / FastF1 availability.")

    df.to_csv(RAW_CSV, index=False)
    print(f"[ingest] Saved {len(df)} total rows to {RAW_CSV}")
    return df


def merge_prices(race_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Fuzzy-merge race data with price CSV on driver name."""
    validate_prices(prices_df)

    price_names = prices_df["driver_name"].tolist()
    race_names  = race_df["FullName"].fillna("").tolist()

    matched_prices = []
    for name in race_names:
        # Apply static map first
        lookup = NAME_MAP.get(name, name)
        result = process.extractOne(lookup, price_names,
                                    scorer=fuzz.token_sort_ratio)
        if result and result[1] >= 90:
            row = prices_df[prices_df["driver_name"] == result[0]].iloc[0]
            if result[1] < 95:
                print(f"  [merge] Low-confidence match: '{name}' -> '{result[0]}' ({result[1]:.0f})")
            matched_prices.append(row.to_dict())
        else:
            best_score = result[1] if result else 0
            print(f"  [merge] UNMATCHED: '{name}' (best={best_score:.0f})")
            matched_prices.append({"driver_name": name, "price_m": None,
                                   "team": None, "race_week": None})

    price_frame = pd.DataFrame(matched_prices)
    merged = pd.concat([race_df.reset_index(drop=True),
                        price_frame.reset_index(drop=True)], axis=1)
    unmatched = merged["price_m"].isna().sum()
    if unmatched:
        print(f"[merge] WARNING: {unmatched} unmatched driver records (price_m is null)")

    merged.to_csv(MERGED_CSV, index=False)
    print(f"[merge] Merged dataset saved to {MERGED_CSV} ({len(merged)} rows)")
    return merged


if __name__ == "__main__":
    race_df   = fetch_race_data(use_cache=True)
    prices_df = pd.read_csv(PRICES_CSV)
    merged    = merge_prices(race_df, prices_df)
    print(merged.head())
