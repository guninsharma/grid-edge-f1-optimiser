#!/usr/bin/env python3
"""
GridEdge pipeline runner.
Runs all stages in sequence: ingest → features → train → evaluate.

Usage:
  python run_pipeline.py --real-data          # full pipeline with FastF1 data
  python run_pipeline.py --skip-ingest        # re-train on existing data
  python run_pipeline.py --real-data --weekly  # weekly retrain mode (incremental)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    parser = argparse.ArgumentParser(description="GridEdge pipeline")
    parser.add_argument("--real-data",   action="store_true",
                        help="Fetch real F1 data from FastF1 (requires internet)")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip data ingestion, use existing CSV files")
    parser.add_argument("--no-eval",     action="store_true",
                        help="Skip evaluation step")
    parser.add_argument("--weekly",      action="store_true",
                        help="Weekly retrain mode — incremental fetch, recent-only training")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force re-download all race data (ignore cache)")
    args = parser.parse_args()

    print("=" * 60)
    print("  GridEdge Pipeline")
    print("=" * 60)

    # Stage 1 & 2: Data ingestion
    if not args.skip_ingest:
        if args.real_data:
            print("\n[Stage 1] Fetching real FastF1 data…")
            import pandas as pd
            from ingest import fetch_race_data, merge_prices
            PRICES_CSV = os.path.join(os.path.dirname(__file__), "data", "prices.csv")

            race_df   = fetch_race_data(
                use_cache=True,
                force_refresh=args.force_refresh
            )
            prices_df = pd.read_csv(PRICES_CSV)
            merged    = merge_prices(race_df, prices_df)

            print("\n[Stage 3] Engineering features…")
            from features import build_features
            feat_df = build_features(merged)
        else:
            print("\nERROR: Must use --real-data flag. Synthetic data removed.")
            print("Usage: python run_pipeline.py --real-data")
            sys.exit(1)
    else:
        print("\n[Stage 1-3] Skipping ingest — loading existing features.csv")
        import pandas as pd
        feat_path = os.path.join(os.path.dirname(__file__), "data", "features.csv")
        feat_df = pd.read_csv(feat_path)

    # Stage 4: Model training
    if args.weekly:
        print("\n[Stage 4] Weekly retrain — training on recent races only…")
        from model import train_model
        model, train_df, test_df = train_model(feat_df, lookback_races=12)
    else:
        print("\n[Stage 4] Training LightGBM model…")
        from model import train_model
        model, train_df, test_df = train_model(feat_df)

    # Stage 5: Optimizer sanity check
    print("\n[Stage 5] Optimizer sanity check…")
    import subprocess
    result = subprocess.run([sys.executable, "src/optimizer.py"],
                            capture_output=True, text=True)
    print(result.stdout.strip())

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("  Run weekly predictions: python weekly_retrain.py")
    print("  Validate approach: python validate_weekly.py")
    print("  Launch app: python app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
