#!/usr/bin/env python3
"""
GridEdge Scheduled Weekly Retraining Configuration

This module provides two methods for scheduling weekly model retraining:

1. Windows Task Scheduler (recommended for Windows)
   - Run: python schedule_config.py --install
   - This creates a Windows Task Scheduler task that runs every Tuesday at 6 AM

2. Python schedule library (cross-platform alternative)
   - Run: python schedule_config.py --daemon
   - This starts a long-running daemon that executes weekly retraining

Usage:
  python schedule_config.py --install       # Install Windows Task Scheduler task
  python schedule_config.py --uninstall     # Remove the scheduled task
  python schedule_config.py --daemon        # Run as a Python scheduling daemon
  python schedule_config.py --run-now       # Run retraining immediately (for testing)
"""
import os
import sys
import subprocess
import argparse
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable
RETRAIN_SCRIPT = os.path.join(BASE, "weekly_retrain.py")
TASK_NAME = "GridEdge_Weekly_Retrain"


def install_windows_task(day="TUE", time="06:00"):
    """Register a Windows Task Scheduler task for weekly retraining.

    Args:
        day: Day of week (MON, TUE, WED, THU, FRI, SAT, SUN)
        time: Time in HH:MM format (24-hour)
    """
    # Build the command that Task Scheduler will execute
    cmd = f'"{PYTHON_EXE}" "{RETRAIN_SCRIPT}"'

    # Create the scheduled task
    schtasks_cmd = [
        "schtasks", "/Create",
        "/TN", TASK_NAME,
        "/TR", cmd,
        "/SC", "WEEKLY",
        "/D", day,
        "/ST", time,
        "/F",  # Force overwrite if exists
        "/RL", "HIGHEST",  # Run with highest privileges
    ]

    print(f"Creating scheduled task: {TASK_NAME}")
    print(f"  Schedule: Every {day} at {time}")
    print(f"  Command: {cmd}")
    print(f"  Working dir: {BASE}")

    try:
        result = subprocess.run(schtasks_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"\n✓ Task '{TASK_NAME}' created successfully!")
            print(f"  The model will retrain every {day} at {time}.")
            print(f"\n  To verify: schtasks /Query /TN {TASK_NAME}")
            print(f"  To remove: python schedule_config.py --uninstall")
            print(f"  To test:   python schedule_config.py --run-now")
        else:
            print(f"\n✗ Failed to create task: {result.stderr}")
            print("  Try running this script as Administrator.")
    except FileNotFoundError:
        print("\n✗ schtasks not found — are you on Windows?")
        print("  Use --daemon for cross-platform scheduling instead.")


def uninstall_windows_task():
    """Remove the Windows Task Scheduler task."""
    try:
        result = subprocess.run(
            ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"✓ Task '{TASK_NAME}' removed successfully.")
        else:
            print(f"✗ Could not remove task: {result.stderr}")
    except FileNotFoundError:
        print("✗ schtasks not found — are you on Windows?")


def run_daemon():
    """Run as a long-lived Python scheduling daemon.

    This is a cross-platform alternative to Windows Task Scheduler.
    The process must stay running (e.g., in a terminal or as a service).
    """
    try:
        import schedule
        import time
    except ImportError:
        print("Installing 'schedule' package...")
        subprocess.check_call([PYTHON_EXE, "-m", "pip", "install", "schedule"])
        import schedule
        import time

    def retrain_job():
        print(f"\n[{datetime.now()}] Starting weekly retraining...")
        result = subprocess.run(
            [PYTHON_EXE, RETRAIN_SCRIPT],
            cwd=BASE,
            capture_output=False
        )
        if result.returncode == 0:
            print(f"[{datetime.now()}] Retraining completed successfully.")
        else:
            print(f"[{datetime.now()}] Retraining failed with code {result.returncode}")

    # Schedule for every Tuesday at 6:00 AM
    schedule.every().tuesday.at("06:00").do(retrain_job)

    print("=" * 60)
    print("  GridEdge Weekly Retrain Daemon")
    print("=" * 60)
    print(f"  Schedule: Every Tuesday at 06:00")
    print(f"  Script: {RETRAIN_SCRIPT}")
    print(f"  Started: {datetime.now()}")
    print(f"  Press Ctrl+C to stop")
    print("=" * 60)

    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


def run_now():
    """Run retraining immediately (for testing)."""
    print(f"Running retraining now at {datetime.now()}...")
    result = subprocess.run(
        [PYTHON_EXE, RETRAIN_SCRIPT],
        cwd=BASE,
    )
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridEdge Scheduling Configuration")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--install", action="store_true",
                       help="Install Windows Task Scheduler task (every Tuesday 6 AM)")
    group.add_argument("--uninstall", action="store_true",
                       help="Remove the Windows Task Scheduler task")
    group.add_argument("--daemon", action="store_true",
                       help="Run as a Python scheduling daemon (cross-platform)")
    group.add_argument("--run-now", action="store_true",
                       help="Run retraining immediately (for testing)")
    parser.add_argument("--day", default="TUE",
                        help="Day for scheduled task (default: TUE)")
    parser.add_argument("--time", default="06:00",
                        help="Time for scheduled task in HH:MM (default: 06:00)")
    args = parser.parse_args()

    if args.install:
        install_windows_task(day=args.day, time=args.time)
    elif args.uninstall:
        uninstall_windows_task()
    elif args.daemon:
        run_daemon()
    elif args.run_now:
        sys.exit(run_now())
