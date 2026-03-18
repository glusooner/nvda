# """
# run_all.py – Execute the full option data pipeline.
#
# This script simply calls the `main` functions from the two stage
# scripts `download_options.py` and `compute_options.py`.
# It allows you to fetch the latest raw data from Yahoo Finance and
# then immediately compute the derived columns without having to run
# the two scripts manually.
#
# Usage (from the project root):
#   .\.venv\Scripts\python run_all.py
#"""

import sys
from download_options import main as download_main
from compute_options import main as compute_main

def run():
    if len(sys.argv) < 2:
        print("Usage: python run_all.py [0|1]")
        print("  0: Run both download and compute (default)")
        print("  1: Only run compute using existing raw data")
        # Defaulting to 0 if no argument is provided, or we can just exit.
        # Let's default to 0 to be helpful.
        mode = "0"
    else:
        mode = sys.argv[1]

    if mode == "0":
        print("Mode 0: Running full pipeline (Download + Compute)...")
        download_main()
        compute_main()
    elif mode == "1":
        print("Mode 1: Running compute only...")
        compute_main()
    else:
        print(f"Error: Unknown mode '{mode}'. Use 0 for full pipeline or 1 for compute only.")

if __name__ == "__main__":
    run()







