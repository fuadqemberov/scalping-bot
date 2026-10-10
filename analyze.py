"""
Analyse an exported trade history.

    python analyze.py trade_history.csv

Accepts the CSV from the dashboard's "Download CSV" button (old or new
format) or from `python backtest.py --csv trades.csv`.
"""

import argparse

import pandas as pd

import analysis


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="trade history CSV")
    args = ap.parse_args()
    print(analysis.report_text(analysis.analyze(pd.read_csv(args.csv))))


if __name__ == "__main__":
    main()
