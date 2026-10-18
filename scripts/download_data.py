"""Download the official CDC BRFSS 2015 file, verify it and build the local cache.

Usage:
    python scripts/download_data.py            # download from CDC if not present
    python scripts/download_data.py --xpt PATH # use an existing LLCP2015.XPT
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_loading  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xpt", type=Path, default=None, help="path to an existing LLCP2015.XPT")
    args = parser.parse_args()
    try:
        df, report = data_loading.load_brfss_2015(xpt_path=args.xpt, use_cache=False)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report.to_dict(), indent=2, default=str))
    print(f"Cached {df.shape[0]:,} rows x {df.shape[1]} columns to {data_loading.CACHE_PATH}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
