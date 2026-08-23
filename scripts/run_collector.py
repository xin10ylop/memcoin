#!/usr/bin/env python3
"""Entrypoint for the long-running cohort collector."""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alpha.data.collector import Collector, CollectorConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Solana new-pool cohort collector")
    ap.add_argument("--db", default="data/alpha.db")
    ap.add_argument("--rpm", type=float, default=25.0, help="API requests per minute budget")
    ap.add_argument("--pages", type=int, default=10, help="new_pools pages to sweep per cycle")
    ap.add_argument("--cycle", type=float, default=60.0, help="seconds per cycle")
    ap.add_argument("--max-runtime", type=float, default=None, help="stop after N seconds")
    ap.add_argument("--no-trades", action="store_true", help="skip swap-level collection")
    ap.add_argument("--log", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = CollectorConfig(
        db_path=args.db,
        requests_per_minute=args.rpm,
        discovery_pages=args.pages,
        cycle_seconds=args.cycle,
        max_runtime_seconds=args.max_runtime,
        track_trades=not args.no_trades,
    )
    collector = Collector(cfg)
    collector.install_signal_handlers()
    collector.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
