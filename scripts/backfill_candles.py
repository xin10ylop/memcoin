#!/usr/bin/env python3
"""Backfill minute OHLCV for pools already in the panel.

Only pools discovered prospectively by the collector are fetched. The cohort was
fixed before outcomes were known, so looking up those outcomes now introduces no
selection bias — we are not choosing pools based on how they turned out.

Pools are processed oldest-first so that the ones whose outcomes have fully
settled are labelled first.
"""
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alpha.data.geckoterminal import GeckoTerminalClient  # noqa: E402
from alpha.data.store import Store  # noqa: E402

log = logging.getLogger("backfill")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/alpha.db")
    ap.add_argument("--rpm", type=float, default=8.0, help="stay under the shared rate budget")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--min-age-min", type=float, default=50.0,
                    help="only pools old enough for the label horizon to have elapsed")
    ap.add_argument("--refresh", action="store_true", help="re-fetch pools that already have candles")
    ap.add_argument("--loop", action="store_true", help="run continuously")
    ap.add_argument("--log", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    store = Store(args.db)
    client = GeckoTerminalClient()
    client.http.bucket.rate = args.rpm / 60.0

    while True:
        having = "" if args.refresh else "AND p.pool NOT IN (SELECT DISTINCT pool FROM candles)"
        rows = store.conn.execute(
            f"""SELECT p.pool FROM pools p
                WHERE p.created_at IS NOT NULL
                  AND (julianday('now') - julianday(p.created_at)) * 1440.0 >= ?
                  {having}
                ORDER BY p.created_at ASC LIMIT ?""",
            (args.min_age_min, args.limit),
        ).fetchall()
        pools = [r[0] for r in rows]
        if not pools:
            log.info("nothing to backfill")
            if not args.loop:
                return 0
            time.sleep(120)
            continue

        log.info("backfilling %d pools", len(pools))
        done = saved = 0
        for pool in pools:
            try:
                candles = client.ohlcv_history(pool, max_calls=1)
                if candles:
                    saved += store.record_candles(pool, candles)
                done += 1
                if done % 25 == 0:
                    log.info("%d/%d pools, %d candles saved", done, len(pools), saved)
            except Exception:
                log.exception("failed on %s", pool)
        log.info("backfill pass complete: %d pools, %d candles", done, saved)
        if not args.loop:
            return 0
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
