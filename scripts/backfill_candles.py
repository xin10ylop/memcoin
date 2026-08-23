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
    ap.add_argument("--no-refresh", action="store_true",
                    help="only fetch pools that have no candles at all")
    ap.add_argument("--refresh-after-min", type=float, default=25.0,
                    help="re-fetch a pool whose newest candle is older than this many minutes")
    ap.add_argument("--max-track-age-min", type=float, default=1440.0,
                    help="stop refreshing pools older than this")
    ap.add_argument("--loop", action="store_true", help="run continuously")
    ap.add_argument("--log", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    store = Store(args.db)
    client = GeckoTerminalClient()
    client.http.bucket.rate = args.rpm / 60.0

    while True:
        # Two kinds of work, in priority order.
        #
        # 1. Pools never fetched. Without candles they cannot be labelled at all.
        # 2. Pools whose stored candles stop before their label window closes.
        #    A pool fetched when it was 20 minutes old holds no data covering a
        #    decision at minute 15 with a 45-minute horizon, so its label would
        #    be silently truncated to whatever happened to be cached. Refreshing
        #    these is what turns a young pool into a usable training row.
        never = store.conn.execute(
            """SELECT p.pool FROM pools p
               WHERE p.created_at IS NOT NULL
                 AND (julianday('now') - julianday(p.created_at)) * 1440.0 >= ?
                 AND p.pool NOT IN (SELECT DISTINCT pool FROM candles)
               ORDER BY p.created_at ASC LIMIT ?""",
            (args.min_age_min, args.limit),
        ).fetchall()

        stale = [] if args.no_refresh else store.conn.execute(
            """SELECT c.pool FROM (
                   SELECT pool, MAX(ts) AS last_ts FROM candles GROUP BY pool
               ) c
               JOIN pools p ON p.pool = c.pool
               WHERE (strftime('%s','now') - c.last_ts) / 60.0 >= ?
                 AND (julianday('now') - julianday(p.created_at)) * 1440.0 <= ?
               ORDER BY c.last_ts ASC LIMIT ?""",
            (args.refresh_after_min, args.max_track_age_min, max(1, args.limit // 2)),
        ).fetchall()

        seen = set()
        pools = []
        for r in list(never) + list(stale):
            if r[0] not in seen:
                seen.add(r[0])
                pools.append(r[0])
        log.info("queue: %d never-fetched, %d stale", len(never), len(stale))
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
