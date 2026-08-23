#!/usr/bin/env python3
"""Records pump.fun token creations at t=0 from the PumpPortal websocket.

This is the low-latency half of the data pipeline. The polling collector sees
pools once GeckoTerminal lists them, at 1-15 minutes old; measurement on this
panel showed the large multiples are already gone by then (best outcome from a
2-minute entry was 7.3x, versus 318x from the launch candle). This process sees
the same tokens within seconds of deployment.

It also accumulates the deployer track record first-hand, which is the strongest
free rug signal there is.
"""
import argparse
import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alpha.data.pumpportal import Launch, Migration, PumpPortalStream  # noqa: E402
from alpha.data.store import Store  # noqa: E402

log = logging.getLogger("launchstream")


async def main_async(args: argparse.Namespace) -> int:
    store = Store(args.db)
    stream = PumpPortalStream()
    counters = {"new": 0, "dupe": 0, "migrations": 0}
    last_report = time.monotonic()

    def on_launch(launch: Launch) -> None:
        nonlocal last_report
        if store.record_launch(launch):
            counters["new"] += 1
            record = store.dev_record(launch.dev_wallet) or {}
            prior = max(0, int(record.get("launches", 1)) - 1)
            if prior >= args.warn_repeat_devs:
                log.info(
                    "repeat deployer: %s has now launched %d tokens (latest %s)",
                    launch.dev_wallet[:16], prior + 1, launch.symbol[:16],
                )
        else:
            counters["dupe"] += 1

        if time.monotonic() - last_report >= args.report_seconds:
            last_report = time.monotonic()
            log.info("launches=%s | store=%s", counters, store.launch_stats())

    def on_migration(migration: Migration) -> None:
        if store.record_migration(migration):
            counters["migrations"] += 1
            log.info("GRADUATION: %s -> PumpSwap", migration.mint[:16])

    log.info("connecting to PumpPortal launch stream")
    stats = await stream.run(
        on_launch=on_launch, on_migration=on_migration, max_seconds=args.max_runtime
    )
    log.info("stream ended: %s | %s", stats, store.launch_stats())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/alpha.db")
    ap.add_argument("--max-runtime", type=float, default=None, help="stop after N seconds")
    ap.add_argument("--report-seconds", type=float, default=300.0)
    ap.add_argument("--warn-repeat-devs", type=int, default=5,
                    help="log when a deployer exceeds this many launches")
    ap.add_argument("--log", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
