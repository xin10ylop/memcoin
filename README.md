# alpha — quantitative Solana memecoin trading system

A research-grade system for systematically trading newly-launched Solana tokens.
It discovers every new pool on the chain, screens it for structural safety,
builds point-in-time features, scores it with a calibrated model, sizes the
position against explicit risk limits, and simulates or executes the trade.

**It defaults to paper trading and requires no private key to run.**

## Status

Under active development. See `docs/` for the design rationale and
`docs/FINDINGS.md` for measured results from live data.

## Why this exists

Most published memecoin strategies fail for reasons that have nothing to do
with signal quality:

- **Survivorship bias.** Datasets assembled from tokens still visible today
  contain only survivors, so a model trained on them never learns what death
  looks like. This system records every pool *at birth*, before its outcome is
  known.
- **Fantasy execution.** Backtests that fill at mid price ignore that a $500
  order into a $30k pool moves the price ~3.3% each way. Measured round-trip
  cost here is 3.6–9% depending on size and depth, and it is modelled explicitly.
- **Fake volume.** Wash-trading farms manufacture the exact volume and
  transaction-count signals naive filters screen on. One pool measured here had
  97% of its volume manufactured.
- **Unfalsifiable validation.** Random k-fold on rows from the same token leaks
  across folds. This system uses purged, pool-grouped, walk-forward CV and
  refuses to emit a model that cannot beat a base-rate baseline.

## Quick start

```bash
pip install -e ".[ml,dev]"

# Start collecting the panel dataset (runs continuously)
python scripts/collectorctl.py start --db data/alpha.db --rpm 16

# Check what has accumulated
python -m alpha.cli.main status
```

## Layout

```
src/alpha/
  http.py             rate-limited HTTP transport shared by all sources
  data/               ingestion: GeckoTerminal client, panel store, collector
  safety/             hard-reject screening: RPC mint checks, RugCheck, verdicts
  features/           point-in-time features, triple-barrier labels, wallets
  models/             purged CV, deflated Sharpe, PBO, calibrated scorer
  execution/          AMM cost model
  risk/               position sizing, portfolio limits, circuit breakers
  backtest/           event-driven simulator
```

## Safety

This software trades speculative assets where total loss of any individual
position is the normal case. It is provided for research. Run it in paper mode,
read `docs/FINDINGS.md`, and do not deploy capital you cannot afford to lose.
