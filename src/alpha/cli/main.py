"""Command-line interface for the alpha system.

Every command that could spend money is explicit about which mode it is in, and
paper trading is the default everywhere.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from alpha.data.store import Store

app = typer.Typer(
    add_completion=False,
    help="Quantitative Solana memecoin trading system. Paper trading by default.",
    no_args_is_help=True,
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def status(db: str = typer.Option("data/alpha.db", help="Panel database path")) -> None:
    """Show what the collector has accumulated."""
    store = Store(db)
    stats = store.stats()
    table = Table(title="Panel dataset", show_header=True, header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in stats.items():
        table.add_row(key, f"{value:,}" if isinstance(value, int) else str(value))
    console.print(table)

    labelled = store.conn.execute("SELECT COUNT(DISTINCT pool) FROM candles").fetchone()[0]
    if stats["pools"]:
        console.print(
            f"\n[dim]{labelled} of {stats['pools']} pools have candle history "
            f"({100 * labelled / stats['pools']:.0f}%)[/dim]"
        )


@app.command()
def collect(
    db: str = typer.Option("data/alpha.db"),
    rpm: float = typer.Option(16.0, help="API requests per minute budget"),
    pages: int = typer.Option(8, help="new_pools pages per sweep"),
    minutes: Optional[float] = typer.Option(None, help="stop after N minutes"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the cohort collector in the foreground."""
    _setup_logging(verbose)
    from alpha.data.collector import Collector, CollectorConfig

    collector = Collector(
        CollectorConfig(
            db_path=db, requests_per_minute=rpm, discovery_pages=pages,
            max_runtime_seconds=minutes * 60 if minutes else None,
        )
    )
    collector.install_signal_handlers()
    collector.run()


@app.command()
def screen(
    mint: str = typer.Argument(..., help="Token mint address"),
    liquidity: float = typer.Option(0.0, help="Known pool liquidity in USD"),
) -> None:
    """Run the safety screen against one token."""
    from alpha.safety import SafetyScreener

    report = SafetyScreener().screen(mint, liquidity_usd=liquidity or None)
    colour = {"PASS": "green", "IMMATURE": "yellow", "REJECT": "red"}[report.verdict.name]
    console.print(f"\n[bold {colour}]{report.verdict.name}[/bold {colour}]  risk score {report.risk_score:.0f}/100\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("check"); table.add_column("result"); table.add_column("detail")
    for check in report.checks:
        if check.passed:
            table.add_row(check.name, "[green]pass[/green]", check.detail)
        else:
            tag = "structural" if check.structural else "maturity"
            table.add_row(check.name, f"[red]{check.severity.name}[/red] ({tag})", check.detail)
    console.print(table)

    if report.rugcheck and report.rugcheck.ok:
        rc = report.rugcheck
        console.print(
            f"\n[dim]deployer {rc.creator[:16]}… has launched {rc.creator_token_count} tokens · "
            f"{rc.total_holders} holders · top holder {rc.top_holder_pct:.1f}% · "
            f"launchpad {rc.launchpad or 'unknown'}[/dim]"
        )


@app.command()
def wash(
    pool: str = typer.Argument(..., help="Pool address"),
    db: str = typer.Option("data/alpha.db"),
) -> None:
    """Check a pool's swap history for manufactured volume."""
    from alpha.data.geckoterminal import GeckoTerminalClient
    from alpha.features.wallets import detect_wash_trading

    store = Store(db)
    rows = [dict(r) for r in store.conn.execute("SELECT * FROM trades WHERE pool=?", (pool,)).fetchall()]
    if not rows:
        console.print("[dim]no stored trades; fetching live…[/dim]")
        rows = GeckoTerminalClient().trades(pool)
    report = detect_wash_trading(pool, rows)
    colour = "red" if report.is_suspicious else "green"
    console.print(f"\n[bold {colour}]wash score {report.wash_score:.3f}[/bold {colour}] "
                  f"over {report.n_trades} trades by {report.n_wallets} wallets\n")
    for key, value in report.signals.items():
        console.print(f"  {key:24} {value}")
    for reason in report.reasons:
        console.print(f"  [yellow]•[/yellow] {reason}")
    if report.suspect_volume_share:
        console.print(
            f"\n  [red]{report.suspect_volume_share:.1%} of volume looks manufactured[/red]"
        )


@app.command()
def dataset(
    db: str = typer.Option("data/alpha.db"),
    out: str = typer.Option("data/processed/dataset.csv"),
    horizon: int = typer.Option(45, help="Label horizon in minutes"),
    take_profit: float = typer.Option(1.50),
    stop_loss: float = typer.Option(0.45),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build the labelled training table from collected data."""
    _setup_logging(verbose)
    from alpha.features.dataset import DatasetBuilder, DatasetConfig
    from alpha.features.label import LabelConfig

    store = Store(db)
    builder = DatasetBuilder(
        store,
        config=DatasetConfig(
            label=LabelConfig(take_profit=take_profit, stop_loss=stop_loss, horizon_min=horizon)
        ),
    )
    rows = builder.build()
    if not rows:
        console.print("[yellow]No labelled rows yet.[/yellow] Pools need candle history and "
                      "must be older than the label horizon. Run the collector longer, then "
                      "`python scripts/backfill_candles.py`.")
        raise typer.Exit(1)

    frame = builder.to_frame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    from alpha.features.label import summarise
    summary = summarise([r.label for r in rows])
    console.print(f"\n[green]{len(rows)} rows[/green] from {frame['pool'].nunique()} pools → {out}\n")
    for key, value in summary.items():
        console.print(f"  {key:20} {value}")


@app.command()
def train(
    data: str = typer.Option("data/processed/dataset.csv"),
    out: str = typer.Option("data/processed/model.pkl"),
    permutation: bool = typer.Option(False, help="Run a permutation test (slow but definitive)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Train and validate the scoring model."""
    _setup_logging(verbose)
    import numpy as np
    import pandas as pd

    from alpha.features.build import FEATURE_NAMES
    from alpha.models.scorer import Scorer

    frame = pd.read_csv(data)
    features = [c for c in FEATURE_NAMES if c in frame.columns]
    X = frame[features].to_numpy(dtype=float)

    from alpha.models.two_stage import TwoStageScorer

    scorer = TwoStageScorer()
    report = scorer.fit(
        X,
        frame["y_survived"].to_numpy(dtype=int),
        frame["y_is_win"].to_numpy(dtype=int),
        frame["decision_ts"].to_numpy(dtype=float),
        frame["pool"].astype(str).tolist(),
        frame.get("sample_weight"),
        features,
    )
    colour = "green" if report.usable else "red"
    console.print(f"\n[bold {colour}]{report.summary()}[/bold {colour}]\n")
    for note in report.notes:
        console.print(f"  [yellow]•[/yellow] {note}")
    for stage_name, stage in (("survival", report.survival), ("conditional", report.conditional)):
        if stage is None:
            continue
        console.print(f"\n  [bold]{stage_name} stage[/bold]: {stage.summary()}")
        for note in stage.notes:
            console.print(f"    [yellow]•[/yellow] {note}")
        if stage.feature_importance:
            top = list(stage.feature_importance.items())[:8]
            console.print("    top features: " + ", ".join(f"{k}({v:.3f})" for k, v in top))

    if permutation and report.survival:
        console.print("\n  running permutation test on the survival stage…")
        result = scorer.survival.permutation_test(
            X, frame["y_survived"].to_numpy(dtype=int),
            frame["decision_ts"].to_numpy(dtype=float), frame["pool"].astype(str).tolist(),
        )
        console.print(
            f"    real AUC {result['real_auc']:.4f} vs null max {result['null_auc_max']:.4f}, "
            f"p={result['p_value']:.4f} → "
            f"{'[green]significant[/green]' if result['significant'] else '[red]not significant[/red]'}"
        )

    if report.usable:
        scorer.save(out)
        console.print(f"\n  saved to {out}")
    else:
        console.print("\n  [red]Model not saved: neither stage beats its baseline.[/red]")
        console.print("  [dim]This is a legitimate result, not an error. Collect more data "
                      "and retrain.[/dim]")
        raise typer.Exit(1)


@app.command()
def trade(
    db: str = typer.Option("data/alpha.db"),
    model: Optional[str] = typer.Option(None, help="Path to a trained model"),
    equity: float = typer.Option(10_000.0),
    min_score: float = typer.Option(0.32),
    cycles: Optional[int] = typer.Option(None, help="stop after N cycles"),
    live: bool = typer.Option(False, "--live", help="DANGER: trade real funds"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the trading loop. Paper mode unless --live is given."""
    _setup_logging(verbose)
    from alpha.execution.broker import PaperBroker
    from alpha.execution.trader import Trader, TraderConfig
    from alpha.risk.portfolio import PortfolioConfig

    if live:
        console.print("[bold red]Live trading is not implemented.[/bold red] "
                      "See docs/GOING_LIVE.md.")
        raise typer.Exit(1)

    scorer_fn = None
    if model and Path(model).exists():
        scorer_fn = _load_scorer(model)
        console.print(f"[dim]loaded model from {model}[/dim]")
    else:
        console.print("[yellow]No model supplied — running in observe-only mode. "
                      "Every candidate scores 0, so no positions will open.[/yellow]")

    config = TraderConfig(
        db_path=db, min_score=min_score,
        portfolio=PortfolioConfig(starting_equity_usd=equity),
    )
    trader = Trader(config, broker=PaperBroker(), scorer=scorer_fn)
    trader.install_signal_handlers()
    trader.run(max_cycles=cycles)
    console.print_json(json.dumps(trader.summary(), default=str))


@app.command()
def backtest(
    data: str = typer.Option("data/processed/dataset.csv"),
    db: str = typer.Option("data/alpha.db"),
    model: Optional[str] = typer.Option(None),
    equity: float = typer.Option(10_000.0),
    min_score: float = typer.Option(0.32),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Replay the strategy over collected history."""
    _setup_logging(verbose)
    import pandas as pd

    from alpha.backtest.engine import BacktestConfig, Backtester, Candidate
    from alpha.features.build import FEATURE_NAMES
    from alpha.risk.portfolio import PortfolioConfig

    frame = pd.read_csv(data)
    store = Store(db)
    features = [c for c in FEATURE_NAMES if c in frame.columns]

    if model and Path(model).exists():
        score_fn = _load_scorer(model)
    else:
        console.print("[yellow]No model — scoring every candidate at a constant 0.35 "
                      "to measure the strategy's structural edge without a model.[/yellow]")
        score_fn = lambda _f: 0.35  # noqa: E731

    candidates = []
    candles: dict[str, list] = {}
    for _, row in frame.iterrows():
        pool = str(row["pool"])
        if pool not in candles:
            candles[pool] = [dict(r) for r in store.candles_for(pool)]
        candidates.append(
            Candidate(
                pool=pool,
                decision_ts=int(row["decision_ts"]),
                features={f: float(row[f]) for f in features},
                liquidity_usd=float(row.get("liquidity_usd", 0.0) or 0.0),
                price_usd=float(row.get("price_usd", 0.0) or 0.0),
                dex=str(row.get("dex", "") or ""),
                symbol=str(row.get("symbol", "") or ""),
                mint=str(row.get("mint", "") or ""),
            )
        )

    result = Backtester(
        BacktestConfig(min_score=min_score, portfolio=PortfolioConfig(starting_equity_usd=equity))
    ).run(candidates, candles, score_fn)
    console.print(f"\n[bold]{result.summary()}[/bold]\n")
    if result.rejected:
        console.print("  rejections:")
        for reason, count in sorted(result.rejected.items(), key=lambda kv: -kv[1])[:10]:
            console.print(f"    {count:>5}  {reason}")


def _load_scorer(path: str):
    """Load either scorer type — two-stage is preferred, single-stage still works."""
    from alpha.models.scorer import Scorer
    from alpha.models.two_stage import TwoStageScorer

    try:
        return TwoStageScorer.load(path).score_features
    except (KeyError, AttributeError, TypeError):
        return Scorer.load(path).score_features


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main() or 0)
