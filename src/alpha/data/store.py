"""SQLite storage for the token panel dataset.

The central asset this system builds is a *prospectively recorded* panel: every
new Solana pool is written down at birth and re-observed on a decaying schedule.
Because tokens are recorded before we know whether they succeed, the resulting
dataset is free of the survivorship bias that ruins retrospective memecoin
studies — dead tokens are still in it, which is exactly what a model needs in
order to learn what death looks like.

SQLite is used deliberately: the workload is a single writer plus occasional
analytical readers, the dataset is millions of rows rather than billions, and
zero operational overhead matters more than throughput. WAL mode lets readers
run while the collector writes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

-- One row per pool, written once at first discovery. This is the "birth
-- certificate": the state of the token when we first saw it, before any
-- outcome is known.
CREATE TABLE IF NOT EXISTS pools (
    pool            TEXT PRIMARY KEY,
    base_mint       TEXT NOT NULL,
    quote_mint      TEXT,
    name            TEXT,
    dex             TEXT,
    created_at      TEXT,
    discovered_at   TEXT NOT NULL,
    discovery_age_min REAL,
    birth_liquidity_usd REAL,
    birth_fdv_usd   REAL,
    birth_price_usd REAL,
    is_sol_quoted   INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pools_discovered ON pools(discovered_at);
CREATE INDEX IF NOT EXISTS idx_pools_created ON pools(created_at);
CREATE INDEX IF NOT EXISTS idx_pools_mint ON pools(base_mint);

-- Time series of observations. One row per (pool, observation time).
CREATE TABLE IF NOT EXISTS snapshots (
    pool            TEXT NOT NULL,
    observed_at     TEXT NOT NULL,
    age_min         REAL,
    price_usd       REAL,
    price_native    REAL,
    fdv_usd         REAL,
    liquidity_usd   REAL,
    buys_m5         INTEGER, sells_m5   INTEGER, buyers_m5  INTEGER, sellers_m5 INTEGER,
    vol_m5          REAL,    chg_m5     REAL,
    buys_m15        INTEGER, sells_m15  INTEGER, buyers_m15 INTEGER, sellers_m15 INTEGER,
    vol_m15         REAL,    chg_m15    REAL,
    buys_h1         INTEGER, sells_h1   INTEGER, buyers_h1  INTEGER, sellers_h1 INTEGER,
    vol_h1          REAL,    chg_h1     REAL,
    buys_h24        INTEGER, sells_h24  INTEGER, buyers_h24 INTEGER, sellers_h24 INTEGER,
    vol_h24         REAL,    chg_h24    REAL,
    PRIMARY KEY (pool, observed_at),
    FOREIGN KEY (pool) REFERENCES pools(pool) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_snap_pool ON snapshots(pool, observed_at);
CREATE INDEX IF NOT EXISTS idx_snap_time ON snapshots(observed_at);

-- Tracking scheduler state: when is this pool next due for an observation.
CREATE TABLE IF NOT EXISTS tracking (
    pool            TEXT PRIMARY KEY,
    next_due_at     TEXT NOT NULL,
    observations    INTEGER DEFAULT 0,
    last_observed_at TEXT,
    retired         INTEGER DEFAULT 0,
    retire_reason   TEXT,
    FOREIGN KEY (pool) REFERENCES pools(pool) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_track_due ON tracking(retired, next_due_at);

-- Individual swaps, used for wallet attribution and microstructure features.
-- The primary key deliberately excludes volume_usd: the API returns
-- high-precision decimal strings that round inconsistently between fetches, so
-- including a float in the key caused the same swap to be stored repeatedly and
-- silently inflated every volume feature computed from this table.
CREATE TABLE IF NOT EXISTS trades (
    tx_hash         TEXT NOT NULL,
    pool            TEXT NOT NULL,
    wallet          TEXT,
    block_number    INTEGER,
    ts              TEXT,
    kind            TEXT,
    volume_usd      REAL,
    price_usd       REAL,
    PRIMARY KEY (tx_hash, pool, kind)
);
CREATE INDEX IF NOT EXISTS idx_trades_pool ON trades(pool, ts);
CREATE INDEX IF NOT EXISTS idx_trades_wallet ON trades(wallet, ts);

-- Cached OHLCV candles for outcome labelling and backtesting.
CREATE TABLE IF NOT EXISTS candles (
    pool            TEXT NOT NULL,
    ts              INTEGER NOT NULL,
    open            REAL, high REAL, low REAL, close REAL, volume_usd REAL,
    PRIMARY KEY (pool, ts)
);

-- Computed outcomes: what actually happened to each token. Written by the
-- labeller after enough time has passed.
CREATE TABLE IF NOT EXISTS outcomes (
    pool              TEXT PRIMARY KEY,
    labelled_at       TEXT NOT NULL,
    horizon_min       INTEGER,
    entry_price       REAL,
    max_price         REAL,
    min_price         REAL,
    final_price       REAL,
    max_multiple      REAL,
    max_drawdown      REAL,
    minutes_to_peak   REAL,
    survived          INTEGER,
    is_rug            INTEGER,
    n_candles         INTEGER,
    FOREIGN KEY (pool) REFERENCES pools(pool) ON DELETE CASCADE
);

-- Free-form key/value blobs (safety reports, social data, model metadata).
CREATE TABLE IF NOT EXISTS artifacts (
    kind            TEXT NOT NULL,
    key             TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    payload         TEXT NOT NULL,
    PRIMARY KEY (kind, key)
);

-- Token creations observed live on the PumpPortal websocket, recorded at t=0.
-- This is the low-latency cohort: the polling collector sees pools at 1-15
-- minutes old, which measurement showed is already too late for large
-- multiples. These rows are written within seconds of deployment.
CREATE TABLE IF NOT EXISTS launches (
    mint            TEXT PRIMARY KEY,
    name            TEXT,
    symbol          TEXT,
    uri             TEXT,
    dev_wallet      TEXT,
    signature       TEXT,
    bonding_curve   TEXT,
    pool_kind       TEXT,
    dev_buy_sol     REAL,
    dev_tokens      REAL,
    dev_curve_share REAL,
    v_sol           REAL,
    v_tokens        REAL,
    market_cap_sol  REAL,
    observed_at     TEXT NOT NULL,
    -- Filled in later, once the token is matched to a tradeable pool.
    pool            TEXT,
    graduated_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_launch_dev ON launches(dev_wallet);
CREATE INDEX IF NOT EXISTS idx_launch_time ON launches(observed_at);
CREATE INDEX IF NOT EXISTS idx_launch_pool ON launches(pool);

-- Graduations to PumpSwap. Rare, and the gateway to multiples above the
-- 14.696x bonding-curve ceiling.
CREATE TABLE IF NOT EXISTS migrations (
    mint            TEXT PRIMARY KEY,
    signature       TEXT,
    pool            TEXT,
    observed_at     TEXT NOT NULL
);

-- Deployer track record, accumulated first-hand from the launch stream rather
-- than bought. Deployer history is the strongest free rug signal available.
CREATE TABLE IF NOT EXISTS dev_wallets (
    dev_wallet      TEXT PRIMARY KEY,
    first_seen      TEXT,
    last_seen       TEXT,
    launches        INTEGER DEFAULT 0,
    graduations     INTEGER DEFAULT 0,
    total_dev_buy_sol REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

SNAPSHOT_TIMEFRAMES = ("m5", "m15", "h1", "h24")
_SNAP_COLS = ["pool", "observed_at", "age_min", "price_usd", "price_native", "fdv_usd", "liquidity_usd"]
for _tf in SNAPSHOT_TIMEFRAMES:
    _SNAP_COLS += [f"buys_{_tf}", f"sells_{_tf}", f"buyers_{_tf}", f"sellers_{_tf}", f"vol_{_tf}", f"chg_{_tf}"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


class Store:
    """Thread-safe SQLite wrapper for the panel dataset."""

    def __init__(self, path: str | Path = "data/alpha.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self.connect() as con:
            con.executescript(SCHEMA)
            self._migrate(con)
            con.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _migrate(con: sqlite3.Connection) -> None:
        """Apply in-place schema fixes to databases created by older versions."""
        cols = con.execute("PRAGMA index_list('trades')").fetchall()
        pk_cols: list[str] = []
        for idx in cols:
            name = idx["name"] if hasattr(idx, "keys") else idx[1]
            origin = idx["origin"] if hasattr(idx, "keys") else idx[3]
            if origin == "pk":
                pk_cols = [
                    (r["name"] if hasattr(r, "keys") else r[2])
                    for r in con.execute(f"PRAGMA index_info('{name}')").fetchall()
                ]
        if "volume_usd" in pk_cols:
            log.warning("migrating trades table: dropping volume_usd from primary key")
            con.executescript(
                """
                CREATE TABLE trades_new (
                    tx_hash TEXT NOT NULL, pool TEXT NOT NULL, wallet TEXT,
                    block_number INTEGER, ts TEXT, kind TEXT,
                    volume_usd REAL, price_usd REAL,
                    PRIMARY KEY (tx_hash, pool, kind)
                );
                INSERT OR IGNORE INTO trades_new
                    SELECT tx_hash, pool, wallet, block_number, ts, kind, volume_usd, price_usd
                    FROM trades;
                DROP TABLE trades;
                ALTER TABLE trades_new RENAME TO trades;
                CREATE INDEX IF NOT EXISTS idx_trades_pool ON trades(pool, ts);
                CREATE INDEX IF NOT EXISTS idx_trades_wallet ON trades(wallet, ts);
                """
            )

    @property
    def conn(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            con.execute("PRAGMA busy_timeout=30000")
            self._local.con = con
        return con

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Run a block in a transaction.

        ``executescript`` and DDL implicitly commit, so the transaction may
        already be closed by the time we get here; ``in_transaction`` is checked
        rather than assumed. Nested use reuses the outer transaction.
        """
        con = self.conn
        if con.in_transaction:
            yield con  # already inside a transaction; let the outermost commit
            return
        try:
            con.execute("BEGIN")
            yield con
            if con.in_transaction:
                con.execute("COMMIT")
        except Exception:
            if con.in_transaction:
                try:
                    con.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
            raise

    # ------------------------------------------------------------------ writes

    def record_pools(self, pools: Sequence[Any]) -> int:
        """Insert newly discovered pools. Existing pools are left untouched so
        the birth certificate is never overwritten by a later observation."""
        if not pools:
            return 0
        now = iso(utcnow())
        rows = [
            (
                p.address,
                p.base_mint,
                p.quote_mint,
                p.name,
                p.dex,
                iso(p.created_at),
                now,
                round(p.age_minutes, 3) if p.created_at else None,
                p.liquidity_usd,
                p.fdv_usd,
                p.price_usd,
                int(p.is_sol_quoted),
            )
            for p in pools
            if p.address
        ]
        with self.connect() as con:
            before = con.execute("SELECT COUNT(*) FROM pools").fetchone()[0]
            con.executemany(
                """INSERT OR IGNORE INTO pools
                   (pool, base_mint, quote_mint, name, dex, created_at, discovered_at,
                    discovery_age_min, birth_liquidity_usd, birth_fdv_usd, birth_price_usd, is_sol_quoted)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            after = con.execute("SELECT COUNT(*) FROM pools").fetchone()[0]
        return after - before

    def record_snapshots(self, pools: Sequence[Any]) -> int:
        if not pools:
            return 0
        rows = []
        for p in pools:
            if not p.address:
                continue
            vals: list[Any] = [
                p.address,
                iso(p.observed_at),
                round(p.age_minutes, 3) if p.created_at else None,
                p.price_usd,
                p.price_native,
                p.fdv_usd,
                p.liquidity_usd,
            ]
            for tf in SNAPSHOT_TIMEFRAMES:
                t = p.tf(tf)
                vals += [t.buys, t.sells, t.buyers, t.sellers, t.volume_usd, t.price_change_pct]
            rows.append(tuple(vals))
        placeholders = ",".join("?" * len(_SNAP_COLS))
        with self.connect() as con:
            con.executemany(
                f"INSERT OR REPLACE INTO snapshots ({','.join(_SNAP_COLS)}) VALUES ({placeholders})",
                rows,
            )
        return len(rows)

    def record_trades(self, pool: str, trades: Sequence[Any]) -> int:
        if not trades:
            return 0
        rows = [
            (t.tx_hash, pool, t.wallet, t.block_number, iso(t.timestamp), t.kind, t.volume_usd, t.price_usd)
            for t in trades
            if t.tx_hash
        ]
        with self.connect() as con:
            con.executemany(
                """INSERT OR IGNORE INTO trades
                   (tx_hash, pool, wallet, block_number, ts, kind, volume_usd, price_usd)
                   VALUES (?,?,?,?,?,?,?,?)""",
                rows,
            )
        return len(rows)

    def record_candles(self, pool: str, candles: Sequence[Any]) -> int:
        if not candles:
            return 0
        rows = [(pool, c.ts, c.open, c.high, c.low, c.close, c.volume_usd) for c in candles]
        with self.connect() as con:
            con.executemany(
                "INSERT OR REPLACE INTO candles (pool, ts, open, high, low, close, volume_usd) VALUES (?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def schedule(self, pool: str, next_due: datetime, *, increment: bool = False) -> None:
        with self.connect() as con:
            if increment:
                con.execute(
                    """UPDATE tracking
                       SET next_due_at=?, observations=observations+1, last_observed_at=?
                       WHERE pool=?""",
                    (iso(next_due), iso(utcnow()), pool),
                )
            else:
                con.execute(
                    "INSERT OR IGNORE INTO tracking (pool, next_due_at) VALUES (?, ?)",
                    (pool, iso(next_due)),
                )

    def schedule_many(self, pools: Sequence[str], next_due: datetime) -> None:
        if not pools:
            return
        due = iso(next_due)
        with self.connect() as con:
            con.executemany(
                "INSERT OR IGNORE INTO tracking (pool, next_due_at) VALUES (?, ?)",
                [(p, due) for p in pools],
            )

    def mark_observed(self, updates: Sequence[tuple[str, datetime]]) -> None:
        """Bump observation count and set the next due time, in one batch."""
        if not updates:
            return
        now = iso(utcnow())
        with self.connect() as con:
            con.executemany(
                """UPDATE tracking SET next_due_at=?, observations=observations+1, last_observed_at=?
                   WHERE pool=?""",
                [(iso(due), now, pool) for pool, due in updates],
            )

    def retire(self, pools: Sequence[str], reason: str) -> None:
        if not pools:
            return
        with self.connect() as con:
            con.executemany(
                "UPDATE tracking SET retired=1, retire_reason=? WHERE pool=?",
                [(reason, p) for p in pools],
            )

    def put_artifact(self, kind: str, key: str, payload: Any) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO artifacts (kind, key, created_at, payload) VALUES (?,?,?,?)",
                (kind, key, iso(utcnow()), json.dumps(payload, default=str)),
            )

    def get_artifact(self, kind: str, key: str) -> Any | None:
        row = self.conn.execute(
            "SELECT payload FROM artifacts WHERE kind=? AND key=?", (kind, key)
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def record_outcome(self, pool: str, outcome: dict[str, Any]) -> None:
        with self.connect() as con:
            con.execute(
                """INSERT OR REPLACE INTO outcomes
                   (pool, labelled_at, horizon_min, entry_price, max_price, min_price, final_price,
                    max_multiple, max_drawdown, minutes_to_peak, survived, is_rug, n_candles)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    pool,
                    iso(utcnow()),
                    outcome.get("horizon_min"),
                    outcome.get("entry_price"),
                    outcome.get("max_price"),
                    outcome.get("min_price"),
                    outcome.get("final_price"),
                    outcome.get("max_multiple"),
                    outcome.get("max_drawdown"),
                    outcome.get("minutes_to_peak"),
                    int(bool(outcome.get("survived"))),
                    int(bool(outcome.get("is_rug"))),
                    outcome.get("n_candles"),
                ),
            )

    # ------------------------------------------------------------------- reads

    def due_pools(self, limit: int = 450, now: datetime | None = None) -> list[str]:
        """Pools whose next observation is due, oldest-due first."""
        cutoff = iso(now or utcnow())
        rows = self.conn.execute(
            """SELECT pool FROM tracking
               WHERE retired=0 AND next_due_at <= ?
               ORDER BY next_due_at ASC LIMIT ?""",
            (cutoff, limit),
        ).fetchall()
        return [r["pool"] for r in rows]

    def observation_counts(self, pools: Sequence[str]) -> dict[str, int]:
        if not pools:
            return {}
        marks = ",".join("?" * len(pools))
        rows = self.conn.execute(
            f"SELECT pool, observations FROM tracking WHERE pool IN ({marks})", tuple(pools)
        ).fetchall()
        return {r["pool"]: r["observations"] for r in rows}

    def snapshots_for(self, pool: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM snapshots WHERE pool=? ORDER BY observed_at ASC", (pool,)
        ).fetchall()

    def candles_for(self, pool: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM candles WHERE pool=? ORDER BY ts ASC", (pool,)
        ).fetchall()

    def pool_row(self, pool: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM pools WHERE pool=?", (pool,)).fetchone()

    def pools_needing_labels(self, min_age_min: float = 360.0, limit: int = 500) -> list[str]:
        """Pools old enough that their outcome is settled but not yet labelled."""
        rows = self.conn.execute(
            """SELECT p.pool FROM pools p
               LEFT JOIN outcomes o ON o.pool = p.pool
               WHERE o.pool IS NULL
                 AND p.created_at IS NOT NULL
                 AND (julianday('now') - julianday(p.created_at)) * 1440.0 >= ?
               ORDER BY p.created_at ASC LIMIT ?""",
            (min_age_min, limit),
        ).fetchall()
        return [r["pool"] for r in rows]

    def stats(self) -> dict[str, Any]:
        c = self.conn
        def one(sql: str, *a: Any) -> Any:
            r = c.execute(sql, a).fetchone()
            return r[0] if r else 0
        return {
            "pools": one("SELECT COUNT(*) FROM pools"),
            "sol_quoted": one("SELECT COUNT(*) FROM pools WHERE is_sol_quoted=1"),
            "snapshots": one("SELECT COUNT(*) FROM snapshots"),
            "trades": one("SELECT COUNT(*) FROM trades"),
            "wallets": one("SELECT COUNT(DISTINCT wallet) FROM trades"),
            "candles": one("SELECT COUNT(*) FROM candles"),
            "outcomes": one("SELECT COUNT(*) FROM outcomes"),
            "tracking_active": one("SELECT COUNT(*) FROM tracking WHERE retired=0"),
            "tracking_retired": one("SELECT COUNT(*) FROM tracking WHERE retired=1"),
            "earliest": one("SELECT MIN(discovered_at) FROM pools"),
            "latest": one("SELECT MAX(discovered_at) FROM pools"),
            "db_mb": round(self.path.stat().st_size / 1e6, 2) if self.path.exists() else 0,
        }

    # ------------------------------------------------------- launch stream

    def record_launch(self, launch: Any) -> bool:
        """Persist a launch and update the deployer's running record.

        Returns True if this mint was new. The deployer counters are updated in
        the same transaction so the track record can never drift from the
        launch table.
        """
        row = launch.to_row()
        with self.connect() as con:
            cur = con.execute(
                """INSERT OR IGNORE INTO launches
                   (mint, name, symbol, uri, dev_wallet, signature, bonding_curve, pool_kind,
                    dev_buy_sol, dev_tokens, dev_curve_share, v_sol, v_tokens, market_cap_sol,
                    observed_at)
                   VALUES (:mint,:name,:symbol,:uri,:dev_wallet,:signature,:bonding_curve,
                           :pool_kind,:dev_buy_sol,:dev_tokens,:dev_curve_share,:v_sol,
                           :v_tokens,:market_cap_sol,:observed_at)""",
                row,
            )
            is_new = cur.rowcount > 0
            if is_new and row["dev_wallet"]:
                con.execute(
                    """INSERT INTO dev_wallets (dev_wallet, first_seen, last_seen, launches, total_dev_buy_sol)
                       VALUES (?,?,?,1,?)
                       ON CONFLICT(dev_wallet) DO UPDATE SET
                           last_seen = excluded.last_seen,
                           launches = launches + 1,
                           total_dev_buy_sol = total_dev_buy_sol + excluded.total_dev_buy_sol""",
                    (row["dev_wallet"], row["observed_at"], row["observed_at"], row["dev_buy_sol"]),
                )
        return is_new

    def record_migration(self, migration: Any) -> bool:
        with self.connect() as con:
            cur = con.execute(
                "INSERT OR IGNORE INTO migrations (mint, signature, pool, observed_at) VALUES (?,?,?,?)",
                (migration.mint, migration.signature, migration.pool, iso(migration.observed_at)),
            )
            is_new = cur.rowcount > 0
            if is_new:
                con.execute(
                    "UPDATE launches SET graduated_at=? WHERE mint=?",
                    (iso(migration.observed_at), migration.mint),
                )
                con.execute(
                    """UPDATE dev_wallets SET graduations = graduations + 1
                       WHERE dev_wallet = (SELECT dev_wallet FROM launches WHERE mint=?)""",
                    (migration.mint,),
                )
        return is_new

    def dev_record(self, dev_wallet: str) -> dict[str, Any] | None:
        """Our own first-hand record for a deployer."""
        row = self.conn.execute(
            "SELECT * FROM dev_wallets WHERE dev_wallet=?", (dev_wallet,)
        ).fetchone()
        return dict(row) if row else None

    def link_launches_to_pools(self) -> int:
        """Match stream-observed launches to discovered pools by mint.

        Costs nothing: the collector already records ``base_mint`` for every
        pool, so the join is local. This is what makes the t=0 cohort
        measurable — the launch supplies the deployer, bundle size and exact
        curve state at creation, and the pool supplies the subsequent price
        path.
        """
        with self.connect() as con:
            cur = con.execute(
                """UPDATE launches
                   SET pool = (SELECT p.pool FROM pools p WHERE p.base_mint = launches.mint)
                   WHERE pool IS NULL
                     AND EXISTS (SELECT 1 FROM pools p WHERE p.base_mint = launches.mint)"""
            )
            return cur.rowcount

    def link_launch_to_pool(self, mint: str, pool: str) -> None:
        with self.connect() as con:
            con.execute("UPDATE launches SET pool=? WHERE mint=? AND pool IS NULL", (pool, mint))

    def launch_stats(self) -> dict[str, Any]:
        c = self.conn
        def one(sql: str) -> Any:
            r = c.execute(sql).fetchone()
            return r[0] if r else 0
        return {
            "launches": one("SELECT COUNT(*) FROM launches"),
            "migrations": one("SELECT COUNT(*) FROM migrations"),
            "dev_wallets": one("SELECT COUNT(*) FROM dev_wallets"),
            "repeat_devs": one("SELECT COUNT(*) FROM dev_wallets WHERE launches > 1"),
            "linked_to_pool": one("SELECT COUNT(*) FROM launches WHERE pool IS NOT NULL"),
            "graduation_rate": round(
                (one("SELECT COUNT(*) FROM launches WHERE graduated_at IS NOT NULL")
                 / max(1, one("SELECT COUNT(*) FROM launches"))), 5
            ),
        }

    def close(self) -> None:
        con = getattr(self._local, "con", None)
        if con is not None:
            con.close()
            self._local.con = None
