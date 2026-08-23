# Measured findings

Everything here was measured from live Solana data on 2026-08-23, not taken from
secondary sources. Where a number is an estimate or an assumption, it says so.

---

## 1. The universe

| Measurement | Value | How |
|---|---|---|
| New pools created on Solana | **~14 per minute (~20,000/day)** | 200 pools observed across 10 `new_pools` pages spanning 14 minutes |
| Discovery window of the API | ~15 minutes across 10 pages | page 1 ≈ 1–3 min old, page 10 ≈ 15 min old |
| Median liquidity at birth | **$2,092** | 200-pool sample |
| Pools with < $300 liquidity | 18% | same sample |
| Pools with **zero** buys in their first 5 minutes | **33%** | same sample |

The practical consequence: polling all ten pages once a minute captures
essentially the entire new-pool universe with a wide safety margin. That is the
whole basis of the unbiased panel — the firehose is small enough to drink from
completely, for free.

## 2. What a memecoin's life actually looks like

The clearest single observation from the sample, a pool with $2 of liquidity
left:

```
maxmult = 36.35x        now_vs_high = -94.6%
```

A 36x move and a 95% collapse, complete, within minutes of launch. This is the
shape of the problem, and it drives three design decisions:

- **Path matters more than destination.** A token that reaches +300% in minute
  two and ends at −95% is a win for a strategy that takes profit and a
  catastrophe for one that holds. Labelling on terminal return alone would
  conflate the two, so labelling is done with barriers, not endpoints.
- **Exits must be evaluated before entries** on every cycle.
- **Absence of data is a label, not a gap.** A pool that stops trading has not
  merely stopped being observed. It went to zero.

## 2a. A hard ceiling nobody mentions

The pump.fun bonding curve has fixed, public launch constants, so its entire
pre-graduation price path is determined by one number: net SOL bought.
Graduation occurs at exactly **85.0054 SOL**, at which point the virtual reserves
have moved 30 → 115.0054 SOL and 1.073B → 279.9M tokens.

Since price ∝ vSol/vTokens, that fixes the maximum pre-graduation appreciation:

```
MAX_PREGRAD_MULTIPLE = 14.696x
```

**A token that never graduates cannot have risen more than 14.7x.** Verified
independently in `alpha.data.bondingcurve` (market-cap inversion round-trips to
1e-6). Two consequences:

- A strategy hunting 50x or 100x on the curve is hunting something structurally
  impossible there. Those returns exist only post-migration.
- A +150% take-profit sits comfortably inside the achievable range, which is why
  the default barrier geometry is set where it is rather than by preference.

Price is also *continuous* across migration while depth is not — the PumpSwap
pool receives materially less depth than the curve quoted against. There is no
"graduation pop" to hold for, only a step down in liquidity that raises the cost
of every later exit.

## 3. Execution costs — the hurdle

Modelled from the constant-product formula, with fees, priority fee, Jito tip
and ATA rent included. Round-trip cost as a percentage of notional:

| Order size | $5k pool | $15k pool | $30k pool | $100k pool | $500k pool |
|---|---|---|---|---|---|
| $50 | 8.9% | 6.2% | 5.6% | 5.1% | 5.0% |
| $100 | 12.0% | 6.6% | 5.3% | 4.4% | 4.0% |
| $250 | reject | 10.1% | 6.7% | 4.4% | 3.6% |
| $500 | reject | reject | 9.9% | 5.2% | 3.6% |
| $1,000 | reject | reject | reject | 7.1% | 3.9% |

**Cost is U-shaped in order size.** Fixed network costs (~$0.64 per round trip)
dominate small orders — a $10 order pays 6.4% in fees alone — while AMM price
impact dominates large ones. Minimising total cost analytically gives:

```
A* = sqrt(network_fee × L / 2)
```

which is $98 into a $30k pool, $179 into $100k, $400 into $500k. **Position size
should scale with the square root of pool liquidity**, not as a fixed dollar
amount. This is implemented in `alpha.risk.sizing.optimal_order_usd` and matches
the empirical minima of the table above.

### The break-even win rate

With a +150% take-profit and −45% stop:

| Round-trip cost | Required win rate |
|---|---|
| 0% | 23.1% |
| 4% | 25.1% |
| 6% | **26.2%** |
| 10% | 28.2% |

**A strategy must win more than about one trade in four to break even.** Every
claim of profitability reduces to whether the model clears this line.

## 4. Safety screening

Screened 45 randomly-selected live tokens:

| Verdict | Share | Meaning |
|---|---|---|
| PASS | 13% | tradeable now |
| **IMMATURE** | **69%** | young, not unsafe — re-screen later |
| REJECT | 18% | structurally unsafe, never buy |

The structural/maturity split matters more than it looks. A first implementation
that treated every failed check as a rejection discarded 90% of the universe,
because RugCheck flags "Low Liquidity" and "LP Unlocked" as `danger` — conditions
that are *by design* true of every new bonding-curve token. Those are maturity
states, not malice. Only 18% of tokens are genuinely disqualified.

Structural rejects, in order of frequency: single-holder concentration (13%),
deployer rug history (4%), serial deployer (4%).

The strongest single signal available for free is **deployer history**: RugCheck
exposes `creatorTokens`, and the observed distribution is median 0, p75 = 1,
max = 50. A wallet that has launched fifty tokens is running a factory.

## 5. Wash trading is pervasive and detectable

Per-wallet profiling of collected swaps found this pattern repeatedly:

```
17 wallets clustering at ~91 trades each, each active in exactly one pool,
each with a near-identical buy/sell split, together accounting for 100% of volume
```

Measured wash scores on six live pools with ≥40 trades:

| Pool | Trades | Wallets | Wash score | Suspect volume |
|---|---|---|---|---|
| FTiQ15…| 1,800 | 220 | 0.72 | **97.2%** |
| HkvrLF…| 600 | 422 | 0.83 | 71.4% |
| 3c9dt3…| 600 | 102 | 0.71 | 58.7% |
| 8rjptc…| 300 | 231 | 0.00 | — |
| AEogVd…| 300 | 135 | 0.00 | — |
| CQYQUh…| 300 | 124 | 0.00 | — |

**On the worst pool, $100k of reported volume is really $2,750.** Volume and
transaction-count filters — the ones every published strategy screens on — are
precisely what this activity is manufactured to satisfy. Any feature derived
from raw volume must be discounted by the suspect share first.

## 6. Data sources that actually work, for free, with no key

Verified live from this environment:

| Source | Key needed | What it gives |
|---|---|---|
| GeckoTerminal `new_pools` | no | full discovery, 10 pages, ~15 min window |
| GeckoTerminal `pools/multi` | no | **30 pools per call** — the key efficiency lever |
| GeckoTerminal `ohlcv/minute` | no | 1000 candles/call, paginates back indefinitely, **survives for dead pools** |
| GeckoTerminal `trades` | no | last ~300 swaps **including the trader's wallet** |
| RugCheck `/v1/tokens/{mint}/report` | no | deployer history, insider flags, LP locks, holder distribution |
| Solana public RPC `getAccountInfo` | no | mint/freeze authority, Token-2022 extensions |

Two of these are more valuable than they look:

- **OHLCV survives for dead pools.** This is what makes an honest backtest
  possible at all: outcomes can be reconstructed for tokens that already died.
- **`trades` exposes wallets.** Free wallet-level attribution, which is what
  section 5 is built on. Most systems pay for this.

Rate limit is ~30 requests/minute. Exceeding it returns `403`, and so does
sending a default Python User-Agent — both are handled in `alpha.http`.

Not usable: `frontend-api.pump.fun` (HTTP 530), `quote-api.jup.ag` (blocked at
the proxy; Jupiter has moved base URLs).

## 7. X / social — what the research changed

The plan to use Grok as the X data source does not survive contact with the
current API, for a structural reason rather than a quality one:

- The `search_parameters` Live Search API was **removed** — it now returns
  HTTP 410, even with `mode: "off"`, even unauthenticated.
- Its replacement, `x_search` on `/v1/responses`, **never returns server-side
  tool outputs**. Grok reads the posts and returns prose plus bare citation
  URLs. There is no parameter that returns post text, author handles, follower
  counts, or timestamps.
- Date filters are **day-granularity only**. "Last hour" cannot be expressed.
- Third-party reports put agentic search latency at **60–120 seconds**, which is
  longer than many memecoins exist.

So any count Grok states was generated, not computed — precision is not degraded
in that channel, it is structurally absent. The system therefore splits the two
roles:

- **Counting** → deterministic providers. Elfa is preferred because it keys on
  *contract addresses* rather than tickers, which sidesteps symbol collisions
  and impersonation entirely.
- **Judging** → Grok, restricted to a boolean veto, off the hot path.

The veto restriction is not fussiness. In May 2026 an attacker drained roughly
$150k from a Grok-linked trading wallet by posting a Morse-coded instruction
that a bot treated as an authenticated command. `x_search` ingests
attacker-controlled text by construction. A boolean veto bounds the damage of a
successful injection to a skipped trade; a number the model can influence would
let an attacker size the book.

On detecting manufactured attention: **structural signals outrank textual ones.**
Modern campaigns write varied copy with an LLM, which defeats text-similarity
detection outright. Account creation dates, smart-follower ratios and posting
burst structure are far more expensive to fake. The implementation treats two
structural flags as conclusive regardless of the additive score — an additive
threshold is exactly what lets the most sophisticated campaigns through.

## 8. The most important result: avoiding death beats picking winners

First end-to-end run on collected data (159 decision points, 28 pools, 30-minute
horizon). The sample is far too small to conclude anything about edge, but it is
large enough to reveal the *shape* of the problem, and the shape is decisive.

**The outcome distribution is bimodal, not continuous:**

| Bucket | Count | Share | Mean return |
|---|---|---|---|
| Wins | 45 | 30.2% | **+71.9%** |
| Small losses | 57 | 38.3% | −9.9% |
| **Total losses** | **47** | **31.5%** | **−100.0%** |

Barrier breakdown: `time` 82, **`no_data` 47**, `take_profit` 14, `stop_loss` **6**.

The stop-loss fired **six times out of 159**. The dominant loss mode — 31.5% of
all trades — is `no_data`: the token stopped trading entirely. **A stop-loss
offers no protection against that.** There is no bid to sell into. You do not
lose 45%, you lose everything.

### What that is worth

| Strategy | Expectancy per trade |
|---|---|
| Buy indiscriminately | **−13.6%** |
| Buy only survivors (perfect filter) | **+26.2%** |
| Buy only eventual winners (perfect filter) | +71.9% |

Both perfect filters are unattainable, but they degrade very differently:

| Filter catches this share of deaths | Expectancy |
|---|---|
| 0% | −13.6% |
| 25% | −6.2% |
| **50%** | **+2.6%** ← profitable |
| 75% | +13.1% |
| 90% | +20.6% |

**A filter that catches roughly half the total-loss cases is enough to flip this
sample positive.** A survival filter only has to be approximately right, because
it removes a −100% tail; a return forecaster has to be accurate before it is
worth anything.

### Consequence for the model

The scorer is therefore decomposed as

    P(win) = P(survive) · P(win | survive)

with each stage trained, calibrated and gated independently
(`alpha.models.two_stage`). The second stage is trained *only on survivors*, so
the two terms do not multiply in the same information twice.

This decomposition also degrades gracefully. On the first real run the survival
stage reached **AUC 0.705** while the conditional stage sat at 0.500 — i.e. all
the available signal was in predicting survival and none in ranking survivors.
A single-target model would have reported one mediocre number and obscured that.
Where survival clears its gate and the conditional stage does not, the system
uses the empirical conditional base rate for the second term and says so, rather
than discarding a genuine edge.

*(Neither stage cleared its significance gate on this sample — 28 pools produced
too few purged folds to bound the AUC. See section 9.)*

### Consequence for exits

Because the stop-loss is not protective against the dominant loss mode, two
path-dependent rules were added (`alpha.risk.exits`):

- **Dump detection** on a 4-σ robust outlier in log-returns, using median
  absolute deviation rather than standard deviation. This matters: on a
  contaminated series, one outlier inflates the standard deviation **48.8×**
  while MAD moves **1.33×**. A σ-based threshold widens precisely when it needs
  to stay tight. Published analysis finds 92% of pump.fun tokens with ≥30 swaps
  suffer at least one such dump.
- **Liquidity-withdrawal detection**, checked *before* price. A draining pool is
  the mechanical precursor of a rug: price may still look healthy while the
  ability to exit at size has already gone.

## 9. What is not yet known

Honest status of the central question — *does the model have edge?*

**Not yet answered.** The panel is hours old at the time of writing. The
machinery to answer it is built and tested:

- purged, pool-grouped, walk-forward cross-validation
- isotonic calibration, because Kelly consumes the probability directly
- honesty gates requiring mean AUC ≥ 0.55, a confidence bound above 0.52, ≥ 2%
  relative Brier improvement, and ≥ 60% of folds beating 0.5
- a permutation test comparing the real AUC against models fitted to shuffled
  labels

Those gates are calibrated: an earlier, laxer version (mean AUC > 0.52 and any
Brier improvement) **passed a model trained on randomly shuffled labels**. The
current gates reject it with explicit reasons. That failure is why the gates
look paranoid — they were tuned against a known-null case, not against a hope.

The system will report "NO EDGE — do not trade" and refuse to save a model that
does not clear them. That outcome is a legitimate result, not a bug.
