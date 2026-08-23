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


---

## 10. Where the 10x–30x actually lives: entry timing, not filtering

This is the answer to "how do people get 20x?", measured on our own panel
(30 pools with ≥8 minutes of minute-candle history — a small sample, but the
pattern is monotonic and very strong).

**Best multiple achievable, by how late you enter:**

| Entry | median | p90 | max observed | reached ≥2x | ≥5x | ≥10x |
|---|---|---|---|---|---|---|
| **launch candle** | 2.17x | 20.4x | **318x** | 56.7% | 33.3% | **20.0%** |
| +1 min | 1.74x | 8.6x | 12.3x | 43.3% | 26.7% | 6.7% |
| +2 min | 1.65x | 7.1x | 7.3x | 43.3% | 23.3% | **0%** |
| +3 min | 1.58x | 5.7x | 6.0x | 36.7% | 23.3% | 0% |
| +5 min | 1.44x | 4.5x | 4.6x | 36.7% | **0%** | 0% |
| +15 min | 1.33x | 4.1x | 4.9x | 29.2% | 0% | 0% |
| +30 min | 1.17x | 2.1x | 2.2x | 10.0% | 0% | 0% |

**The entire 10x opportunity is gone within two minutes.** From a 2-minute
entry the best outcome in the whole sample was 7.3x; from 5 minutes, 4.6x; from
30 minutes, 2.2x.

This settles a question that filter-based strategies implicitly get wrong. The
big multiples are not a filtering achievement — they are an *entry-timing*
achievement. No screening rule applied at minute five can recover a 10x, because
by minute five the 10x no longer exists in the price path. Filters change *which*
of the available outcomes you get; they cannot change what is available.

Combined with the 14.696x bonding-curve ceiling (§2a), the decomposition of any
claimed 30x is:

1. **Bought in the launch block or first seconds** — requires sniper
   infrastructure, or being the deployer.
2. **Held through graduation into PumpSwap**, where the cap no longer applies.
3. **Measured from a price ordinary buyers could not get** — deployer,
   bundler or same-block sniper allocation.

None of these is "found a better filter".

### What this changed in the system

The polling collector discovers pools from GeckoTerminal at **1–15 minutes old**,
which the table above shows is structurally too late for anything above ~5x. So
a second, low-latency path was added: `alpha.data.pumpportal` consumes the
PumpPortal websocket and records pump.fun creations **within seconds**, carrying

- `traderPublicKey` — the deployer wallet, at deployment,
- `solAmount` — the deployer's own buy, i.e. bundle size, known immediately
  rather than inferred later from holder distributions,
- `vSolInBondingCurve` — exact curve state, feeding the curve model with no
  estimation,
- `uri` — metadata, so static social presence is scorable before the first trade.

Measured launch rate: **~6.7 pump.fun creations/minute (~9.6k/day)**, against
~20k/day new pools across all Solana venues.

Seeing every deployer at creation also accumulates something no vendor sells:
a first-hand record of which wallets launch which tokens and how they turn out.
Deployer history is the strongest free rug signal available, and after running
this stream it is measured rather than bought.

### The honest constraint

Being *fast* is necessary but not sufficient. Published latency work puts the
first-block game at sub-60ms detect-to-submit for competitive snipers, against
~800ms for a public-RPC path — and that game is contested by well-capitalised
operators. A websocket feed plus a public RPC does not win block zero.

What it does reach is the **first 30–60 seconds**, where latency still matters
but is no longer the sole determinant, and where the table above still shows
5–10x outcomes present. That is the window this system is now built for.


---

## 11. Exit tuning cannot rescue a bad entry

Grid search over barrier geometry, re-labelling the **same 447 decision points**
(77 pools) under every combination of take-profit, stop-loss and horizon:

| TP | SL | horizon | hit rate | expectancy | payoff | total losses |
|---|---|---|---|---|---|---|
| +30% | −35% | 15m | 18.1% | **−35.2%** | 0.44 | 33.1% |
| +50% | −45% | 30m | 19.9% | −32.8% | 0.70 | 33.1% |
| +100% | −45% | 30m | 19.0% | −27.9% | 1.30 | 33.1% |
| +150% | −45% | 30m | 18.8% | −24.7% | 1.67 | 33.1% |
| +250% | −45% | 60m | 19.2% | **−20.9%** | 2.03 | 33.1% |

**Every geometry loses money.** The best of 72 combinations returns −20.9% per
trade; the worst returns −35.9%. Three things follow, and all of them matter:

**1. The total-loss rate is invariant at 33.1% across every single geometry.**
Not approximately — identically. Widening the stop from −25% to −60% changes it
by nothing at all, because these tokens never traded again after the decision
point. There was no price at which to stop out. This is the same result as §8,
now confirmed on three times the data and across the whole parameter space.

**2. Taking profit early is the worst thing you can do here.** Expectancy
improves monotonically as the take-profit widens (−35.2% at +30%, −20.9% at
+250%) and the payoff ratio rises from 0.44 to 2.03. In a distribution where a
third of trades lose everything, the rare large winner is what pays for them.
Clipping winners at +30% keeps all of the downside and discards the only thing
that funds it. This is the exact opposite of the "take profits early and often"
advice common in retail content.

**3. Stop-loss placement barely matters.** Moving the stop between −25% and −60%
moves expectancy by around one percentage point. The stop is not where the risk
is.

The conclusion is uncomfortable but clean: **selection is the entire game.** No
exit rule, at any setting, makes indiscriminate entry profitable. Combined with
§10 — that the large multiples are gone within two minutes — the two constraints
define the system precisely:

- **enter early enough that the upside still exists** (hence the websocket path),
- **select hard enough to avoid the third of tokens that go to zero** (hence the
  survival-first model),
- and then let winners run rather than clipping them.

Caveat: this sample is 447 rows from 77 pools, entering at 3–90 minutes of age.
It is enough to establish the *shape* — the invariance of the total-loss rate is
not a small-sample artefact — but not enough to fit parameters to. The specific
optimum of +250% is not a recommendation; the direction is.


---

## 12. The edge is the deployer, not the millisecond

Research into early-information channels produced a result that corrects §10's
implication. Both halves matter.

**Speed is not the edge.**

- Solana has no public mempool, so nothing can be seen before a mint lands.
  Every product marketed as "pre-launch detection" is either sub-second
  post-mint detection or inference from days-old funding activity.
- For bundled launches the gap between mint and first buy is **exactly zero** —
  the deployer packs create+buy into one atomic bundle. **Over 50% of pump.fun
  tokens are bought in the block they are created**, by wallets the deployer
  funded. That race is unwinnable *by construction*, not for want of hardware.
- Analysis of 655,770 pump.fun tokens found **no tradeable predictive signal at
  t=0**, with most conditional probability curves sitting below breakeven.
  Winning the millisecond race gets you into a negative-EV bet faster.

**The deployer prior is the edge.**

| | Graduation rate |
|---|---|
| pump.fun platform baseline | **0.63%** (4,338 of 655,770) |
| elite deployers | **40–71%** |

That is a **20–100x lift in prior probability**, available before the token has
traded at all, requiring no latency advantage whatsoever. The elite set is tiny —
around 34 deployers at the strictest threshold — so the signal fires rarely.
That is a feature: it is a rare, high-conviction filter, not a scoring nudge.

### Why this needs Bayesian treatment

A 0.63% base rate makes small samples actively misleading. A deployer with one
graduation from three launches shows a 33% raw rate — an apparent 53x lift that
is almost certainly luck. `alpha.features.deployer` therefore never uses raw
rates; it applies a Beta-Binomial posterior anchored on the platform rate with
60 pseudo-launches of prior strength:

| Record | Raw rate | Posterior | Lower bound | Tier |
|---|---|---|---|---|
| 1/3 | 33.3% | 2.19% | 0.00% | neutral |
| 5/10 | 50.0% | 7.68% | 2.48% | promising |
| 8/20 | 40.0% | 10.47% | 4.88% | **elite** |
| 45/100 | 45.0% | 28.36% | 22.52% | **elite** |
| 0/50 | 0.0% | 0.34% | — | **factory** |

The prior is deliberately heavy. With a base rate this low, promoting a lucky
deployer costs far more than being slow to recognise a genuine one.

This record is accumulated **first-hand** from the launch stream — every
creation adds a launch, every migration adds a graduation. It is slow to
bootstrap and cannot be copied by a competitor who has not been recording.

### Other corrections worth recording

- **Bundled ≠ bad, by itself.** The discriminating measurement is total bundled
  supply versus *currently held* supply. High bundled with current-held near
  zero means the dump already happened; high bundled with high current-held
  means it has not. Retail checks the first and ignores the second.
- **"A credible team bundling is bullish" is wrong.** Insider concentration is
  the top discriminator of *high-risk* launches — early-ten buyers held 17
  percentage points more supply in high-risk tokens.
- **The 2026 profitability "recovery" (30% → 73% of wallets) is survivorship.**
  Active wallets fell 68% from peak; the losers left. Only 5.37% of profitable
  wallets cleared $1,000.
- **Operational note:** Solana slot time was cut from 400ms to 350ms on
  2026-08-22, targeting 200ms. Any hardcoded slot-time assumption is already
  wrong. Jito ShredStream shuts down 2026-09-05 — do not build on it.
- **Unmeasured and worth measuring:** the lag between a KOL's on-chain buy and
  their post is not published anywhere. If the median is under ~2s the channel
  is worthless. This is cheap to measure first-hand and nobody has.


---

## 11. The exact economics of a hold-to-graduation bet

Price on the pump.fun curve is proportional to the **square** of the virtual SOL
reserve. Buying at `vSol` and holding to graduation therefore returns exactly
`(115.0054 / vSol)²`, which fixes the break-even probability:

```
p* = vSol² / 115.0054²
```

| Net SOL raised | vSol | Multiple if it graduates | Break-even P(grad) | EV at 1.4% base rate |
|---|---|---|---|---|
| 0 (launch) | 30.0 | 14.70x | **6.80%** | −79% |
| 5 | 35.0 | 10.80x | 9.26% | −85% |
| 20 | 50.0 | 5.29x | 18.90% | −93% |
| 40 | 70.0 | 2.70x | 37.05% | −96% |
| 70 | 100.0 | 1.32x | 75.61% | −98% |

Published graduation rates are 0.63% (655,770-token study) to ~1.4% all-time.
**This system's own launch stream has now measured 1.85% first-hand** across 542
launches and 24 graduations.

Against a 6.80% break-even, **an unconditional hold-to-graduation bet is roughly
five times short of viable — and it gets worse further up the curve**, because
the remaining multiple shrinks quadratically while the required probability
rises. This is the exact-arithmetic counterpart to the empirical entry-decay
measurement in §10: two independent routes to the same conclusion.

### What closes the gap

Only one conditioning variable has enough lift:

| P(graduate) | EV at launch | EV at 30 SOL raised |
|---|---|---|
| 1.4% (base rate) | −79% | −95% |
| 10% (good deployer) | +47% | −63% |
| 40% (elite deployer) | +488% | +47% |
| 71% (top-tier deployer) | +943% | +161% |

Published work puts elite pump.fun deployers at **40–71% graduation against a
0.63–2% base — a lift of 20 to 100x**, available before the token has traded at
all. That is why `alpha.features.deployer` exists and why its posterior feeds
`alpha.risk.ev_gate` directly.

Note what the second column says: even an elite deployer stops being worth
trading once the token is 30 SOL up the curve. **Being right about the token is
not sufficient; you also have to be early.**

### Post-migration is negative-sum by construction

At graduation roughly 85.0054 real SOL and 793.1M tokens enter the PumpSwap pool.
Under constant product, selling every circulating token back into that pool
leaves SOL permanently stuck: holders collectively pay in 85 SOL and can extract
at most ~67.4 SOL.

```
dead liquidity = 20.7% of migrated SOL
```

Holding through migration is therefore not a neutral act with upside — it is a
bet that you exit ahead of the queue. Depth also drops ~26% at migration (the
curve quotes against 115 SOL of virtual depth, the pool receives ~85 real), so
the same-sized exit costs more afterwards. The default exit policy is to leave
**on the curve**, before graduation.

## 12. What the scanner actually does, and why it stays silent

`alpha.execution.launch_scanner` applies the above to live launches. Run against
400 real launches from the panel it emitted **zero signals**:

| Rejection reason | Count |
|---|---|
| expected value below break-even | 199 |
| deployer has fewer than 3 prior launches | 181 |
| deployer allocation too large | 16 |

This is the correct output, not a defect. Of 342 deployers tracked so far, **65%
have exactly one launch**, and eight show a "100% graduation rate" from a single
launch apiece — which is what luck looks like when you screen hundreds of
wallets. The shrunk lower bound refuses them, as it should.

**The deployer edge is a slow-building asset.** It requires weeks of stream data
before any wallet has enough history to clear the gate. The system is designed to
say nothing until it does, and a sanity check flags any signal rate above 2% as
evidence that something is broken rather than that something was found.
