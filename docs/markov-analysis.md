# Markov analysis model guide

The dashboard estimates a discrete-time, first-order Markov chain over observed
one-bar return states. It answers how return-state transitions behaved in the
selected data and whether conditioning on the last state improved chronological
one-step probability forecasts. It does not fit hidden states or produce orders.

## Data and state definitions

For selected close prices `q[0], …, q[T]`, observations are
`r[t] = ln(q[t]) − ln(q[t−1])`. Adjusted-close mode uses the stored provider
adjustment; raw mode uses the unadjusted close. The engine rejects missing,
nonpositive or nonfinite prices and duplicate, invalid or unordered timestamps.
It does not fill missing sessions or reconstruct a calendar. A bar can include
an overnight or weekend return. No OHLC, volume or external data is required.

The estimation window selects the latest `W` returns plus their preceding price
bar from the loaded range. Zero selects all loaded observations, with a cap of
50,000 returns. At least 60 returns are required. Forecast horizons count bars
at the chosen chart interval, not clock time. Selecting a long estimation window
does not download more history; load a longer chart range first.

The initial training segment contains `floor(T × trainingPercent / 100)` returns.

| Definition | Boundaries | Equality handling |
| --- | --- | --- |
| Fixed, 3 states | `−b`, `+b`, where `b = threshold / 100` in log-return units | Both boundaries belong to the flat state |
| Quantiles, 3 states | Training 1/3 and 2/3 quantiles | Boundary belongs to the lower state |
| Quantiles, 5 states | Training 1/5, 2/5, 3/5 and 4/5 quantiles | Boundary belongs to the lower state |

Quantiles interpolate linearly between sorted training observations at index
`(N−1) × quantile`. Boundaries stay fixed throughout validation and final fitting.
Repeated values can produce repeated boundaries and empty states. Quantile
labels describe ordered returns, so a “high return” state need not mean a
positive return. Fixed thresholds are percentages of log return, not simple return.

## Estimating transitions

`C[i,j]` counts consecutive observations in origin state `i` and destination state
`j`; `N[i] = Σj C[i,j]`. A sequence of `T` states contributes `T−1` transitions.
The row-stochastic estimate is:

```text
P[i,j] = (C[i,j] + alpha) / (N[i] + K × alpha)
```

For positive alpha, this is the posterior mean under independent symmetric
Dirichlet row priors. A positive alpha allows transitions that were not observed.
For alpha zero, the estimate is the empirical transition proportion; an
unobserved origin row uses a uniform fallback and produces a diagnostic. This
fallback is part of the reported model and can influence its stationary behavior.

Row details show unsmoothed proportions and approximate 95% Wilson binomial
intervals alongside fitted probabilities. With `p = count / N` and
`z = 1.959963984540054`, the Wilson center and half width are:

```text
center = (p + z²/(2N)) / (1 + z²/N)
half   = z × sqrt(p(1−p)/N + z²/(4N²)) / (1 + z²/N)
```

There is no interval when `N = 0`. These are individual raw-count intervals,
not simultaneous multinomial bounds or Bayesian credible intervals. The binomial
approximation does not capture changing probabilities or additional serial
dependence. Fewer than 30 observed outgoing transitions triggers a descriptive
sample-size warning, not a statistical decision threshold.

## Forecasts and chain structure

Starting from the observed current state `s`, the h-bar state distribution is
`e[s] P^h`. The engine propagates the probability vector without rounding. Table
formatting rounds for display; the JSON retains full precision.

To calculate the probability of visiting state `j` at least once, the engine
propagates only paths that have not visited `j`, zeroing the mass arriving in `j`
after each step. One minus surviving mass is the visit probability. The starting
observation is included: a visit to the current state has probability one at all
horizons. Visits to other states start counting from the next bar.

Expected run length from entry to state `i` is `1 / (1 − P[i,i])`, including its
first observation. For an absorbing state, it is infinite, shown as `∞` and
encoded as `null` in the JSON `dwell` field. The observed latest run is separately
reported. A first-order chain does not change its persistence estimate just
because the current run has already lasted several bars.

The engine finds communicating classes from positive-probability edges and
identifies closed classes. With one closed class, it solves `πP = π`, `Σπ = 1`
using pivoted elimination and reports the numerical residual. Multiple closed
classes have no unique stationary distribution, so stationary occupancy is
unavailable rather than selecting an arbitrary solution. Class periods are
computed from the greatest common divisor of cycle-length differences. Periodic
chains can have stationary occupancy while horizon forecasts continue cycling.
Stationary values are properties of the fitted model, not claims that market
behavior will remain stationary.

The overview also reports sample occupancy and mean simple return
`mean(exp(r) − 1)` within each state. These means describe the return that defines
the state; they are not future returns conditional on today's state.

## Chronological validation

Training uses the first segment's states and its internal transitions. The first
validation forecast starts from the last training state and predicts the first
validation state. The boundary-crossing transition is therefore held out from
that first forecast.

- **Expanding:** predict and score the next state, then add its transition and
  state-frequency observation before the following forecast.
- **Frozen:** score every validation observation with unchanged training
  transition counts and baseline frequencies, conditioning on the most recently
  observed state.

The baseline predicts historical state frequencies with additive alpha smoothing
and the same update schedule. It does not condition on the origin state. It uses
all training state observations; the transition model necessarily has one fewer
initial training transition. Baseline and model are scored on identical outcomes.

| Metric | Definition | Interpretation |
| --- | --- | --- |
| Multiclass Brier | Mean `Σj (p[j] − 1[j=actual])²` | Lower is better; range 0–2 |
| Log loss | Mean `−ln(max(p[actual], 10⁻¹⁵))` | Lower is better; natural logarithm |
| Accuracy | Fraction whose most probable state is correct | Higher is better; ties choose the first state |
| Brier skill | `1 − Brier(model) / Brier(baseline)` | Positive favors Markov; unavailable if baseline Brier is zero |

The confusion matrix has actual states in rows and predicted states in columns.
Calibration groups predictions by top-class confidence into five bins of width
0.2, left inclusive and right exclusive, except the last includes 1.0. It compares
mean top-class confidence with observed top-class accuracy and reports bin counts.
An empty bin has no confidence or accuracy estimate.

Fewer than 100 validation forecasts produces a sample-size diagnostic. Zero or
negative Brier skill explicitly reports that the Markov model did not beat its
baseline. Scores do not validate simulated multi-step return ranges or trading
profitability. Choosing parameters after inspecting these scores uses the
validation data for model selection; use a separate untouched period for a final
assessment of a selected configuration.

After validation, current forecasts refit transitions on the entire estimation
window. They retain the original training boundaries. Their counts and matrix
therefore differ from the earlier validation forecast estimates. Each validation
record contains its own probability vector and baseline in the JSON export.

## Simulation and uncertainty

For each path and horizon, draw a destination state using the full-window fitted
transition matrix, then draw one empirical log return with replacement from that
destination state's observed returns. Sum log returns and convert the cumulative
sum to simple percent return using `100 × expm1(sum)`.

The seed controls a deterministic 32-bit pseudorandom generator. The same ordered
prices, settings, engine version and seed reproduce the output. Path count is
100–10,000; horizon is 1–250 bars. Simulations run in a dedicated worker and can
be canceled without blocking chart interaction. Any reachable state with no
observed return disables simulation; the engine does not invent returns for it.

The fan shows pointwise 5th, 25th, 50th, 75th and 95th percentiles. Loss
probability is the fraction of paths with negative cumulative return at the
selected horizon. Maximum drawdown measures the largest decline from the running
peak of each simulated path's bar-close value, including its initial value. The
reported median and 5th percentile are summaries of these pathwise drawdowns.

The simulation assumes returns are independent given the destination-state
sequence, holds estimated parameters fixed, and resamples only observed returns.
It omits parameter uncertainty, shocks beyond observed return pools, intrabar
extremes, costs and execution. It is neither a posterior predictive distribution
integrating parameter uncertainty nor a trading backtest. Its percentile band is
not a confidence interval for the true future return or a simultaneous envelope
containing the stated proportion of complete paths.

## Stability diagnostics

Row entropy is `−Σj P[i,j] log2 P[i,j]`. Conditional entropy weights row entropies
by each origin's share of observed outgoing transitions. Larger values indicate
more dispersed next-state distributions within the fitted model.

Half-sample drift splits the observed state sequence into two contiguous halves,
reuses the same boundaries, fits each half with the selected smoothing, and
excludes the transition crossing that split. Per-row total variation is
`0.5 × Σj |Pearly[i,j] − Plate[i,j]|`. A row is unavailable if either half has no
observed outgoing transition. Distances above 0.25 produce a descriptive warning.
This is not an independence test, a stationarity test or a calibrated p-value.

## Reproducibility and implementation

`web/markov-engine.js` exposes the pure `AtlasMarkov.run(bars, parameters)` API.
`web/markov-worker.js` performs the calculation, and `web/markov.js` renders
results. All assets are routed by the local dashboard and included in static
builds and their ZIP manifests. Worker URLs resolve relative to the script, so
static hosting under a repository path works.

Changing model settings or loading new chart data terminates pending work and
invalidates results and exports. Responses from superseded work cannot restore
stale results. Settings persist in local storage; analysis results are recomputed.

The versioned JSON report includes symbol/currency/interval context, normalized
settings, data dates, training cutoff, boundaries, raw transition counts, fitted
matrix, state summaries, communicating classes, forecasts, simulation quantiles,
validation scores and predictions, diagnostics and full timestamped state history.
The transition CSV supplies raw counts, fitted and empirical probabilities,
Wilson intervals and basic estimation context. Use the JSON for full model
reproduction. Undefined statistics are `null`, never nonstandard JSON infinities.

Tests cover hand-counted transitions, smoothing, matrix propagation, stationary
solutions, periodic and absorbing chains, fixed boundaries, ties, Wilson
intervals, score formulas, causality, deterministic simulation, visit
probabilities, malformed inputs, price basis and window selection. Browser checks
exercise workers, exports, persistence, stale results, cancellation, recovery,
keyboard controls, themes, mobile layouts and static builds under nested paths.

References: [Berkeley stationary-distribution lecture](https://www.stat.berkeley.edu/~aldous/150/Lectures/lecture_9_post.pdf)
and [Forecasting: Principles and Practice, time series cross-validation](https://otexts.com/fpp3/tscv.html).
