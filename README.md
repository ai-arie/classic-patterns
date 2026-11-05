# classic-patterns

A chart-pattern screener for US stocks on daily data. It finds classic long setups, shows each
setup's trade plan (trigger, stop, target) and estimates how that plan would play out, with two
models that run on a laptop CPU.

**The screener shows pattern setups with estimated probabilities; you decide.**

> **Disclaimer.** This is a research and education project. It is **not investment or financial
> advice** and not a signal service. **Do not take every pattern it finds — use your own
> judgement.** The models describe how similar setups behaved in the past, on free Yahoo Finance
> data (via `yfinance`, with known gaps and errors) and on a universe of currently listed stocks
> (survivorship bias). No warranty of any kind.

Patterns: cup-and-handle (and deep cups separately), inverse head-and-shoulders, consolidation
breakouts at highs, ascending triangles. Each is reported when it is *forming* (close near the
trigger level) and when it *triggers* (close through it).

**The screener shows long setups only.** Short setups are not shown because their results in testing
were not good enough. They remain in the detectors and in the models' training data.

## How it works

- **No lookahead.** Everything computed for a day uses data up to that day's close, and a trade
  decided at the close is entered at the next open. Tests check that features and detectors give
  the same output on data cut at that day as on the full history.
- **Eligible setups.** On the day of a setup, a stock needs a close of at least $5, a 50-day dollar
  volume of at least $15M and 300 days of history.
- **Outcomes.** The models learned from how each setup's plan played out: entry at the next open,
  the pattern's stop and target fixed at entry, at most 60 trading days. If stop and target are
  touched on the same day, the stop counts first. Costs are 10 bps per side.
- **Testing.** The models were tested walk-forward on nine two-year blocks from 2006 to 2023, each
  trained only on earlier data, and then once on a locked holdout from 2024 onwards. The released
  models were retrained on all the data afterwards.

## What the app shows

**Triggered setups:** ticker, pattern, close, stop, risk in percent and, for the pattern's own plan
(entry at the next open, the pattern's stop and target, at most 60 trading days):

- P(stop first), P(neither within 60 days), P(target first),
- an expected R (multiple of the initial risk, net of costs, from training-period averages),
- the target, as a reference (see [Targets](#targets-use-your-own-judgement)).

**Forming setups:** ticker, pattern, close, trigger, distance to the trigger, stop, risk from the
trigger and the chance of triggering within 20 trading days. They get no trade estimates, because
the trade models were trained on triggered setups.

Also shown: the market regime (VIX, S&P 500 vs its 200-day average) and a chart per setup with the
trigger, stop, target and pattern points. Filters: pattern, stage, maximum P(stop first).

Two models, switchable in the app:

- **student** (default): Chronos-2-Small fine-tuned on the stock's price history, distilled from
  larger Chronos-2 models, ONNX int8;
- **LightGBM**: hand-crafted pattern, stock and market features.

Both show probabilities, not a ranking of trades to take. The tables are sorted by ticker. At
screening time the next open is unknown, so a triggered setup's distances are measured from the
day's close.

## Targets: use your own judgement

The target shown is the pattern's textbook measured move, which is usually far from the entry. For
long setups triggered since 1995, the median target is about 12% above the trigger, and about 32% for
deep cups. Many targets are not reached within the 60-day window.

Better target placement is expected to raise the share of successful trades, but the models were not
trained to choose take-profit levels. They treat the pattern's target as given and estimate how that
plan plays out. **Use your own judgement, especially with the targets.**

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11. The app does not need PyTorch or a GPU.

```bash
uv sync --extra data --extra app
```

## First run

1. **Download the price data** (about 2 hours, resumable; the data providers' terms apply to you):

   ```bash
   uv run python scripts/build_universe.py     # ticker universe
   uv run python scripts/download_prices.py    # daily prices (re-run the same command if it stops)
   uv run python scripts/validate_data.py      # data checks
   uv run python scripts/build_clean.py        # cleaned data
   ```

   On macOS, keep the machine awake during the download, e.g.
   `nohup caffeinate -is uv run python scripts/download_prices.py > logs/download_stdout.log 2>&1 &`.

2. **Build features and detect patterns** (a few minutes):

   ```bash
   uv run python scripts/build_features.py
   uv run python scripts/detect_patterns.py --steps build
   ```

3. **Put a model bundle in `models/production/`** (see [Models](#models)) and check that this
   machine reproduces its scores:

   ```bash
   uv run python scripts/parity_check.py       # prints PARITY OK
   ```

## Daily use

```bash
uv run python scripts/screen.py --refresh       # update prices, rebuild, score the latest session
uv run streamlit run src/classic_patterns/app/ui.py
```

`--refresh` brings prices up to the last completed NYSE session (re-downloading a ticker if Yahoo
re-adjusted its history), validates and cleans, rebuilds features and patterns, and scores the
setups into `artifacts/screener/<date>.parquet`. A later run for the same session never overwrites
that file: it is saved next to it as `<date>_<run time>.parquet`, and the app opens the newest run
by default. Without `--refresh` it scores the data on disk;
`--date YYYY-MM-DD` scores an earlier session. Run it after the US close.

## Models

Trained models are not part of this repository. A trained model bundle may be available on
request through a [gated Hugging Face repository](https://huggingface.co/ai-arie/classic-patterns-models)
(requests are reviewed). It comes with a model card and checksums.

The hand labels used to tune the detectors are not published. If you're interested in them, open an
issue on this repository or start a discussion on the Hugging Face model page.

## Also tested

I also tested SEC filing data (fundamentals, earnings timing, sector strength) as extra model
inputs, and a few alternative entry and exit rules. None gave a clear improvement, so the released
models and the screener do not use them. The research code and detailed results are not part of
this repository.

I also tried these models on hourly crypto prices. In tests on 2021–2023, no version made a
reliable profit, while the S&P 500 rose about 33% over the same three years. The screener does not
use crypto, and no crypto model is released.

## Limitations

- Survivorship bias: the universe is today's listed stocks; delisted companies are missing.
- Yahoo Finance data quality (adjustments, gaps, bad prints) is not guaranteed.
- Costs are a flat 10 bps per side.
- Pattern definitions are one operationalisation of subjective chart patterns.
- Targets are textbook measured moves; setting take-profit levels was not trained (see
  [Targets](#targets-use-your-own-judgement)).
- Probabilities describe how similar setups behaved in the past; markets change.
- Built for US stocks on daily data.

## Compute

The project used approximately 141 GPU-hours on NVIDIA A100 80GB GPUs. The app runs on CPU.

## License

Code: Apache License 2.0 (see `LICENSE`).
