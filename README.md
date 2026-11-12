# makefun-pumpfun-top5

Entry for make.fun's *Predict the top 5 new Solana coins by market cap, 24 hours
out* competition (`pumpfun-top5-forecast-01`).

`main.py` (Python 3.13, standard library only, no network) reads the house's
ten-minute market-cap snapshots at `/input` and writes `/output/forecast.json`:
for each of the five places, a probability for every coin it names, the top five
in order, and a probability for each of the 82 log10 market-cap bins.

The forecaster treats a coin's log10 market cap as a mean-reverting random walk
toward the pump.fun launch floor, with dispersion growing as the square root of
the horizon, and simulates a population of coins the history never ranked
(coins born after the cut, and coins too small for the snapshots) whose few
breakouts are what a top-5 chart is usually made of. Monte Carlo over both
populations gives the place probabilities, the chart-presence probabilities
that decide which history coins to name, and the size-bin distributions; the
forecast averages over the seven answer reads, since one forecast is scored at
all of them.

Run it locally:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
