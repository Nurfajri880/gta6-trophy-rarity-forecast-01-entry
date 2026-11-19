# GTA VI Trophy Rarity Forecast

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, the percent of
the players Exophase tracks who will have earned it at 2026-11-26T00:00:00Z (a week
after launch), as five quantiles (5/25/50/75/95%) per trophy. Scored by pinball loss
on the logit scale.

Entry for the [gta6-trophy-rarity-forecast-01](https://make.fun/c/gta6-trophy-rarity-forecast-01)
competition on [make.fun](https://make.fun).

## How it works

`main.py` (Python 3.13, standard library only, no network) reads every `<id>.in`
case file and writes the matching `<id>.out` forecast. The model is a
saturation-aware growth model on the logit scale: each trophy's earn rate grows
toward a trophy-specific ceiling (from its tier, points and description), with
growth that decays over time, and per-trophy trajectories shrunk toward
tier-level means. The spread widens where the trajectory is uncertain and
narrows where it is clear. It beats the persistence starter because real
earn-rate growth decelerates toward its ceiling instead of continuing linearly.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT. See [LICENSE](LICENSE).
