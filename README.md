# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Nurfajri880/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Nurfajri880/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, the percent of
the players Exophase tracks who will have earned it at 2026-11-26T00:00:00Z (a week
after launch), as five quantiles (5/25/50/75/95%) per trophy. Scored by pinball loss
on the logit scale.

Entry for the [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
competition on [make.fun](https://make.fun).

## How it works

`main.py` (Python 3.13, standard library only, no network) reads every `<id>.in`
case file and writes the matching `<id>.out` forecast. The model is a hybrid:
each trophy's earn rate is projected from its earners and the tracked player
base separately (so the rate can fall when the base grows faster than the
earners), toward a trophy-specific ceiling (from its tier and description),
with growth that decays over time, and per-trophy trajectories shrunk toward
tier-level means. The spread widens where the trajectory is uncertain and
narrows where it is clear. It beats the persistence starter because real
earn-rate growth decelerates toward its ceiling instead of continuing linearly.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT. See [LICENSE](LICENSE).

---

<sub>Entered on [make.fun](https://make.fun/?ref=byetntnj&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
