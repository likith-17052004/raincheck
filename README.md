# ForecastTrust AI

**Will tomorrow's rainfall forecast fail? Find out before the rain falls.**

Weather apps show one rainfall number and no sign of how far to trust it. ForecastTrust AI reads the
recent forecasts for a city and the rain that has been verified so far, and predicts whether the
day-ahead forecast will miss by more than max(5 mm, 50% of the forecast). It answers on the day the
forecast is issued, before the rain is observed.

- Demo: https://claude.ai/artifact/61fMmuQV294ZnqJrK7GPKz
- TimeNet dataset PR: https://github.com/OpenTSLM/TimeNet/pull/846
- Dataset id: `forecasttrust/india-rainfall-reliability` v1.0.0

## What we built

1. **A TimeNet dataset and connector.** 12 Indian cities, 7,620 records. Each record is a city and a
   valid day, holding 14 days of day-1, day-2 and day-3 ECMWF IFS forecasts as originally issued, and
   ERA5 rainfall verified through two days before the valid day. The rain of the valid day, and the day
   before it, never appear as input. Tasks: a `reliable`/`failure` classification and the absolute
   error in mm. Splits are chronological (train to 2024-12-31, validation to 2025-05-31, test from
   2025-06-01). Every source file is pinned by SHA-256, and missing data is skipped, never filled.
2. **Baselines** on the stored splits (logistic regression, XGBoost, base rate).
3. **An OpenTSLM fine-tune.** OpenTSLM-SP with a frozen Qwen2.5-0.5B, LoRA, and a trainable time-series
   encoder and projector, fine-tuned on one NVIDIA H100. The failure probability is the model's own
   log-probability of the word "failure" against "reliable", with no text parsing.
4. **A demo** that lets you pick a city and a day of the held-out test period and see the forecast, the
   verified rain, the predicted failure risk and what actually happened.

## Results (held-out test, June to November 2025, 2,016 city-days, 23.2% failures)

| Model | AUROC | Avg. precision | Brier |
| --- | --- | --- | --- |
| Always predict the base rate | 0.500 | 0.232 | 0.185 |
| Forecast amount only (logistic) | 0.785 | 0.421 | 0.205 |
| XGBoost with error history | 0.809 | 0.488 | 0.142 |
| OpenTSLM, 3 seeds, history | 0.810 / 0.816 / 0.815 | 0.477 / 0.488 / 0.481 | 0.140 / 0.154 / 0.140 |
| OpenTSLM, no history | 0.815 | 0.484 | 0.139 |

The OpenTSLM model **matches the best classical baseline and does not beat it.** Removing the
history of verified rain did not hurt, so most of the signal comes from the forecasts themselves.
Each fine-tune takes about 4.5 minutes on one H100.

## Limits we want you to know about

- The "truth" is ERA5 reanalysis, not rain gauges, so a failure means "disagrees with ERA5".
- Only 12 cities, and 21 months of data. The test period is the monsoon, where failures are common.
- The validation split (January to May) has few failures, so it is a weak signal for choosing
  checkpoints.
- One failure threshold. It treats missed rain and false alarms alike.
- Open-Meteo data is CC BY 4.0, and its free API is limited to non-commercial use.

## Repository layout

| Path | What it holds |
| --- | --- |
| `connector/` | Copy of the TimeNet connector and dataset card, as submitted in the pull request. |
| `training/` | Loader, baselines, the OpenTSLM fine-tune, a CPU test of the scoring code and the GPU runbook. |
| `demo/` | The page source and `build_data.py`, which exports the real test predictions it shows. |
| `results/` | Metrics of every run, the test scores, the baselines and the results chart. |

## Reproduce

Clone [TimeNet](https://github.com/OpenTSLM/TimeNet) and [OpenTSLM](https://github.com/OpenTSLM/OpenTSLM) next
to this folder. Copy `connector/` into
`TimeNet/packages/timenet-connectors/src/timenet_connectors/datasets/forecasttrust/` (or use the merged
dataset once the pull request lands), then:

```bash
cd TimeNet && uv run timenet-build build forecasttrust/india-rainfall-reliability --local-registry ../registry
cd .. && python training/baselines_timenet.py
python training/train_forecasttrust.py --llm Qwen/Qwen2.5-0.5B --out results/qwen05_s0   # needs a GPU
```

See `training/RUNBOOK.md` for the GPU steps. Weather data by Open-Meteo.com (CC BY 4.0).
