# Crop yield forecasting

Weekly harvest forecasts 1 to 8 weeks ahead for each polyhouse, with an 80% range. Pick a polyhouse on the dashboard to see its crop, forecast, backtest and the errors of every model.

**Live dashboard:** [crop-yield-forecast.html](https://yasharma9.github.io/YashaSharma_portfolio/crop-yield-forecast.html)

All data here is **sample data** made by `make_sample_data.py` with a fixed seed. It is realistic in shape (crop cycles, seasons, weather effects, pest dips, missing records) but does not come from any farm or company.

## Results on the held-out year

Accuracy = 100% − MAPE. Weeks with under 0.05 kg/m² are left out of MAPE because percentages explode near zero. MAE and WAPE in `results/model_metrics.csv` include every week.

| Crop | Best model | 1 week | 2 weeks | 4 weeks | 8 weeks | Lowest at any distance |
|---|---|---|---|---|---|---|
| Tomato | Random forest | 90.0% | 88.9% | 89.0% | 85.9% | 85.9% |
| Cucumber | Ensemble | 89.9% | 88.2% | 80.3% | 84.0% | 80.0% |
| All crops | | 89.9% | 88.7% | 86.0% | 85.4% | 85.4% |
| Naive, tomato | repeat this week | 82.4% | 71.7% | 60.4% | 42.8% | |
| Naive, cucumber | repeat this week | 69.9% | 47.7% | 26.4% | 5.9% | |

The 80% ranges held 68% to 85% of weeks on the later half of the test year. They are a guide, not a guarantee: pest and disease dips can't be seen coming.

## How it works

1. **Features** for each polyhouse and week:
   - recent harvest (last 4 weeks, mean and trend)
   - fruit set this week and in each of the 7 weeks before (fruit that sets now is picked about 6 weeks later for tomato and 2 for cucumber)
   - the crop's typical yield curve from past cycles, lined up by the week harvest began
   - weeks to the planned removal of the crop
   - crop stage (age, height, nodes)
   - weather over the last 2 weeks, and expected weather for the target week (seasonal normals or a weather forecast file)
   - season and variety
2. **One model per crop and distance** (direct multi-horizon). Separate models for 1, 2, … 8 weeks ahead, so errors don't compound the way they do when a 1-week model is fed its own forecasts.
3. **Five candidates:** naive baseline, ridge regression, random forest, gradient boosting, and an ensemble that averages the two tree models (scikit-learn). Each starts with a median imputer, so missing records never stop a run.
4. **Fair model choice.** Every model is scored on 39 weeks before the test year, in 3 rolling folds. Each fold is predicted by models trained only on the weeks before it. Each crop gets the model with the lowest average validation MAE over all 8 distances. The test year plays no part in the choice, so its accuracy is a fair estimate of live accuracy.
5. **Time-based backtest:** every model is trained on all weeks before the test year and scored on the 52 weeks after. All models' predictions are saved for every polyhouse, so the dashboard can compare them.
6. **No leakage.** Weather normals and typical yield curves used in the backtest come only from weeks before the test year. `--tune` uses `TimeSeriesSplit` on rows sorted by date, and the tuned settings are kept for the final fit.
7. **80% ranges:** the 10th to 90th percentile of backtest errors (kg/m²) at each distance, per crop. They are learned on the first half of the test year and checked on the second half.
8. **Grouped permutation importance:** related inputs (for example crop age, height and nodes) are shuffled together, so credit isn't split between near-copies.
9. **Live forecast:** the chosen model is refit on all data and forecasts every polyhouse with a crop in the ground. It forecasts no harvest before the first harvest age or after the planned removal week.

### What moved accuracy

| Change | Effect |
|---|---|
| Separate models per crop | Cucumber and tomato no longer share one model |
| Fruit set in earlier weeks, typical yield curve by harvest week, planned removal week | Most of the error was at the start and end of each crop. These records, which growers usually keep, fixed most of it: tomato went from about 82% to 86–90% |
| Ensemble candidate and rolling-origin validation | A steadier choice for cucumber, which now stays at 80% or above at every distance |

## Files

| File | What it is |
|---|---|
| `make_sample_data.py` | Makes `data/weekly_records.csv`: 3 years, 20 polyhouses, 122 crop cycles |
| `yield_forecast.py` | The pipeline: features, models, validation, backtest, ranges, live forecast |
| `embed_results.py` | Puts `forecast_output.json` into the dashboard page |
| `data/weekly_records.csv` | Sample input, one row per polyhouse per week |
| `results/forecast_output.json` | Everything the dashboard shows, including every model's backtest predictions |
| `results/forecast.csv` | Live forecast per polyhouse and week, in kg/m² and kg, with ranges and the model used |
| `results/model_metrics.csv` | Accuracy, MAE, RMSE, MAPE, WAPE and R² for every crop, model and distance |

## Run it

```bash
pip install -r requirements.txt
python make_sample_data.py --out data/weekly_records.csv
python yield_forecast.py --data data/weekly_records.csv --out results
python embed_results.py results/forecast_output.json ../crop-yield-forecast.html
```

The forecast takes about 3 minutes on a laptop. Options:

- `--horizon 8`: weeks ahead (default 8)
- `--test-weeks 52`: weeks held out for the backtest (default 52)
- `--val-weeks 39`, `--val-folds 3`: the validation weeks before the test year used to choose the model
- `--tune`: grid search for gradient boosting, with time-ordered folds (slower)
- `--weather file.csv`: use a weather forecast instead of seasonal normals (see below)

### Your own data

Edit `COLUMNS` and `OPTIONAL` at the top of `yield_forecast.py` to match your column names. Only these are required:

- week start date
- polyhouse
- crop
- variety
- harvest in kg
- growing area in m²

Missing optional columns are skipped. The fruit-set (`set_clusters`) and planned-removal (`planned_end_week`) records add the most accuracy at the start and end of a crop. If there is no cycle id, a gap of more than 3 weeks or a change of crop starts a new cycle.

### Weather forecast file

A CSV with:

- `polyhouse`: a polyhouse id, or `ALL` for every polyhouse
- `week_start`
- any of `day_temp_c`, `night_temp_c`, `rh_pct`, `solar_radiation_mj_m2`, `vpd_kpa`

Where a week and polyhouse are in the file, its values replace the seasonal normals for that week.
