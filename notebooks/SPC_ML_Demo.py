# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # SPC signal detection — AttainX demo
# MAGIC
# MAGIC This notebook generates synthetic application-receipt and workflow-completion events for a USCIS-related teaching example. It reads no agency records or external data. It does not model eligibility, adjudication, individual applicants or agency performance.
# MAGIC It demonstrates XmR, CUSUM, and EWMA control rules; a classifier that reproduces their
# MAGIC same-window label; a separate **next-business-day count forecast**; chronological evaluation;
# MAGIC analyst review candidates; five-day ARIMA/Holt-Winters forecasts; Isolation Forest;
# MAGIC subgroup analysis; fixture lineage and investigation; historical assessment; lifecycle exercises;
# MAGIC a daily replay dashboard; drift monitoring; an interactive dashboard; and optional MLflow and Delta integration.
# MAGIC
# MAGIC **Dataframe walkthrough:** Cells 2–10 show compact views of Intake A across the same five dates. Headers explain added columns, filters, joins and changes in what a row represents. Classifier, forecast, review and subgroup results are branches of the source data, not one long chain.
# MAGIC
# MAGIC **Run all 22 code cells.** Install `requirements-demo-lock.txt` in the notebook environment first.
# MAGIC The lineage, operational events and reviewer actions are fictional fixtures. No notifications are sent.
# MAGIC
# MAGIC **Interpretation:** A signal means a statistical rule fired, not that a real problem was confirmed.
# MAGIC The classifier reproduces a rule-generated label from same-window measurements; it does not predict
# MAGIC a future incident. The forecast is genuinely forward-looking, but its scores describe only this
# MAGIC fictional dataset. Neither model decides whether a process problem occurred.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 1: Set the rules for this run
# MAGIC **Plain English:** Choose the random seed, how much history each rule reads, and whether to save results.
# MAGIC **Technique:** Python libraries handle tables, numerical calculations, charts, and random forest models.
# MAGIC **Check:** `OUTPUT_SCHEMA` is `ml_statistical_process_controls.demo_schema`. `SPC_DEMO_LOCAL_TEST=1` disables Delta and MLflow writes during local validation. `UC_MODEL_NAME` remains empty until candidate registration is configured.
# MAGIC **Developer note:** Change the settings here, then rerun all cells so labels, splits, charts, and logged metrics agree.

# COMMAND ----------

# DBTITLE 1,Configuration and imports
import re
import os
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, mean_absolute_error,
)
from sklearn.ensemble import RandomForestRegressor

SEED = 42
N_DAYS = 420
WINDOW = 25
BASELINE_DAYS = 90
XMR_MR_MULTIPLIER = 2.66
CUSUM_K_FACTOR = 0.5
CUSUM_H_MULTIPLIER = 4.77
EWMA_LAMBDA = 0.10
EWMA_L = 2.703
DATASET_ID = f"attainx_applications_v3_seed{SEED}"

# Dedicated demonstration destination. Local test entry points MUST set the explicit override.
OUTPUT_SCHEMA = "ml_statistical_process_controls.demo_schema"
LOCAL_TEST = os.environ.get("SPC_DEMO_LOCAL_TEST") == "1"
if LOCAL_TEST:
    OUTPUT_SCHEMA = ""
# Optional: set to a permitted three-part catalog.schema.model name to register the model.
# Leave empty until the notebook and MLflow experiment work.
UC_MODEL_NAME = ""

plt.rcParams.update({"figure.figsize": (11, 4), "axes.grid": True, "grid.alpha": 0.2})

# Small views of actual intermediate results keep the same queue and dates visible.
DEMO_SERIES = "Intake A"
DATAFRAME_PREVIEW_ROWS = 5
dataframe_stages = {}

def show_dataframe_stage(title, frame, columns, *, grain, change, sample=None):
    """Display a bounded copy; leave the calculation's full dataframe unchanged."""
    selected = frame if sample is None else sample
    if sample is None:
        if "series_id" in selected:
            selected = selected[selected.series_id == DEMO_SERIES]
        if "run_date" in selected:
            selected = selected[selected.run_date.isin(DEMO_PREVIEW_DATES)]
            selected = selected.sort_values("run_date")
        selected = selected.head(DATAFRAME_PREVIEW_ROWS)
    preview = selected.loc[:, columns].copy().reset_index(drop=True)
    dataframe_stages[title] = preview
    print(f"\n{title} | {DEMO_SERIES}")
    print(f"Full dataframe: {len(frame):,} rows x {len(frame.columns)} columns. Row meaning: {grain}.")
    print(change)
    print(f"Preview: {len(preview)} rows; rounded for display only.")
    display(preview.round(3)) if "display" in globals() else print(preview.round(3).to_string(index=False))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 2: Make a safe example dataset
# MAGIC **Plain English:** Create over one million synthetic application events across three queues: Intake A and Intake B count application receipts; Completions A counts completed workflow steps. Receipts and completions are independent event streams, not a linked case lifecycle. We plant a few shifts for statistical review.
# MAGIC **Technique:** A seeded process generates event volumes, then materializes individual event rows with unique IDs and timestamps. Normalize those rows and aggregate them to daily counts. Validate IDs, types, date coverage, nonnegative counts and exact reconciliation.
# MAGIC **Output:** `raw_events_df` and `prepared_events_df` contain actual event rows; `daily_df` contains only 1,260 daily series observations. A million events does not mean a million independent training examples. `dataset_id` identifies this synthetic run.
# MAGIC **Developer note:** The planted changes help explain the demo; they are no proof that real agency data behave this way.

# COMMAND ----------

# DBTITLE 1,Generate fictional operational series
rng = np.random.default_rng(SEED)
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
dates = pd.date_range("2025-01-06", periods=N_DAYS, freq=BUSINESS_DAY)
series_config = {"Intake A": (110, 11), "Completions A": (94, 10), "Intake B": (75, 9)}
daily_parts = []
for series_id, (center, sigma) in series_config.items():
    t = np.arange(N_DAYS)
    weekly_pattern = 3 * np.sin(2 * np.pi * t / 5)
    shift = np.where((t >= 140) & (t < 155), 19, 0)
    shift += np.where((t >= 260) & (t < 275), -16, 0)
    shift += np.where((t >= 355) & (t < 370), 16, 0)
    spike = np.where(np.isin(t, [105, 220, 319]), 32, 0)
    values = np.maximum(0, np.round(center + weekly_pattern + shift + spike + rng.normal(0, sigma, N_DAYS)))
    daily_parts.append(pd.DataFrame({"series_id": series_id, "run_date": dates, "daily_count": values.astype(int) * 10}))
# Generate actual event rows from the seeded daily process; then derive the analytical
# counts from those rows. Ten-fold units give a million-row engineering demonstration.
generation_plan_df = pd.concat(daily_parts, ignore_index=True)
plan_index = np.repeat(np.arange(len(generation_plan_df)), generation_plan_df.daily_count.to_numpy())
raw_events_df = pd.DataFrame({
    "event_id": np.arange(1, len(plan_index) + 1, dtype=np.int64),
    "series_id": pd.Categorical(generation_plan_df.series_id.to_numpy()[plan_index]),
    "event_timestamp": generation_plan_df.run_date.to_numpy()[plan_index]
        + pd.to_timedelta(rng.integers(8 * 3600, 18 * 3600, len(plan_index)), unit="s"),
})
raw_events_df["event_type"] = pd.Categorical(np.where(
    raw_events_df.series_id.eq("Completions A"), "workflow_completion", "application_receipt"))
assert len(raw_events_df) >= 1_000_000
assert raw_events_df.event_id.is_unique and raw_events_df.event_id.notna().all()
assert pd.api.types.is_integer_dtype(raw_events_df.event_id)
assert pd.api.types.is_datetime64_any_dtype(raw_events_df.event_timestamp)
assert not raw_events_df.isna().any().any()
assert set(raw_events_df.event_type) == {"application_receipt", "workflow_completion"}
prepared_events_df = raw_events_df.assign(run_date=raw_events_df.event_timestamp.dt.normalize())
assert prepared_events_df.run_date.isin(dates).all()
assert set(prepared_events_df.series_id) == set(series_config)
daily_df = prepared_events_df.groupby(["series_id", "run_date"], observed=True).size().rename("daily_count").reset_index()
daily_df["series_id"] = daily_df.series_id.astype(str)
daily_df = daily_df.sort_values(["series_id", "run_date"]).reset_index(drop=True)
expected_counts = generation_plan_df.sort_values(["series_id", "run_date"]).reset_index(drop=True)
pd.testing.assert_frame_equal(daily_df, expected_counts)
assert int(daily_df.daily_count.sum()) == len(raw_events_df) == len(prepared_events_df)
daily_df["source_name"] = "synthetic_application_events"
daily_df["dataset_id"] = DATASET_ID
assert daily_df.groupby(["series_id", "run_date"]).size().max() == 1
assert daily_df["daily_count"].notna().all() and daily_df["daily_count"].ge(0).all()
assert daily_df.groupby("series_id").run_date.nunique().eq(N_DAYS).all()
dataset_summary = {"event_rows": len(raw_events_df), "daily_rows": len(daily_df),
                   "series": len(series_config), "business_dates": N_DAYS,
                   "first_date": str(dates.min().date()), "last_date": str(dates.max().date())}
print(f"Materialized {len(raw_events_df):,} synthetic event rows; aggregated to {len(daily_df):,} daily observations.")
print("Scale is event-row count, not training-set size. Keys: event_id; then series_id + run_date.")
DEMO_PREVIEW_DATES = daily_df.loc[daily_df.series_id == DEMO_SERIES, "run_date"].sort_values().tail(DATAFRAME_PREVIEW_ROWS)
show_dataframe_stage(
    "0a. Raw application events — raw_events_df", raw_events_df,
    ["event_id", "event_timestamp", "series_id", "event_type"], sample=raw_events_df.head(5),
    grain="one synthetic application-receipt or workflow-completion event; unique event_id",
    change="START: materialized events with typed timestamp, event ID, queue and event type. No person or decision fields.",
)
show_dataframe_stage(
    "0b. Prepared events — prepared_events_df", prepared_events_df,
    ["event_id", "event_timestamp", "series_id", "event_type", "run_date"], sample=prepared_events_df.head(5),
    grain="one event, same event_id and row count as raw input",
    change="ADD run_date by normalizing event_timestamp; validate all dates against the demonstration business calendar.",
)
show_dataframe_stage(
    "1. Starting counts — daily_df", daily_df, ["series_id", "run_date", "daily_count"],
    grain="one queue on one business date; unique series_id + run_date",
    change="GROUP event rows by series_id + run_date; daily_count is COUNT of events. Counts reconcile exactly to event-row count.",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 3: Define the three SPC checks
# MAGIC **Plain English:** XmR notices unusually large individual values; CUSUM adds small departures from a past average; EWMA smooths recent values to reveal a shift.
# MAGIC **Technique:** XmR uses the average moving range for limits. CUSUM accumulates deviations. EWMA gives newer observations more weight.
# MAGIC **Dataframe walkthrough:** This cell defines functions; it does not transform rows yet. Cell 4 applies them.
# MAGIC **Output:** Three functions return a yes/no rule signal; XmR also returns the center, limits, and moving range for the chart.
# MAGIC **Developer note:** Each function scans the **whole preceding window**. A flag can stay true across several later runs after one event.

# COMMAND ----------

# DBTITLE 1,Define the three SPC rules
def xmr_signal(window):
    """Reference notebook: estimate limits from this window and flag any point beyond them."""
    center = float(np.mean(window))
    mr_bar = float(np.mean(np.abs(np.diff(window))))
    ucl = center + XMR_MR_MULTIPLIER * mr_bar
    lcl = center - XMR_MR_MULTIPLIER * mr_bar
    return bool(np.any((window > ucl) | (window < lcl))), center, ucl, lcl, mr_bar


def cusum_signal(window, baseline_mean, baseline_sigma):
    k = CUSUM_K_FACTOR * baseline_sigma
    h = CUSUM_H_MULTIPLIER * baseline_sigma
    upper, lower = 0.0, 0.0
    fired = False
    for value in window:
        upper = max(0.0, upper + value - baseline_mean - k)
        lower = min(0.0, lower + value - baseline_mean + k)
        fired |= upper > h or lower < -h
    return fired


def ewma_signal(window, baseline_mean, baseline_sigma):
    z = baseline_mean
    fired = False
    for index, value in enumerate(window, start=1):
        z = EWMA_LAMBDA * value + (1 - EWMA_LAMBDA) * z
        band = EWMA_L * baseline_sigma * np.sqrt(
            EWMA_LAMBDA / (2 - EWMA_LAMBDA) * (1 - (1 - EWMA_LAMBDA) ** (2 * index))
        )
        fired |= abs(z - baseline_mean) > band
    return fired

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 4: Turn past counts into signals and features
# MAGIC **Plain English:** For each date, inspect the prior 25 business days. XmR derives limits from that window; CUSUM and EWMA use an earlier 90-day reference period. Save which SPC check fired.
# MAGIC **Technique:** Rolling-window feature engineering produces means, variation, recent trend, and distance from the baseline. `signal_detected` is `xmr_signal OR cusum_signal OR ewma_signal`.
# MAGIC **Output:** `window_features_df` holds the measurements; joining four rule-label columns creates `signals_df`, which feeds the later models.
# MAGIC **Developer note:** The features and rule label describe the **same past window**. The classifier below imitates these rules; it is not advance warning.

# COMMAND ----------

# DBTITLE 1,Calculate the rule labels and modeling features
rows = []
for series_id, group in daily_df.groupby("series_id", sort=True):
    group = group.sort_values("run_date").reset_index(drop=True)
    values = group["daily_count"].to_numpy(dtype=float)
    # The reference notebook labels a 25-day window ending yesterday for each run_date.
    # Start after 90 baseline days to avoid an unstable early-history fallback.
    for end in range(BASELINE_DAYS + WINDOW, len(group)):
        window = values[end - WINDOW:end]
        baseline = values[end - WINDOW - BASELINE_DAYS:end - WINDOW]
        baseline_mean = float(np.mean(baseline))
        baseline_sigma = max(float(np.std(baseline, ddof=1)), 1e-6)
        xmr, center, ucl, lcl, mr_bar = xmr_signal(window)
        cusum = cusum_signal(window, baseline_mean, baseline_sigma)
        ewma = ewma_signal(window, baseline_mean, baseline_sigma)
        rows.append({
            "series_id": series_id, "run_date": group.loc[end, "run_date"],
            "window_start": group.loc[end - WINDOW, "run_date"],
            "window_end": group.loc[end - 1, "run_date"],
            "window_mean": center, "window_std": float(np.std(window, ddof=1)),
            "window_range": float(np.ptp(window)), "mr_mean": mr_bar,
            "last_value": float(window[-1]), "last_5_mean": float(np.mean(window[-5:])),
            "last_5_std": float(np.std(window[-5:], ddof=1)),
            "trend": float(np.mean(window[-5:]) - np.mean(window[:5])),
            "baseline_mean": baseline_mean, "baseline_sigma": baseline_sigma,
            "ucl": ucl, "lcl": lcl,
            "max_baseline_deviation": float(np.max(np.abs(window - baseline_mean)) / baseline_sigma),
            "max_window_deviation": float(np.max(np.abs(window - center)) / max(mr_bar, 1e-6)),
            "xmr_signal": xmr, "cusum_signal": cusum, "ewma_signal": ewma,
            "signal_detected": bool(xmr or cusum or ewma),
            "source_name": "fictional_operational_feed", "dataset_id": DATASET_ID,
        })

rule_columns = ["xmr_signal", "cusum_signal", "ewma_signal", "signal_detected"]
window_features_df = pd.DataFrame(rows).drop(columns=rule_columns).sort_values(["run_date", "series_id"]).reset_index(drop=True)
signals_df = window_features_df.merge(
    pd.DataFrame(rows)[["series_id", "run_date"] + rule_columns],
    on=["series_id", "run_date"], validate="one_to_one",
)
assert len(signals_df) > 100 and signals_df["signal_detected"].nunique() == 2
assert (signals_df["signal_detected"] == signals_df[["xmr_signal", "cusum_signal", "ewma_signal"]].any(axis=1)).all()
print(f"{len(signals_df):,} labeled windows; rule signal rate: {signals_df['signal_detected'].mean():.1%}")
print(signals_df[["xmr_signal", "cusum_signal", "ewma_signal"]].mean().map(lambda v: f"{v:.1%}").to_string())
show_dataframe_stage(
    "2. Historical measurements — window_features_df", window_features_df,
    ["run_date", "window_start", "window_end", "last_value", "window_mean", "window_std", "last_5_mean"],
    grain="one queue's preceding 25-day window, evaluated on run_date",
    change="TRANSFORM: summarize history; add window dates, mean, variation and recent average. The first 115 dates per queue need more history. last_value is from window_end, not run_date.",
)
show_dataframe_stage(
    "3. Rule flags added — signals_df", signals_df,
    ["run_date", "window_mean", "ucl", "lcl", "xmr_signal", "cusum_signal", "ewma_signal", "signal_detected"],
    grain="the same completed historical window",
    change="ADD four label columns to window_features_df: three rule flags and their logical OR, signal_detected. Control limits were calculated with the historical measurements; the counts are unchanged.",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 5: Test whether ML can copy the rule label
# MAGIC **Plain English:** Train on earlier dates and ask a random forest if the SPC rules fired on later dates.
# MAGIC **Technique:** A chronological holdout separates the 25-day train and test windows. The earlier 90-day reference histories can overlap; all inputs still precede their run date. Compare accuracy with constant predictions, and read precision, recall, F1, and ROC AUC.
# MAGIC **Output:** `predictions_df` holds the model calls; `metrics` holds both model and simple-baseline scores.
# MAGIC **Decision:** This classification is useful only if it adds something to directly running the known rules. A strong AUC alone does not establish added mission value.

# COMMAND ----------

# DBTITLE 1,Train chronologically and compare against a simple baseline
# Each feature is known at the end of the 25-day window; no future values are inputs.
# The evaluation starts 25 business days after training ends, so the 25-day
# measurement windows cannot overlap. Earlier 90-day reference histories can overlap.
# Metrics describe rule reproduction on fictional data, not real-world performance.
feature_cols = ["window_mean", "window_std", "window_range", "mr_mean",
                "last_value", "last_5_mean", "last_5_std", "trend", "baseline_mean", "baseline_sigma",
                "max_baseline_deviation", "max_window_deviation"]
unique_dates = sorted(signals_df["run_date"].unique())
cut_index = int(len(unique_dates) * 0.67)
test_index = cut_index + WINDOW
assert test_index < len(unique_dates) - 20
train_df = signals_df[signals_df["run_date"] <= unique_dates[cut_index]].copy()
test_df = signals_df[signals_df["run_date"] >= unique_dates[test_index]].copy()
assert train_df["window_end"].max() < test_df["window_start"].min()
assert train_df["signal_detected"].nunique() == 2 and test_df["signal_detected"].nunique() == 2

X_train, X_test = train_df[feature_cols], test_df[feature_cols]
y_train = train_df["signal_detected"].astype(int)
y_test = test_df["signal_detected"].astype(int)
CLASSIFIER_CONFIG = dict(n_estimators=100, max_depth=8, min_samples_leaf=3,
                         class_weight="balanced", random_state=SEED, n_jobs=-1)
model = RandomForestClassifier(**CLASSIFIER_CONFIG)
model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]
majority_class = int(y_train.mean() >= 0.5)
baseline_pred = np.full(len(y_test), majority_class)
metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred, zero_division=0),
    "recall": recall_score(y_test, pred, zero_division=0),
    "f1": f1_score(y_test, pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, prob),
    "majority_baseline_accuracy": accuracy_score(y_test, baseline_pred),
    "always_signal_accuracy": float(y_test.mean()),
    "always_no_signal_accuracy": float(1 - y_test.mean()),
    "test_signal_rate": float(y_test.mean()),
}
print(f"Training: {len(train_df)} rows through {train_df.run_date.max().date()}")
print(f"Test: {len(test_df)} rows from {test_df.run_date.min().date()} to {test_df.run_date.max().date()}")
print(f"Training signal rate: {y_train.mean():.1%}; test signal rate: {y_test.mean():.1%}")
print(pd.Series(metrics).round(3).to_string())
if metrics["accuracy"] <= max(metrics["majority_baseline_accuracy"], metrics["always_signal_accuracy"]):
    print("The classifier did not beat both simple constant baselines on this test. Use the rules directly.")
else:
    print("The classifier beat constant baselines on this test; this measures rule reproduction, not future warning.")

predictions_df = test_df[["series_id", "run_date", "signal_detected", "xmr_signal", "cusum_signal", "ewma_signal"]].copy()
predictions_df["model_signal"] = pred.astype(bool)
predictions_df["model_probability"] = prob
predictions_df["dataset_id"] = DATASET_ID
show_dataframe_stage(
    "4. Classifier results added — predictions_df", predictions_df,
    ["run_date", "signal_detected", "model_signal", "model_probability"],
    grain="one held-out historical window",
    change="FILTER to later test dates; ADD model_signal and model_probability beside the known rule label. This branch classifies the past window.",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 6: Predict one future business-day count
# MAGIC **Plain English:** Use information available yesterday to estimate today's workload before today's count arrives.
# MAGIC **Technique:** A random forest regressor learns from past counts, recent trend, weekday, and series. A later-date test with a gap compares mean absolute error (MAE) to the simple five-day moving average.
# MAGIC **Output:** `forecast_results_df` holds actual and predicted counts; `forecast_metrics` reports errors in **daily-count units**.
# MAGIC **Developer note:** This is a **one-business-day forecast**. It does not predict a confirmed incident, and the test score comes only from fabricated data.

# COMMAND ----------

# DBTITLE 1,Forecast the next business day's count (separate from rule classification)
# Each run_date's features use only the 25 business days ending the previous day.
# The target is the count observed on run_date, which was unavailable when predicting.
# The last value and trailing mean are transparent baselines; no rule label is used as an input.
forecast_df = signals_df.merge(
    daily_df[["series_id", "run_date", "daily_count"]],
    on=["series_id", "run_date"], validate="one_to_one",
)
forecast_df["weekday"] = forecast_df["run_date"].dt.weekday
forecast_features = ["series_id", "weekday", "last_value", "last_5_mean",
                     "window_mean", "window_std", "trend", "baseline_mean"]
forecast_X = pd.get_dummies(forecast_df[forecast_features], columns=["series_id"], dtype=float)
forecast_train = forecast_df["run_date"].le(unique_dates[cut_index])
forecast_test = forecast_df["run_date"].ge(unique_dates[test_index + 1])
assert forecast_df.loc[forecast_train, "run_date"].max() < forecast_df.loc[forecast_test, "window_start"].min()
forecast_model = RandomForestRegressor(
    n_estimators=120, max_depth=7, min_samples_leaf=4, random_state=SEED, n_jobs=-1
)
forecast_model.fit(forecast_X.loc[forecast_train], forecast_df.loc[forecast_train, "daily_count"])
forecast_values = np.maximum(0, forecast_model.predict(forecast_X.loc[forecast_test]))
forecast_actual = forecast_df.loc[forecast_test, "daily_count"].to_numpy()
forecast_metrics = {
    "forecast_mae": float(mean_absolute_error(forecast_actual, forecast_values)),
    "forecast_bias": float(np.mean(forecast_values - forecast_actual)),
    "last_value_baseline_mae": float(mean_absolute_error(
        forecast_actual, forecast_df.loc[forecast_test, "last_value"])),
    "trailing_mean_baseline_mae": float(mean_absolute_error(
        forecast_actual, forecast_df.loc[forecast_test, "last_5_mean"])),
}
forecast_results_df = forecast_df.loc[forecast_test, ["series_id", "run_date", "daily_count", "dataset_id"]].copy()
forecast_results_df["predicted_count"] = forecast_values
forecast_results_df["trailing_mean_baseline"] = forecast_df.loc[forecast_test, "last_5_mean"].to_numpy()
forecast_results_df["absolute_error"] = np.abs(forecast_actual - forecast_values)
show_dataframe_stage(
    "5. Forecast inputs joined — forecast_df", forecast_df,
    ["run_date", "window_end", "last_value", "last_5_mean", "weekday", "daily_count"],
    grain="one queue and target business date",
    change="SEPARATE BRANCH: join signals_df to daily_df on series_id + run_date; add weekday. daily_count is the later observed target, never a model input.",
)
show_dataframe_stage(
    "6. Forecast results added — forecast_results_df", forecast_results_df,
    ["run_date", "daily_count", "predicted_count", "trailing_mean_baseline", "absolute_error"],
    grain="one held-out forecast target date",
    change="FILTER to forecast test dates; ADD predicted count, simple baseline and absolute error. No classifier prediction is used by this model.",
)
print("Next-business-day count forecast: held-out later dates, with a 25-day gap")
print(pd.Series(forecast_metrics).round(2).to_string())
if forecast_metrics["forecast_mae"] >= forecast_metrics["trailing_mean_baseline_mae"]:
    print("The forecast did not beat the trailing-mean baseline here; show the comparison honestly.")
else:
    print("The forecast beat the trailing-mean baseline on this demo test only.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 7: Draw the evidence
# MAGIC **Plain English:** The first chart shows recent counts and XmR limits. The second shows which rule labels the classifier matched or missed. The third compares the forecast with later observed counts.
# MAGIC **Technique:** Matplotlib draws the time series; a confusion matrix displays true and false classifier calls on the held-out dates.
# MAGIC **Read carefully:** A circle means **any** rule fired somewhere in the past 25-day window. The point under the circle does not have to cross the displayed XmR limit.
# MAGIC **Dataframe walkthrough:** This cell plots existing results. It adds no analytical columns.
# MAGIC **Developer note:** These charts support explanation; review actual rows and rule flags before interpreting a cause.

# COMMAND ----------

# DBTITLE 1,Show the control chart and classification results
chosen = "Intake A"
chart_df = signals_df[signals_df.series_id == chosen].tail(90)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(chart_df.run_date, chart_df.last_value, color="#145A7D", label="Window's last observed count")
ax.plot(chart_df.run_date, chart_df.ucl, color="#B8603C", linestyle="--", label="XmR upper limit")
ax.plot(chart_df.run_date, chart_df.lcl, color="#B8603C", linestyle=":", label="XmR lower limit")
marked = chart_df[chart_df.signal_detected]
ax.scatter(marked.run_date, marked.last_value, facecolor="white", edgecolor="#853B32",
           s=30, linewidth=1.2, label="Rule signal for 25-day window")
ax.set(title="Fictional SPC history — Intake A (last 90 runs)", xlabel="Run date", ylabel="Daily count")
ax.legend(loc="upper left", fontsize=8)
fig.text(0.08, 0.01, "Circles mean any rule fired in the prior 25 days; the plotted point need not cross an XmR limit.",
         fontsize=8, color="#444444")
fig.autofmt_xdate()
fig.subplots_adjust(bottom=0.36)
plt.show()

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["No signal", "Signal"],
                                        cmap="Blues", colorbar=False, ax=ax)
ax.set_title(f"Demo test: {len(y_test)} labeled windows")
plt.tight_layout()
plt.show()

forecast_chart = forecast_results_df[forecast_results_df.series_id == chosen].tail(55)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(forecast_chart.run_date, forecast_chart.daily_count, label="Observed", color="#145A7D")
ax.plot(forecast_chart.run_date, forecast_chart.predicted_count, label="Predicted before observation", color="#B8603C")
ax.set(title="Next-business-day forecast — held-out demo dates", xlabel="Date", ylabel="Daily count")
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 8: Give people a manageable review list
# MAGIC **Plain English:** Consecutive days of rule flags for one series become one review episode instead of a new alert every day.
# MAGIC **Technique:** Group-by and a cumulative count identify each contiguous run of signals. Basic data and model health indicators go into `monitoring_df`.
# MAGIC **Output:** `review_df` shows the 15 episodes with the latest signal dates, their rules, and a pending disposition. The printed total counts **all** episodes.
# MAGIC **Developer note:** This is a proposed review queue. It does not send notifications, record a completed analyst decision, or automate an operational action.

# COMMAND ----------

# DBTITLE 1,Consolidate persistent signals into analyst review episodes
# A 25-day rule window can stay flagged across many runs. One episode is one
# consecutive span of flagged windows within a series, not one email per day.
ordered_signals = signals_df.sort_values(["series_id", "run_date"]).copy()
previous_flag = ordered_signals.groupby("series_id")["signal_detected"].shift(fill_value=False)
ordered_signals["episode_id"] = (ordered_signals["signal_detected"] & ~previous_flag).groupby(
    ordered_signals["series_id"]
).cumsum()
episode_rows = ordered_signals[ordered_signals["signal_detected"]].copy()
review_df = episode_rows.groupby(["series_id", "episode_id"], as_index=False).agg(
    run_date=("run_date", "min"), last_signal_date=("run_date", "max"),
    signal_windows=("run_date", "size"), window_end=("window_end", "max"),
    last_value=("last_value", "last"), xmr_signal=("xmr_signal", "max"),
    cusum_signal=("cusum_signal", "max"), ewma_signal=("ewma_signal", "max"),
    source_name=("source_name", "first"), dataset_id=("dataset_id", "first"),
)
review_df["rule_fired"] = review_df.apply(
    lambda row: ", ".join(name for name in ("xmr", "cusum", "ewma") if row[f"{name}_signal"]), axis=1
)
review_df["disposition"] = "Pending analyst review"
review_df = review_df[["series_id", "episode_id", "run_date", "last_signal_date", "signal_windows",
                       "window_end", "rule_fired", "last_value", "disposition", "source_name", "dataset_id"]]
total_review_episodes = len(review_df)
review_df = review_df.sort_values(["last_signal_date", "series_id"]).tail(15).reset_index(drop=True)
monitoring_df = pd.DataFrame([{
    "dataset_id": DATASET_ID,
    "source_rows": len(daily_df),
    "missing_counts": int(daily_df["daily_count"].isna().sum()),
    "rule_signal_rate": float(signals_df["signal_detected"].mean()),
    "raw_signal_windows": int(signals_df["signal_detected"].sum()),
    "review_episodes": total_review_episodes,
    "test_model_signal_rate": float(predictions_df["model_signal"].mean()),
    "test_rule_disagreement_rate": float((predictions_df["model_signal"] != predictions_df["signal_detected"]).mean()),
    "classifier_accuracy": float(metrics["accuracy"]),
    "majority_baseline_accuracy": float(metrics["majority_baseline_accuracy"]),
    "always_signal_accuracy": float(metrics["always_signal_accuracy"]),
    "forecast_mae": forecast_metrics["forecast_mae"],
    "trailing_mean_baseline_mae": forecast_metrics["trailing_mean_baseline_mae"],
}])
print("Consolidated review episodes (15 latest signal dates; no automatic action):")
display(review_df) if "display" in globals() else print(review_df.tail(5).to_string(index=False))
# Show one real review episode and the daily signal rows that formed it.
example_episode = review_df[review_df.series_id == DEMO_SERIES].tail(1)
episode_members = episode_rows.merge(example_episode[["series_id", "episode_id"]], on=["series_id", "episode_id"], validate="many_to_one").sort_values("run_date")
show_dataframe_stage(
    "7a. Before grouping — episode_rows", episode_rows,
    ["run_date", "signal_detected", "episode_id"], sample=episode_members.tail(DATAFRAME_PREVIEW_ROWS),
    grain="one flagged daily window",
    change=f"FILTER: one example episode has {len(episode_members)} contributing daily windows; show its final five. These rows will become one review row.",
)
show_dataframe_stage(
    "7b. After grouping — review_df", review_df,
    ["episode_id", "run_date", "last_signal_date", "signal_windows", "rule_fired", "disposition"], sample=example_episode,
    grain="one consecutive episode of flagged windows",
    change="GROUP BY series_id + episode_id; summarize dates and window count, collect rules, then add a pending disposition. This is a change in row meaning, not a one-to-one join.",
)
print("Monitoring snapshot:")
display(monitoring_df) if "display" in globals() else print(monitoring_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 9: Add severity and inspect the data
# MAGIC **Plain English:** A yes/no flag tells us to look. Zone checks add an indication of how unusual the latest pattern is.
# MAGIC **Method:** Moving-range sigma, a latest-point three-sigma check, 2-of-3 beyond two sigma, 4-of-5 beyond one sigma, and eight on one side. These are demo choices, not calibrated agency thresholds.
# MAGIC **Show:** Severity counts, per-series distributions and correlations. A correlation does not establish cause.

# COMMAND ----------
# DBTITLE 1,Zone severity and exploratory analysis
def zone_check(values):
    center = float(np.mean(values))
    sigma = max(float(np.mean(np.abs(np.diff(values)))) / 1.128, 1e-6)
    z = (np.asarray(values) - center) / sigma
    if abs(z[-1]) > 3:
        return "CRITICAL", "latest point beyond 3 sigma"
    if max((z[-3:] > 2).sum(), (z[-3:] < -2).sum()) >= 2:
        return "WARNING", "2 of 3 beyond 2 sigma on one side"
    if np.all(z[-8:] > 0) or np.all(z[-8:] < 0):
        return "WARNING", "8 points on one side"
    if max((z[-5:] > 1).sum(), (z[-5:] < -1).sum()) >= 4:
        return "WATCH", "4 of 5 beyond 1 sigma on one side"
    if abs(z[-1]) > 1:
        return "WATCH", "latest point beyond 1 sigma"
    return "NORMAL", "no zone pattern"

zone_rows = []
for sid, group in daily_df.groupby("series_id"):
    group = group.sort_values("run_date").reset_index(drop=True)
    for end in range(BASELINE_DAYS + WINDOW, len(group)):
        severity, reason = zone_check(group.daily_count.iloc[end-WINDOW:end].to_numpy())
        zone_rows.append({"series_id": sid, "run_date": group.run_date.iloc[end],
                          "zone_severity": severity, "zone_reason": reason})
zone_df = pd.DataFrame(zone_rows)
series_profile_df = daily_df.groupby("series_id").daily_count.agg(["count", "mean", "std", "min", "max"]).reset_index()
print(series_profile_df.round(2).to_string(index=False))
print(zone_df.zone_severity.value_counts().to_string())
severity_view_df = signals_df[["series_id", "run_date", "signal_detected"]].merge(
    zone_df, on=["series_id", "run_date"], validate="one_to_one",
)
show_dataframe_stage(
    "8. Severity joined — severity_view_df", severity_view_df,
    ["run_date", "signal_detected", "zone_severity", "zone_reason"],
    grain="one completed historical window",
    change="JOIN a separate zone-check result onto the rule flag using series_id + run_date. Add severity and its reason; the original rule label stays unchanged.",
)
pivot_counts = daily_df.pivot(index="run_date", columns="series_id", values="daily_count")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for sid in pivot_counts:
    axes[0].hist(pivot_counts[sid], bins=25, alpha=.45, label=sid)
axes[0].set(title="Fictional daily-count distributions", xlabel="Daily count", ylabel="Days")
axes[0].legend(fontsize=8)
corr = pivot_counts.corr()
im = axes[1].imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
axes[1].set_xticks(range(len(corr)), corr.columns, rotation=20)
axes[1].set_yticks(range(len(corr)), corr.index)
for i in range(len(corr)):
    for j in range(len(corr)):
        axes[1].text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center")
axes[1].set_title("Association between series; not causality")
fig.colorbar(im, ax=axes[1], fraction=.04)
plt.tight_layout(); plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cell 10: Find which subgroup changed
# MAGIC **Plain English:** Split each fictional daily count between two offices, then investigate shifts inside each office.
# MAGIC **Method:** Counts reconcile to the parent series. Past-only rolling z-scores and workload shares support subgroup investigation; cardinality, placeholder and duplicate-dimension checks keep categories useful.
# MAGIC **Boundary:** These are invented office and channel labels, with no agency records or claims about people.

# COMMAND ----------
# DBTITLE 1,Subgroup profiles and anomaly evidence
subgroup_rows = []
for sid, group in daily_df.groupby("series_id"):
    for idx, row in enumerate(group.sort_values("run_date").itertuples()):
        share = .58 + (.18 if 355 <= idx < 370 else 0)
        north = int(round(row.daily_count * share))
        for office, count in [("North", north), ("South", row.daily_count-north)]:
            subgroup_rows.append({"series_id": sid, "run_date": row.run_date, "office": office,
                                  "channel": "Online" if office == "North" else "Assisted",
                                  "daily_count": count, "dataset_id": DATASET_ID})
subgroup_df = pd.DataFrame(subgroup_rows).sort_values(["series_id", "office", "run_date"])
subgroup_df["prior_mean"] = subgroup_df.groupby(["series_id", "office"]).daily_count.transform(lambda x: x.shift(1).rolling(25).mean())
subgroup_df["prior_std"] = subgroup_df.groupby(["series_id", "office"]).daily_count.transform(lambda x: x.shift(1).rolling(25).std()).clip(lower=1)
subgroup_df["zscore"] = (subgroup_df.daily_count-subgroup_df.prior_mean)/subgroup_df.prior_std
subgroup_df["subgroup_anomaly"] = subgroup_df.zscore.abs().gt(3)
subgroup_df["workload_share"] = subgroup_df.daily_count / subgroup_df.groupby(["series_id", "run_date"]).daily_count.transform("sum").clip(lower=1)
reconciled = subgroup_df.groupby(["series_id", "run_date"]).daily_count.sum().sort_index()
assert reconciled.equals(daily_df.set_index(["series_id", "run_date"]).daily_count.sort_index())
dimension_profile_df = pd.DataFrame([{"dimension": col, "distinct_values": subgroup_df[col].nunique(),
                                    "missing": int(subgroup_df[col].isna().sum()),
                                    "placeholder_rows": int(subgroup_df[col].isin(["", "UNKNOWN", "N/A"]).sum())}
                                   for col in ["office", "channel"]])
# Office and channel map one-to-one in this deliberately simple fixture; do not count them as independent evidence.
dimension_profile_df["use_for_modeling"] = [True, False]
print(dimension_profile_df.to_string(index=False))
print("Channel is redundant with office in this fixture; retain office for analysis.")
print(subgroup_df[subgroup_df.subgroup_anomaly].tail(8).to_string(index=False))
subgroup_preview = subgroup_df[(subgroup_df.series_id == DEMO_SERIES) & subgroup_df.run_date.isin(DEMO_PREVIEW_DATES.tail(3))].sort_values(["run_date", "office"])
show_dataframe_stage(
    "9a. Parent counts before split — daily_df", daily_df,
    ["run_date", "daily_count"],
    sample=daily_df[(daily_df.series_id == DEMO_SERIES) & daily_df.run_date.isin(DEMO_PREVIEW_DATES.tail(3))].sort_values("run_date"),
    grain="one queue on one date", change="BEFORE: three of the same source dates, one total count per date.",
)
show_dataframe_stage(
    "9b. Office rows after split — subgroup_df", subgroup_df,
    ["run_date", "office", "daily_count", "workload_share", "prior_mean", "zscore", "subgroup_anomaly"], sample=subgroup_preview,
    grain="one office within a queue on one date",
    change="EXPAND each parent into North and South rows; ADD workload share, prior mean, z-score and anomaly flag. The two office counts sum exactly to the parent count on each date.",
)
fig, ax = plt.subplots(figsize=(11, 4))
for office, group in subgroup_df[subgroup_df.series_id == "Intake A"].groupby("office"):
    ax.plot(group.run_date.tail(90), group.workload_share.tail(90), label=office)
ax.set(title="Fictional workload mix — Intake A", ylabel="Share of parent daily count", xlabel="Date")
ax.legend(); fig.autofmt_xdate(); plt.tight_layout(); plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cell 11: Compare classifiers across five later periods
# MAGIC **Plain English:** Compare random forest and logistic regression over several time periods, not just one favorable split.
# MAGIC **Method:** Five expanding training windows, shared date boundaries across series, and a 25-business-day separation of measurement windows. Scaling fits only on training rows. The RF settings exactly match Cell 5; these later-period folds are a separate diagnostic, not an untouched holdout for model selection. Compare each fold with a training-majority baseline.
# MAGIC **Decision:** Both models still imitate known rules. Fold scores and feature importance describe this experiment, not mission value or causality.

# COMMAND ----------
# DBTITLE 1,Walk-forward classification and feature importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score

fold_specs = []
fold_metrics = []
fold_predictions = []
for fold, start in enumerate(np.linspace(155, len(unique_dates)-30, 5, dtype=int), 1):
    train_dates_end = start-WINDOW-1
    fold_train = signals_df.run_date.le(unique_dates[train_dates_end])
    fold_test = signals_df.run_date.between(unique_dates[start], unique_dates[start+24])
    tr, te = signals_df.loc[fold_train], signals_df.loc[fold_test]
    assert tr.window_end.max() < te.window_start.min()
    fold_specs.append((fold, fold_train, fold_test))
    candidates = {
        "Random forest": RandomForestClassifier(**CLASSIFIER_CONFIG),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
    }
    for name, candidate in candidates.items():
        candidate.fit(tr[feature_cols], tr.signal_detected.astype(int))
        calls = candidate.predict(te[feature_cols])
        scores = candidate.predict_proba(te[feature_cols])[:,1]
        majority = int(tr.signal_detected.mean() >= .5)
        fold_metrics.append({"fold": fold, "model": name, "train_end": tr.run_date.max(),
                             "test_start": te.run_date.min(), "test_end": te.run_date.max(), "rows": len(te),
                             "accuracy": accuracy_score(te.signal_detected, calls),
                             "f1": f1_score(te.signal_detected, calls, zero_division=0),
                             "auc": roc_auc_score(te.signal_detected, scores) if te.signal_detected.nunique()==2 else np.nan,
                             "majority_accuracy": accuracy_score(te.signal_detected, np.full(len(te), majority))})
        fold_predictions.extend([{"fold":fold,"model":name,"truth":bool(y),"prediction":bool(p)} for y,p in zip(te.signal_detected,calls)])
        if name == "Logistic regression": logistic_model = candidate
fold_metrics_df = pd.DataFrame(fold_metrics)
# Save a logistic model fitted on the same training split as the primary RF.
logistic_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
logistic_model.fit(X_train, y_train)
print(fold_metrics_df.round(3).to_string(index=False))
importance = permutation_importance(model, X_test, y_test, n_repeats=3, random_state=SEED, scoring="accuracy")
importance_df = pd.DataFrame({"feature":feature_cols,"accuracy_drop":importance.importances_mean}).sort_values("accuracy_drop", ascending=False)
fig, axes = plt.subplots(1,2,figsize=(12,4))
for name,g in fold_metrics_df.groupby("model"):
    axes[0].plot(g.fold,g.accuracy,marker="o",label=name)
base=fold_metrics_df[fold_metrics_df.model=="Random forest"]
axes[0].plot(base.fold,base.majority_accuracy,linestyle="--",label="Training-majority baseline")
axes[0].set(title="Five later-period tests",xlabel="Fold",ylabel="Accuracy",ylim=(0,1)); axes[0].legend(fontsize=8)
importance_df.head(6).iloc[::-1].plot.barh(x="feature",y="accuracy_drop",ax=axes[1],legend=False)
axes[1].set(title="Permutation importance on held-out rows",xlabel="Accuracy drop when shuffled",ylabel="")
plt.tight_layout(); plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cell 12: Forecast five business days and test the alternatives
# MAGIC **Plain English:** Compare five-day ARIMA and Holt-Winters forecasts with the simple guess that the next five values repeat the previous five. Then generate actual future-date forecasts, including offices.
# MAGIC **Method:** Six nonoverlapping forecast origins, five steps each, fit only on history at each origin. The last origin is held aside for lifecycle comparison; earlier origins choose the candidate. US federal holidays are excluded from the calendar.
# MAGIC **Boundary:** Five business observations are a compact seasonal approximation; holiday weeks can differ. Show errors and keep the baseline if it wins.

# COMMAND ----------
# DBTITLE 1,Five-day ARIMA Holt-Winters and subgroup forecasts
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

def five_day_predictions(values):
    values = np.asarray(values, dtype=float)
    predictions = {"Seasonal naive": np.tile(values[-5:], 1)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arima = ARIMA(values[-180:], order=(1,0,0), trend="c").fit()
        hw = ExponentialSmoothing(values[-180:], trend="add", damped_trend=True, seasonal="add", seasonal_periods=5,
                                  initialization_method="estimated").fit(optimized=True)
    predictions["ARIMA"] = np.maximum(0, np.asarray(arima.forecast(5)))
    predictions["Holt-Winters"] = np.maximum(0, np.asarray(hw.forecast(5)))
    return predictions

history_rows, future_rows = [], []
for sid, group in daily_df.groupby("series_id"):
    group = group.sort_values("run_date").reset_index(drop=True)
    values = group.daily_count.to_numpy(dtype=float)
    for end in range(len(group)-60, len(group), 10):
        origin = group.run_date.iloc[end-1]
        for method, vals in five_day_predictions(values[:end]).items():
            for h, value in enumerate(vals,1):
                history_rows.append({"dataset_id":DATASET_ID,"series_id":sid,"model":method,"origin_date":origin,
                                     "target_date":group.run_date.iloc[end+h-1],"horizon":h,
                                     "prediction":float(value),"actual":float(values[end+h-1]),
                                     "origin_count":float(values[end-1])})
    future_dates=pd.date_range(group.run_date.iloc[-1]+BUSINESS_DAY,periods=5,freq=BUSINESS_DAY)
    for method,vals in five_day_predictions(values).items():
        future_rows.extend([{"series_id":sid,"office":"All","model":method,"origin_date":group.run_date.iloc[-1],
                             "target_date":d,"horizon":h,"prediction":float(v),"dataset_id":DATASET_ID}
                            for h,(d,v) in enumerate(zip(future_dates,vals),1)])
forecast_history_df=pd.DataFrame(history_rows)
forecast_history_df["absolute_error"]=(forecast_history_df.actual-forecast_history_df.prediction).abs()
selection_cutoff=forecast_history_df.origin_date.max()
selection_scores=forecast_history_df[forecast_history_df.origin_date<selection_cutoff].groupby("model").absolute_error.mean()
selected_forecaster=selection_scores.idxmin()
five_day_metrics_df=forecast_history_df.groupby("model").absolute_error.agg(["mean","count"]).rename(columns={"mean":"mae","count":"forecast_rows"}).reset_index()
for (sid,office),group in subgroup_df.groupby(["series_id","office"]):
    group=group.sort_values("run_date")
    vals=five_day_predictions(group.daily_count)[selected_forecaster]
    future_dates=pd.date_range(group.run_date.iloc[-1]+BUSINESS_DAY,periods=5,freq=BUSINESS_DAY)
    future_rows.extend([{"series_id":sid,"office":office,"model":selected_forecaster,"origin_date":group.run_date.iloc[-1],
                         "target_date":d,"horizon":h,"prediction":float(v),"dataset_id":DATASET_ID}
                        for h,(d,v) in enumerate(zip(future_dates,vals),1)])
future_forecasts_df=pd.DataFrame(future_rows)
print(five_day_metrics_df.round(3).to_string(index=False))
print("Candidate selected from first five origins:",selected_forecaster)
print("Subgroup forecasts fit independently and need not sum to the parent forecast.")
print(future_forecasts_df.tail(10).round(2).to_string(index=False))
fig,ax=plt.subplots(figsize=(11,4))
g=daily_df[daily_df.series_id=="Intake A"].tail(25)
ax.plot(g.run_date,g.daily_count,label="Observed fictional counts",color="#145A7D")
for method,group in future_forecasts_df[(future_forecasts_df.series_id=="Intake A")&(future_forecasts_df.office=="All")].groupby("model"):
    ax.plot(group.target_date,group.prediction,marker="o",label=method)
ax.set(title="Five business days beyond the latest observation",xlabel="Date",ylabel="Daily count")
ax.legend(fontsize=8);fig.autofmt_xdate();plt.tight_layout();plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cell 13: Look for unusual patterns without rule labels
# MAGIC **Plain English:** Isolation Forest finds unusual combinations of measurements. It can disagree with SPC, which gives the analyst another question to investigate.
# MAGIC **Method:** Five chronological folds, a contamination setting estimated only from training data and capped at 10%, then final fitting on the earlier training set. SPC agreement is a proxy check, not incident-detection accuracy.

# COMMAND ----------
# DBTITLE 1,Isolation Forest and disagreement analysis
from sklearn.ensemble import IsolationForest
iso_rows=[]
for fold,trmask,temask in fold_specs:
    tr,te=signals_df.loc[trmask],signals_df.loc[temask]
    contamination=float(np.clip(tr.signal_detected.mean()*.10,.02,.10))
    detector=IsolationForest(n_estimators=100,contamination=contamination,random_state=SEED,n_jobs=-1).fit(tr[feature_cols])
    flags=detector.predict(te[feature_cols])==-1
    iso_rows.append({"fold":fold,"contamination":contamination,"rows":len(te),"anomaly_rate":float(flags.mean()),
                     "rule_agreement":float((flags==te.signal_detected.to_numpy()).mean()),
                     "rule_proxy_f1":f1_score(te.signal_detected,flags,zero_division=0)})
iso_fold_df=pd.DataFrame(iso_rows)
final_contamination=float(np.clip(y_train.mean()*.10,.02,.10))
iso_model=IsolationForest(n_estimators=100,contamination=final_contamination,random_state=SEED,n_jobs=-1).fit(X_train)
anomaly_df=test_df[["series_id","run_date","signal_detected","dataset_id"]].copy()
anomaly_df["anomaly_score"]=-iso_model.score_samples(X_test)
anomaly_df["model_anomaly"]=iso_model.predict(X_test)==-1
anomaly_df["disagrees_with_spc"]=anomaly_df.model_anomaly != anomaly_df.signal_detected
print(iso_fold_df.round(3).to_string(index=False))
print(anomaly_df.groupby(["signal_detected","model_anomaly"]).size().rename("windows").to_string())
fig,ax=plt.subplots(figsize=(11,3.5))
g=anomaly_df[anomaly_df.series_id=="Intake A"]
ax.plot(g.run_date,g.anomaly_score,label="Isolation Forest unusualness score")
flagged=g[g.model_anomaly];ax.scatter(flagged.run_date,flagged.anomaly_score,color="#B8603C",label="Model flags")
ax.set(title="Anomaly scores on later fictional observations",xlabel="Run date",ylabel="Higher = more unusual")
ax.legend(fontsize=8);fig.autofmt_xdate();plt.tight_layout();plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cell 14: Trace fictional inputs upstream
# MAGIC **Plain English:** Follow a count back through the tables that produced it and inspect whether an upstream refresh was late.
# MAGIC **Method:** Breadth-first graph traversal up to five hops, column mappings and timestamp-based freshness checks over fixture metadata.
# MAGIC **Boundary:** The graph is a fictional fixture, not discovered Unity Catalog lineage. The traversal and checks execute locally.

# COMMAND ----------
# DBTITLE 1,Fixture lineage and upstream freshness
lineage_edges_df=pd.DataFrame([
    ("demo.daily_counts","demo.normalized_actions"),("demo.normalized_actions","demo.raw_actions"),
    ("demo.normalized_actions","demo.office_lookup"),("demo.raw_actions","demo.batch_landing"),
    ("demo.batch_landing","demo.transfer_manifest"),("demo.transfer_manifest","demo.source_export"),
],columns=["target","source"])
column_lineage_df=pd.DataFrame([
    ("demo.daily_counts.daily_count","demo.normalized_actions.action_id"),
    ("demo.normalized_actions.office","demo.office_lookup.office_code"),
],columns=["target_column","source_column"])
def trace_upstream(root, edges, max_hops=5):
    pending=[(root,0,root)];seen={root};result=[]
    while pending:
        target,hop,path=pending.pop(0)
        if hop>=max_hops:continue
        for source in edges.loc[edges.target==target,"source"]:
            result.append({"table":source,"hop":hop+1,"path":path+" -> "+source})
            if source not in seen:
                seen.add(source);pending.append((source,hop+1,path+" -> "+source))
    return pd.DataFrame(result)
lineage_df=trace_upstream("demo.daily_counts",lineage_edges_df)
# Use a historical replay date, not wall-clock time, so this fixture is reproducible.
replay_date=pd.Timestamp(daily_df.run_date.max())+pd.Timedelta(hours=9)
upstream_profile_df=lineage_df[["table","hop"]].copy()
upstream_profile_df["last_refresh"]=[replay_date-pd.Timedelta(hours=30 if "raw_actions" in t else 2) for t in upstream_profile_df.table]
upstream_profile_df["age_hours"]=(replay_date-upstream_profile_df.last_refresh).dt.total_seconds()/3600
upstream_profile_df["late_refresh"]=upstream_profile_df.age_hours>24
upstream_profile_df["inferred_role"]=upstream_profile_df.table.map(lambda x:"Reference data" if "lookup" in x else "Data pipeline")
print(lineage_df.to_string(index=False));print(column_lineage_df.to_string(index=False))
print(upstream_profile_df.to_string(index=False))

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cell 15: Build an investigation hypothesis
# MAGIC **Plain English:** Put the triggered rules, office changes, nearby events and upstream evidence beside each review episode.
# MAGIC **Method:** Evidence joins produce reasons, with temporal checks before linking a late refresh. An event near a signal is an association, not a proven cause.
# MAGIC **Show:** An invented capacity change and a delayed input batch. Evidence labels keep fixture facts distinct from analyst conclusions.

# COMMAND ----------
# DBTITLE 1,Root-cause hypotheses and operational events
change_date=daily_df[daily_df.series_id=="Intake A"].sort_values("run_date").run_date.iloc[355]
events_df=pd.DataFrame([
    {"event_id":"fixture-capacity","event_date":change_date,"series_id":"Intake A","event_type":"Capacity change","description":"Fictional office workload reallocation"},
    {"event_id":"fixture-delay","event_date":replay_date.normalize(),"series_id":"Intake A","event_type":"Input delay","description":"Fictional raw-input batch arrived late"},
])
hypotheses=[]
for row in review_df.itertuples():
    nearby=events_df[(events_df.series_id==row.series_id)&events_df.event_date.between(row.run_date-pd.Timedelta(days=7),row.last_signal_date+pd.Timedelta(days=7))]
    sub=subgroup_df[(subgroup_df.series_id==row.series_id)&subgroup_df.run_date.between(row.run_date,row.last_signal_date)]
    top=sub.loc[sub.zscore.abs().idxmax()] if sub.zscore.notna().any() else None
    late=bool(upstream_profile_df.late_refresh.any() and row.run_date <= replay_date.normalize() <= row.last_signal_date+pd.Timedelta(days=1))
    reason="Rule signal requires investigation"
    if len(nearby):reason="Nearby fixture event; verify timing and mechanism"
    elif top is not None and abs(top.zscore)>3:reason="Localized office deviation; inspect workload mix"
    if late:reason+="; late fixture refresh also needs inspection"
    hypotheses.append({"series_id":row.series_id,"episode_id":row.episode_id,"rule_fired":row.rule_fired,
                       "event_ids":", ".join(nearby.event_id),"top_office":str(top.office) if top is not None else "",
                       "largest_abs_z":float(abs(top.zscore)) if top is not None else 0.,"late_refresh_evidence":late,
                       "hypothesis":reason,"confidence":"Unverified hypothesis","dataset_id":DATASET_ID})
hypotheses_df=pd.DataFrame(hypotheses)
print(events_df.to_string(index=False));print(hypotheses_df.head(6).to_string(index=False))

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cell 16: Assess forecasts after actuals arrive
# MAGIC **Plain English:** Keep past forecasts, match them to later observed counts, and decide whether enough evidence exists to reconsider a model.
# MAGIC **Method:** Idempotent upsert keys preserve distinct forecast origins. Assess MAE, RMSE, sMAPE, bias and direction against the origin count. Earlier runs are superseded for current action; fewer than three actuals cannot trigger retraining.
# MAGIC **Boundary:** This is a replay of several historical forecast runs. It demonstrates the feedback calculation, not a live retraining schedule.

# COMMAND ----------
# DBTITLE 1,Historical forecast assessment and safeguards
HISTORY_KEYS=["dataset_id","series_id","model","origin_date","target_date"]
def upsert_history(existing,incoming,keys):
    return pd.concat([existing,incoming],ignore_index=True).drop_duplicates(keys,keep="last").sort_values(keys).reset_index(drop=True)
forecast_ledger_df=upsert_history(pd.DataFrame(),forecast_history_df,HISTORY_KEYS)
assert len(upsert_history(forecast_ledger_df,forecast_history_df,HISTORY_KEYS))==len(forecast_ledger_df)
def assessment_metrics(group):
    group = group.loc[np.isfinite(group[["prediction", "actual"]]).all(axis=1)].copy()
    if group.empty:
        return dict(n_actuals=0, mae=np.nan, rmse=np.nan, smape=np.nan, bias=np.nan, direction_accuracy=np.nan)
    direction_rows = group.loc[np.isfinite(group.origin_count)]
    err=group.prediction-group.actual
    denominator=(group.prediction.abs()+group.actual.abs())/2
    smape=np.where(denominator>1e-9,err.abs()/denominator.clip(lower=1e-9),0).mean()*100
    return {"n_actuals":len(group),"mae":float(err.abs().mean()),"rmse":float(np.sqrt((err**2).mean())),
            "smape":float(smape),"bias":float(err.mean()),
            "direction_accuracy":float((np.sign(direction_rows.prediction-direction_rows.origin_count)==np.sign(direction_rows.actual-direction_rows.origin_count)).mean()) if len(direction_rows)>=2 else np.nan}
assessment_rows=[]
for keys,g in forecast_ledger_df.groupby(["series_id","model","origin_date"]):
    assessment_rows.append(dict(zip(["series_id","model","origin_date"],keys))|assessment_metrics(g))
assessment_df=pd.DataFrame(assessment_rows).sort_values(["series_id","model","origin_date"])
assessment_df["previous_smape"]=assessment_df.groupby(["series_id","model"]).smape.shift()
assessment_df["latest_origin"]=assessment_df.groupby(["series_id","model"]).origin_date.transform("max")
def retraining_advice(n, current, previous, latest=True):
    if not latest:return "superseded"
    if n<3 or not np.isfinite(current):return "insufficient_data"
    if pd.notna(previous) and current>40 and current>previous:return "architecture_review"
    if pd.notna(previous) and current>previous*1.10:return "review_for_retraining"
    if pd.notna(previous) and current<previous*.95:return "improving"
    return "monitor"
assessment_df["recommendation"]=assessment_df.apply(lambda r:retraining_advice(r.n_actuals,r.smape,r.previous_smape,r.origin_date==r.latest_origin),axis=1)
assessment_df["confidence"]=np.where(assessment_df.n_actuals>=10,"higher sample count","limited sample")
# A deliberately partial arrival exercises the insufficient-data safeguard.
partial_assessment=assessment_metrics(forecast_ledger_df.head(1))
partial_assessment["recommendation"]=retraining_advice(1,partial_assessment["smape"],np.nan)
print(assessment_df[assessment_df.origin_date==assessment_df.latest_origin].round(3).to_string(index=False))
print("One-observation safeguard:",partial_assessment["recommendation"])
fig,ax=plt.subplots(figsize=(11,3.5))
for method,g in assessment_df[assessment_df.series_id=="Intake A"].groupby("model"):
    ax.plot(g.origin_date,g.mae,marker="o",label=method)
ax.set(title="Replay: forecast error by origin — Intake A",xlabel="Forecast origin",ylabel="MAE, daily counts")
ax.legend(fontsize=8);fig.autofmt_xdate();plt.tight_layout();plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cell 17: Challenge the baseline and rehearse withdrawal
# MAGIC **Plain English:** A candidate must beat the current method on a separate period before a person would even consider a pilot. A reviewer can keep the baseline or withdraw a component.
# MAGIC **Method:** Earlier forecast origins choose ARIMA versus Holt-Winters. The last origin compares that candidate with seasonal naive. A 2% improvement is a demo threshold. The small sample supports advisory use only.
# MAGIC **Boundary:** Local lifecycle records and explicit exercise approvals illustrate version selection and rollback. They do not grant production authorization or alter a real registry alias.

# COMMAND ----------
# DBTITLE 1,Champion challenger lifecycle and rollback exercise
last_origin=forecast_ledger_df[forecast_ledger_df.origin_date==selection_cutoff]
last_scores=last_origin.groupby("model").absolute_error.mean()
champion_mae=float(last_scores["Seasonal naive"])
challenger_mae=float(last_scores[selected_forecaster])
improvement=(champion_mae-challenger_mae)/max(champion_mae,1e-9)
lifecycle_df=pd.DataFrame([
    {"component":"Rule-label classifier","version":"rf-v1","recommendation":"withdraw candidate","reason":"Known rules compute the target directly"},
    {"component":"Five-day forecast","version":selected_forecaster,"recommendation":"advisory candidate" if improvement>=.02 else "keep baseline","reason":f"Untouched final-origin relative MAE improvement: {improvement:.1%}; only 15 observations"},
    {"component":"Isolation Forest","version":"if-v1","recommendation":"advisory only","reason":"Rule-label agreement is not verified incident detection"},
])
def rehearse_registry_change(active_version, candidate_version, approved, rollback=False):
    """Pure local exercise: an explicit approval controls a simulated state change."""
    if not approved:
        return active_version, "blocked: exercise approval absent"
    return candidate_version, "rollback" if rollback else "select candidate"

exercise_rows=[{"step":1,"active_version":"baseline-v1","action":"Initial fictional registry state","actor":"Exercise fixture"}]
active_version="baseline-v1"
assert rehearse_registry_change(active_version,"candidate-v1",False)[0]==active_version
for step,target,rollback in [(2,"candidate-v1",False),(3,"baseline-v1",True)]:
    active_version,action=rehearse_registry_change(active_version,target,approved=True,rollback=rollback)
    exercise_rows.append({"step":step,"active_version":active_version,"action":action,"actor":"Simulated reviewer; not a real approval"})
registry_exercise_df=pd.DataFrame(exercise_rows)
assert registry_exercise_df.iloc[-1].active_version==registry_exercise_df.iloc[0].active_version
print(lifecycle_df.to_string(index=False));print(registry_exercise_df.to_string(index=False))

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cell 18: Replay the daily operating loop
# MAGIC **Plain English:** Replay three observation dates, prepare one review notice per episode, suppress repeats, and record an example analyst disposition separately from the pending queue.
# MAGIC **Method:** Stable episode keys make repeated preparation idempotent. The dashboard summarizes the current replay. No email, chat or external notification is sent.
# MAGIC **Boundary:** Job configuration and SQL queries accompany this notebook. Scheduling, delivery connectors and actual analyst authorization still require the target workspace and named owners.

# COMMAND ----------
# DBTITLE 1,Daily replay notification outbox and human review
outbox_rows=[];seen_episodes=set()
replay_days=sorted(signals_df.run_date.unique())[-3:]
for day in replay_days:
    active=episode_rows[episode_rows.run_date==day]
    for row in active.itertuples():
        key=(DATASET_ID,row.series_id,int(row.episode_id))
        outbox_rows.append({"dataset_id":DATASET_ID,"run_date":pd.Timestamp(day),"series_id":row.series_id,
                           "episode_id":int(row.episode_id),"status":"suppressed_repeat" if key in seen_episodes else "prepared_not_sent",
                           "delivery":"No external connector"})
        seen_episodes.add(key)
outbox_df=pd.DataFrame(outbox_rows)
# Explicitly seeded analyst exercise. The actual proposed review_df stays pending.
operator_disposition_df=review_df.head(1)[["series_id","episode_id"]].copy()
operator_disposition_df["disposition"]="Investigate source refresh"
operator_disposition_df["actor"]="Fictional reviewer exercise"
operator_disposition_df["action_taken"]="No operational action"
dashboard_df=pd.DataFrame([
    {"measure":"Rule-flagged windows","value":int(signals_df.signal_detected.sum())},
    {"measure":"Review episodes","value":total_review_episodes},
    {"measure":"Latest pending queue","value":len(review_df)},
    {"measure":"Prepared notices (not sent)","value":int(outbox_df.status.eq("prepared_not_sent").sum())},
    {"measure":"Repeated notices suppressed","value":int(outbox_df.status.eq("suppressed_repeat").sum())},
])
print(outbox_df.to_string(index=False));print(operator_disposition_df.to_string(index=False))
fig,ax=plt.subplots(figsize=(11,3.5))
bars=ax.barh(dashboard_df.measure.iloc[::-1],dashboard_df.value.iloc[::-1],color="#145A7D")
ax.bar_label(bars,padding=4)
ax.set_xlim(0,float(dashboard_df.value.max())*1.12)
ax.set(title="Local operating dashboard — fictional historical replay",xlabel="Count")
plt.tight_layout();plt.show()
print("Delivery integration is off. Replaying history demonstrates the logic, not a live service.")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 19: Record how each model was made
# MAGIC **Plain English:** Save parameters, scores, and four trained models so another person can inspect this run.
# MAGIC **Technique:** MLflow experiment tracking stores run metadata and artifacts. Model registration in Unity Catalog is optional.
# MAGIC **Output:** A run ID in Databricks; if `UC_MODEL_NAME` is set and access is granted, four registered candidate model versions with demo tags and aliases.
# MAGIC **Developer note:** An MLflow run preserves evidence of this execution. Reproducing it also requires the notebook revision, package versions, input dataset ID, and environment to be recorded.

# COMMAND ----------

# DBTITLE 1,Log the result in MLflow when available
try:
    from inspect import signature
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None

if LOCAL_TEST or mlflow is None:
    print("MLflow logging is disabled for local tests or unavailable in this environment.")
else:
    # Databricks notebooks automatically use their notebook experiment.
    # MLflow 2 uses artifact_path; MLflow 3 uses name for logged models.
    model_parameters = signature(mlflow.sklearn.log_model).parameters
    model_path_arg = "name" if "name" in model_parameters else "artifact_path"
    model_serialization_args = {"serialization_format": "cloudpickle"} if "serialization_format" in model_parameters else {}
    with mlflow.start_run(run_name="attainx_spc_demo_models") as run:
        mlflow.log_params({"source": "fictional", "seed": SEED, "window": WINDOW,
                           "dataset_id": DATASET_ID, "baseline_days": BASELINE_DAYS, "model": "random_forest",
                           "n_estimators": 100, "max_depth": 8})
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(forecast_metrics)
        classifier_model_info = mlflow.sklearn.log_model(
            model, **{model_path_arg: "model"}, input_example=X_train.head(2), **model_serialization_args
        )
        forecast_model_info = mlflow.sklearn.log_model(
            forecast_model, **{model_path_arg: "count_forecast"}, input_example=forecast_X.head(2), **model_serialization_args
        )
        logistic_model_info = mlflow.sklearn.log_model(
            logistic_model, **{model_path_arg: "logistic_classifier"}, input_example=X_train.head(2), **model_serialization_args
        )
        anomaly_model_info = mlflow.sklearn.log_model(
            iso_model, **{model_path_arg: "isolation_forest"}, input_example=X_train.head(2), **model_serialization_args
        )
        mlflow.log_metrics({"five_day_" + row.model.lower().replace("-", "_").replace(" ", "_") + "_mae": float(row.mae)
                            for row in five_day_metrics_df.itertuples()})
        print("MLflow run ID:", run.info.run_id)
        print("Logged model URIs:", classifier_model_info.model_uri, forecast_model_info.model_uri, logistic_model_info.model_uri, anomaly_model_info.model_uri)
        print("Five-day statsmodels forecasts and evaluations are in forecast tables; those fitted models are not logged by this cell.")
        if UC_MODEL_NAME:
            assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", UC_MODEL_NAME), (
                "Set UC_MODEL_NAME to a permitted catalog.schema.model using simple identifiers."
            )
            mlflow.set_registry_uri("databricks-uc")
            client = mlflow.MlflowClient()
            for suffix, info in [("", classifier_model_info), ("_count", forecast_model_info),
                                 ("_logistic", logistic_model_info), ("_anomaly", anomaly_model_info)]:
                name = UC_MODEL_NAME + suffix
                registered = mlflow.register_model(info.model_uri, name)
                client.set_model_version_tag(name, registered.version, "demo_only", "true")
                client.set_registered_model_alias(name, "demo_candidate", registered.version)
                print("Registered demo candidate:", name, "version", registered.version)
            print("No production alias changed. Lifecycle and rollback exercises remain local.")

# Save a portable inference bundle when the companion helper is available. A manually
# imported notebook still runs without that file; MLflow remains its saved-model route.
helper_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (helper_dir / "model_reuse.py").exists():
    helper_dir = Path.cwd() / "notebooks"
saved_forecast_manifest = None
if (helper_dir / "model_reuse.py").exists():
    sys.path.insert(0, str(helper_dir))
    from model_reuse import save_forecast_bundle
    saved_forecast_manifest = save_forecast_bundle(
        forecast_model, forecast_X.loc[forecast_test].head(12),
        os.environ.get("SPC_DEMO_MODEL_DIR", "artifacts/saved_forecast"),
        model_uri=forecast_model_info.model_uri if "forecast_model_info" in globals() else None,
    )
    print("Saved forecast bundle with ordered numeric input contract; use SPC_Model_Scoring.py to score without training.")
else:
    print("Portable bundle helper absent: the separate scoring notebook is NOT ready. Upload model_reuse.py as a Python file alongside this notebook and rerun to create the bundle and manifest. MLflow logging alone does not create that bundle.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 20: Optionally retain queryable demo history
# MAGIC **Plain English:** Save the fictional inputs and results for SQL inspection after confirming a dedicated writable demo schema.
# MAGIC **Method:** Keyed Delta MERGE preserves different forecast origins and replaces an identical replay, instead of deleting all previous runs. The configured destination is the dedicated demonstration schema; explicit local-test mode disables writes.
# MAGIC **Boundary:** A new dedicated demo schema is recommended. This cell does not migrate incompatible old table schemas. Local tests check the merge contract; real Spark/Delta execution needs workspace verification.

# COMMAND ----------
# DBTITLE 1,Optional managed Delta history
DEMO_TABLES = {
    "demo_daily": (daily_df, ["series_id", "run_date"]),
    "spc_signals": (signals_df, ["series_id", "run_date"]),
    "spc_predictions": (predictions_df, ["series_id", "run_date"]),
    "count_forecasts": (forecast_results_df, ["series_id", "run_date"]),
    "review_queue": (review_df, ["series_id", "episode_id"]),
    "monitoring_snapshot": (monitoring_df, []),
    "zone_checks": (zone_df, ["series_id", "run_date"]),
    "subgroup_counts": (subgroup_df, ["series_id", "office", "run_date"]),
    "five_day_forecasts": (future_forecasts_df, ["series_id", "office", "model", "origin_date", "target_date"]),
    "forecast_history": (forecast_ledger_df, ["series_id", "model", "origin_date", "target_date"]),
    "forecast_assessment": (assessment_df, ["series_id", "model", "origin_date"]),
    "model_anomalies": (anomaly_df, ["series_id", "run_date"]),
    "lineage": (lineage_df, ["table", "path"]),
    "upstream_profiles": (upstream_profile_df, ["table"]),
    "operational_events": (events_df, ["event_id"]),
    "investigation_hypotheses": (hypotheses_df, ["series_id", "episode_id"]),
    "model_lifecycle": (lifecycle_df, ["component"]),
    "notification_outbox": (outbox_df, ["run_date", "series_id", "episode_id"]),
    "review_exercise": (operator_disposition_df, ["series_id", "episode_id"]),
}
if OUTPUT_SCHEMA:
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", OUTPUT_SCHEMA), "Use an existing dedicated catalog.schema."
    assert "spark" in globals(), "Delta outputs require Databricks Spark."
    for table_name, (frame, keys) in DEMO_TABLES.items():
        output_frame = frame.copy()
        output_frame["dataset_id"] = DATASET_ID
        keys = ["dataset_id"] + [key for key in keys if key != "dataset_id"]
        assert not output_frame.duplicated(keys).any(), f"Duplicate keys: {table_name}"
        for col in output_frame:
            if pd.api.types.is_datetime64_any_dtype(output_frame[col]):
                output_frame[col] = output_frame[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
        view = "spc_demo_" + table_name
        spark.createDataFrame(output_frame).createOrReplaceTempView(view)
        table = f"{OUTPUT_SCHEMA}.{table_name}"
        spark.sql(f"CREATE TABLE IF NOT EXISTS {table} USING DELTA AS SELECT * FROM {view} WHERE 1=0")
        on = " AND ".join(f"target.`{key}` = source.`{key}`" for key in keys)
        spark.sql(f"MERGE INTO {table} AS target USING {view} AS source ON {on} WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *")
        print("Merged", len(output_frame), "rows into", table)
else:
    print(f"Delta writes are off. {len(DEMO_TABLES)} demo tables are ready in memory; set OUTPUT_SCHEMA only after permission checks.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 21: Detect input changes and deteriorating predictions
# MAGIC **Plain English:** Compare three numeric model inputs with their training distributions. Separately, compare forecast errors once actual counts arrive.
# MAGIC **Method:** Three nonoverlapping 25-business-day holdout windows per series. The first is the performance reference; the next two are monitoring periods. The trained one-day model stays frozen.
# MAGIC **Input drift:** Wasserstein distance divided by training standard deviation summarizes distribution change. Above 0.5 is an illustrative investigation threshold, not a significance test. We monitor `last_value`, `window_std` and `trend`, not every input.
# MAGIC **Performance drift:** Bias is predicted minus actual; positive means overprediction. MAE is average absolute error in count units. Review after two consecutive complete windows exceed reference MAE by 25%. At least 20 matched actuals are required per window. Thresholds need calibration on real history.
# MAGIC **Exercise:** A separately labeled scenario adds 350 counts to outcomes in the two monitoring windows. Inputs and saved predictions stay unchanged. This tests the detector; it is not a new fitted model or a forecast backtest.
# MAGIC **Decision:** Investigate data quality, seasonality and operational changes before retraining. These diagnostics do not establish concept drift or its cause.

# COMMAND ----------
# DBTITLE 1,Drift monitoring with a frozen forecast model
from scipy.stats import wasserstein_distance

DRIFT_WINDOW = 25
DRIFT_MIN_ACTUALS = 20
DRIFT_MAE_RATIO = 1.25
DRIFT_INPUT_THRESHOLD = 0.5
DRIFT_FEATURES = ["last_value", "window_std", "trend"]
DRIFT_SCENARIOS = ["Recorded replay", "Performance drift exercise"]

def input_shift_score(reference, current):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < 20 or len(current) < 20:
        return np.nan
    scale = reference.std(ddof=1)
    # A constant reference has no meaningful standard-deviation unit.
    if scale < 1e-9:
        return 0.0 if np.array_equal(np.unique(reference), np.unique(current)) else np.nan
    return float(wasserstein_distance(reference, current) / scale)

def performance_status(n_actuals, reference_n, reference_mae, current_mae, previous_worse, window_number):
    if min(n_actuals, reference_n) < DRIFT_MIN_ACTUALS or not np.isfinite([reference_mae, current_mae]).all():
        return "Insufficient actuals", False
    if window_number == 0:
        return "Reference window", False
    # A one-count floor avoids unstable percentage changes near zero error.
    worse = current_mae > max(reference_mae, 1.0) * DRIFT_MAE_RATIO
    if worse and previous_worse:
        return "Review for retraining", True
    return ("Watch: one worse window" if worse else "Within demo tolerance"), bool(worse)

RUN_GENERATED_AT = pd.Timestamp.now(tz="UTC").isoformat()
MODEL_IDENTITY = (forecast_model_info.model_uri if "forecast_model_info" in globals() else
                  "local_sha256:" + saved_forecast_manifest["model_sha256"] if saved_forecast_manifest else
                  "in_memory_unregistered_one_day_rf_seed42")
drift_dates = np.sort(forecast_results_df.run_date.unique())
drift_date_windows = [drift_dates[i:i + DRIFT_WINDOW] for i in range(0, len(drift_dates), DRIFT_WINDOW)]
drift_input_rows, drift_performance_rows, drift_daily_parts = [], [], []
for series_id, result_group in forecast_results_df.groupby("series_id"):
    reference_inputs = forecast_df.loc[forecast_train & forecast_df.series_id.eq(series_id)]
    for scenario in DRIFT_SCENARIOS:
        previous_worse = False
        reference_mae = None
        reference_n = 0
        for window_number, window_dates in enumerate(drift_date_windows):
            current = result_group[result_group.run_date.isin(window_dates)].sort_values("run_date").copy()
            current["scenario"] = scenario
            current["window_number"] = window_number
            current["evaluation_actual"] = current.daily_count.astype(float)
            if scenario == "Performance drift exercise" and window_number > 0:
                current["evaluation_actual"] += 350.0
            current["model_error"] = current.predicted_count - current.evaluation_actual
            current["baseline_error"] = current.trailing_mean_baseline - current.evaluation_actual
            valid = np.isfinite(current[["evaluation_actual", "predicted_count", "trailing_mean_baseline"]]).all(axis=1)
            matched = current.loc[valid]
            n_actuals = len(matched)
            mae = float(matched.model_error.abs().mean())
            if window_number == 0:
                reference_mae, reference_n = mae, n_actuals
            status, previous_worse = performance_status(n_actuals, reference_n, reference_mae, mae, previous_worse, window_number)
            drift_performance_rows.append({
                "dataset_id": DATASET_ID, "series_id": series_id, "scenario": scenario,
                "window_number": window_number, "window_start": pd.Timestamp(window_dates[0]),
                "window_end": pd.Timestamp(window_dates[-1]), "n_actuals": n_actuals,
                "expected_actuals": len(window_dates), "coverage": n_actuals / len(window_dates),
                "mae": mae, "reference_mae": reference_mae,
                "review_threshold": max(reference_mae, 1.0) * DRIFT_MAE_RATIO,
                "baseline_mae": float(matched.baseline_error.abs().mean()),
                "bias": float(matched.model_error.mean()), "status": status,
                "model_version": "one_day_rf_seed42_frozen", "model_type": "RandomForestRegressor",
                "model_identity": MODEL_IDENTITY, "run_generated_at": RUN_GENERATED_AT,
                "reference_start": pd.Timestamp(drift_date_windows[0][0]),
                "reference_end": pd.Timestamp(drift_date_windows[0][-1]),
                "training_start": reference_inputs.run_date.min(), "training_end": reference_inputs.run_date.max(),
            })
            drift_daily_parts.append(current)
            current_inputs = forecast_df[forecast_df.series_id.eq(series_id) & forecast_df.run_date.isin(window_dates)]
            for feature in DRIFT_FEATURES:
                score = input_shift_score(reference_inputs[feature], current_inputs[feature])
                drift_input_rows.append({
                    "dataset_id": DATASET_ID, "series_id": series_id, "scenario": scenario,
                    "window_number": window_number, "window_end": pd.Timestamp(window_dates[-1]),
                    "feature": feature, "shift_score": score, "threshold": DRIFT_INPUT_THRESHOLD,
                    "reference_start": reference_inputs.run_date.min(), "reference_end": reference_inputs.run_date.max(),
                    "window_start": pd.Timestamp(window_dates[0]),
                    "reference_n": int(reference_inputs[feature].notna().sum()),
                    "current_n": int(current_inputs[feature].notna().sum()),
                    "missing_rate": float(current_inputs[feature].isna().mean()),
                    "status": "Insufficient variation or data" if not np.isfinite(score) else
                              ("Investigate input change" if score > DRIFT_INPUT_THRESHOLD else "Within demo tolerance"),
                })
drift_inputs_df = pd.DataFrame(drift_input_rows)
drift_performance_df = pd.DataFrame(drift_performance_rows)
drift_daily_df = pd.concat(drift_daily_parts, ignore_index=True)
print(drift_performance_df[["series_id", "scenario", "window_number", "n_actuals", "mae", "reference_mae", "status"]].round(2).to_string(index=False))
print("Model inputs and original forecast scores are unchanged. Drift exercise outcomes are separate evaluation values.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 22: Explain the evidence in a dashboard
# MAGIC **Show:** Start with Recorded replay, choose a series, then switch to Performance drift exercise. Watch error cross the review threshold for two windows; input scores stay the same.
# MAGIC **Read:** Overview connects actual counts, predictions and the simple baseline. Drift separates input change from error deterioration. Decisions explains investigation, retraining, validation and rollback.
# MAGIC **Save:** Setting `OUTPUT_SCHEMA` in Cell 1 writes three additional demo tables below. Import `dashboards/AttainX_SPC_Demo.lvdash.json` in Databricks Dashboards and select your SQL warehouse. Its default catalog/schema must match `OUTPUT_SCHEMA` (or bind the JSON using the supplied script).
# MAGIC **Boundary:** This is a fixed historical replay, not live monitoring. Dashboard refresh reads saved results; rerunning all cells also retrains the demo models. Production monitoring would score with a frozen registered model and join newly arrived actuals separately.

# COMMAND ----------
# DBTITLE 1,Interactive drift dashboard and optional Delta outputs
DRIFT_TABLES = {
    "demo_drift_inputs": (drift_inputs_df, ["scenario", "series_id", "window_number", "feature"]),
    "demo_drift_performance": (drift_performance_df, ["scenario", "series_id", "window_number"]),
    "demo_drift_daily": (drift_daily_df, ["scenario", "series_id", "run_date"]),
}
if OUTPUT_SCHEMA:
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", OUTPUT_SCHEMA)
    assert "spark" in globals(), "Delta outputs require Databricks Spark."
    for table_name, (frame, keys) in DRIFT_TABLES.items():
        output_frame = frame.copy()
        keys = ["dataset_id"] + keys
        assert not output_frame.duplicated(keys).any(), table_name
        for col in output_frame:
            if pd.api.types.is_datetime64_any_dtype(output_frame[col]):
                output_frame[col] = output_frame[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
        view = "spc_demo_" + table_name
        spark.createDataFrame(output_frame).createOrReplaceTempView(view)
        table = f"{OUTPUT_SCHEMA}.{table_name}"
        spark.sql(f"CREATE TABLE IF NOT EXISTS {table} USING DELTA AS SELECT * FROM {view} WHERE 1=0")
        on = " AND ".join(f"target.`{key}` = source.`{key}`" for key in keys)
        spark.sql(f"MERGE INTO {table} AS target USING {view} AS source ON {on} WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *")
        print("Merged", len(output_frame), "rows into", table)
else:
    print("Drift tables are in memory. Set OUTPUT_SCHEMA to save them for the native SQL dashboard.")

import json

def dashboard_records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso"))

dashboard_payload = {
    "dataset": DATASET_ID,
    "dataset_summary": dataset_summary,
    "performance": dashboard_records(drift_performance_df),
    "inputs": dashboard_records(drift_inputs_df),
    "daily": dashboard_records(drift_daily_df),
    "reviews": dashboard_records(review_df),
}
DASHBOARD_HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AttainX · SPC and model monitoring</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:#132f42;font:15px/1.5 system-ui,-apple-system,sans-serif}main{max-width:1240px;margin:auto;padding:28px}h1{font-size:27px;letter-spacing:-.7px;margin:0}h2{font-size:18px;margin:0 0 8px}p{margin:6px 0;color:#526675}.top,.controls,.tabs,.legend{display:flex;align-items:center;gap:18px;flex-wrap:wrap}.top{justify-content:space-between}.tag{background:#e0ecf3;padding:5px 10px;border-radius:6px;font-size:12px}.controls{margin:22px 0 14px}.controls label{font-size:12px;font-weight:650;display:grid;gap:5px}select,button{font:inherit;border:1px solid #bccbd4;border-radius:6px;background:white;color:#163b52;padding:8px 12px}button{cursor:pointer}button:focus-visible,select:focus-visible{outline:3px solid #2687c3;outline-offset:2px}.tabs{border-bottom:1px solid #ced9df;gap:5px;margin:18px 0}.tabs button{border:0;border-radius:6px 6px 0 0;background:transparent}.tabs button[aria-selected=true]{background:#123f5c;color:white}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:16px 0}.card,.panel{background:white;border:1px solid #dee6eb;border-radius:10px;padding:20px}.metric{font-size:29px;font-weight:650;line-height:1.3;margin:7px 0}.label{font-size:12px;color:#526675}.grid{display:grid;grid-template-columns:1.15fr 1fr;gap:16px}.panel{margin-bottom:16px;min-width:0}.notice{border-left:4px solid #227c9d;background:#e5f1f6;padding:12px 16px;margin-bottom:16px}.exercise{background:#fff0d9;border-color:#b67a22}.good{color:#087d71}.warn{color:#9a541e}.badge{font-size:13px;font-weight:650}.chart{width:100%;height:auto;display:block}.legend{font-size:12px;gap:16px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:10px 8px;border-bottom:1px solid #e5ebef}th{font-size:11px;color:#526675;white-space:nowrap}td:last-child{min-width:170px}.scroll{overflow:auto}.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.step{border-top:3px solid #168d81;padding-top:12px}.step strong{display:block}.step p{font-size:13px}.foot{font-size:12px;margin-top:18px}section[hidden]{display:none}svg text{font-family:system-ui;font-size:11px;fill:#526675}.empty{padding:30px;text-align:center}@media(max-width:800px){main{padding:16px}.grid,.cards,.steps{grid-template-columns:1fr}.metric{font-size:25px}h1{font-size:23px}.top{gap:8px}}
</style></head><body><main>
<div class="top"><h1>SPC & model monitoring</h1><span class="tag">AttainX · Synthetic application events</span></div>
<p>Are operations changing, are predictions still useful, and what should we do next?</p><p>Intake A and B count application receipts; Completions A counts workflow completion events. No adjudication or agency records.</p>
<div class="controls"><label>Work queue<select id="series"></select></label><label>Evidence<select id="scenario"><option>Recorded replay</option><option>Performance drift exercise</option></select></label><span class="label" id="period"></span></div>
<div id="notice" class="notice"></div>
<nav class="tabs" aria-label="Dashboard sections" role="tablist"><button id="tab-overview" role="tab" aria-controls="overview" aria-selected="true" data-tab="overview">1 · Overview</button><button id="tab-drift" role="tab" aria-controls="drift" aria-selected="false" data-tab="drift">2 · Detect drift</button><button id="tab-exercise" role="tab" aria-controls="exercise" aria-selected="false" data-tab="exercise">3 · Drift exercise</button><button id="tab-decisions" role="tab" aria-controls="decisions" aria-selected="false" data-tab="decisions">4 · Review and retrain</button></nav>
<section id="overview" role="tabpanel" aria-labelledby="tab-overview"><div class="cards" id="cards"></div><div class="panel"><h2>Actual counts and the forecasts made before them</h2><p>One-day predictions from a frozen random forest. Compare with the trailing five-day mean.</p><div id="daily-chart"></div></div><div class="panel"><h2>How the evidence connects</h2><p id="data-scale"></p><div class="steps"><div class="step"><strong>Daily counts</strong><p>One row per queue and business day.</p></div><div class="step"><strong>Historical features</strong><p>Summaries use earlier dates, excluding the day being forecast.</p></div><div class="step"><strong>Rules & predictions</strong><p>SPC finds process changes. The forecast estimates the next count.</p></div><div class="step"><strong>Observed outcomes</strong><p>Join actuals to saved predictions, then assess error and review.</p></div></div></div></section>
<section id="drift" role="tabpanel" aria-labelledby="tab-drift" hidden><div class="grid"><div class="panel"><h2>Have inputs moved away from training?</h2><p>Latest 25 business days. Distribution distance in training standard-deviation units; larger means more change.</p><div id="input-chart"></div><p class="label">0.5 is a demo investigation threshold. Three numeric features only. Input change alone does not mean the model has failed.</p></div><div class="panel"><h2>Are forecast errors getting worse?</h2><p>Average absolute error (MAE), in counts. Three separate 25-business-day windows.</p><div id="error-chart"></div><p class="label">Review after two consecutive windows above 125% of the reference MAE (one-count floor); at least 20 actuals each. Thresholds are illustrative.</p></div></div><div class="panel"><h2>Error evidence by monitoring window</h2><div id="performance-table" class="scroll"></div></div></section>
<section id="exercise" role="tabpanel" aria-labelledby="tab-exercise" hidden><div class="panel"><h2>A controlled deterioration exercise</h2><p>Select Performance drift exercise above: after the reference window, add 350 counts to each observed outcome. Inputs, frozen predictions and SPC review episodes do not change.</p><p>This deliberately simulated outcome shift tests the monitoring rule. It is not evidence of actual agency conditions, a new model fit, or a real approval.</p><button id="activate-exercise">Show simulated deterioration</button><div id="exercise-table" class="scroll"></div></div></section>
<section id="decisions" role="tabpanel" aria-labelledby="tab-decisions" hidden><div class="panel"><h2 id="decision-title"></h2><p id="decision-detail"></p><div class="steps" style="margin-top:22px"><div class="step"><strong>1. Investigate</strong><p>Check missing or late data, workload mix, seasonality and process changes. Confirm actuals are complete.</p></div><div class="step"><strong>2. Train a candidate</strong><p>Use recent, representative, reviewed history and rerun the same feature pipeline. Preserve the current model.</p></div><div class="step"><strong>3. Validate</strong><p>Compare against the current model and simple baseline on later unseen dates, including queue-level results.</p></div><div class="step"><strong>4. Approve & monitor</strong><p>Version the candidate in MLflow, obtain approval, then promote. Retain the prior version for rollback.</p></div></div></div><div class="panel"><h2>SPC review queue · recorded replay only</h2><p>Pending statistical review candidates are separate from model drift warnings. Exercise outcomes do not alter this queue.</p><div id="review-table" class="scroll"></div></div><div class="notice">No automatic retraining, model promotion or notification occurs here. The demonstration shows evidence and a review decision.</div></section>
<p class="foot" id="source"></p>
</main><script>
const D=__DASHBOARD_DATA__;const $=id=>document.getElementById(id);const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const fmt=v=>Number.isFinite(v)?v.toFixed(2):'Unavailable';const date=v=>String(v).slice(0,10);const colors=['#116d88','#d08029','#188b7e'];
[...new Set(D.performance.map(r=>r.series_id))].sort().forEach(s=>$('series').add(new Option(s,s)));$('series').value='Intake A';
function table(rows,cols){if(!rows.length)return '<p class="empty">No records in this selection.</p>';return '<table><thead><tr>'+cols.map(c=>'<th>'+esc(c[1])+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map(c=>'<td>'+esc(c[2]?c[2](r[c[0]]):r[c[0]])+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}
function line(rows,xfield,fields){const w=650,h=260,l=48,r=18,t=15,b=43;const times=rows.map(r=>Date.parse(r[xfield]));const lo=Math.min(...times),hi=Math.max(...times);const vals=rows.flatMap(r=>fields.map(f=>r[f[0]])).filter(Number.isFinite);const max=Math.max(1,...vals)*1.10;const x=i=>l+(times[i]-lo)/Math.max(1,hi-lo)*(w-l-r);const y=v=>h-b-v/max*(h-t-b);let svg=`<svg class="chart" role="img" aria-label="${fields.map(f=>esc(f[1])).join(', ')} over time" viewBox="0 0 ${w} ${h}">`;for(let i=0;i<=4;i++){let v=max*i/4;svg+=`<line x1="${l}" y1="${y(v)}" x2="${w-r}" y2="${y(v)}" stroke="#e5ebef"/><text x="${l-8}" y="${y(v)+4}" text-anchor="end">${v.toFixed(0)}</text>`}fields.forEach((f,k)=>{const points=rows.map((row,i)=>Number.isFinite(row[f[0]])?`${x(i)},${y(row[f[0]])}`:null);let segment=[];const flush=()=>{if(segment.length)svg+=`<polyline fill="none" stroke="${colors[k]}" stroke-width="2.3" ${k===2?'stroke-dasharray="5 4"':''} points="${segment.join(' ')}"/>`;segment=[]};points.forEach(p=>p?segment.push(p):flush());flush();if(rows.length<10)rows.forEach((row,i)=>{if(Number.isFinite(row[f[0]]))svg+=`<circle cx="${x(i)}" cy="${y(row[f[0]])}" r="4" fill="${colors[k]}"><title>${esc(date(row[xfield]))}: ${esc(f[1])} ${fmt(row[f[0]])}</title></circle>`})});[...new Set([0,Math.floor((rows.length-1)/2),rows.length-1])].forEach(i=>svg+=`<text x="${x(i)}" y="${h-14}" text-anchor="${i===0?'start':i===rows.length-1?'end':'middle'}">${date(rows[i][xfield])}</text>`);return svg+'</svg><div class="legend">'+fields.map((f,k)=>`<span><i class="dot" style="background:${colors[k]}"></i>${esc(f[1])}</span>`).join('')+'</div>'}
function bars(rows){const max=Math.max(1,...rows.map(r=>r.shift_score||0))*1.15;const left=110,right=550;let out='<svg class="chart" role="img" aria-label="Input distribution changes compared with training" viewBox="0 0 650 260">';rows.forEach((r,i)=>{let y=30+i*65;out+=`<text x="0" y="${y+17}">${esc(r.feature)}</text><rect x="${left}" y="${y}" width="${Math.max(0,(r.shift_score||0)/max*(right-left))}" height="26" rx="3" fill="${r.shift_score>.5?'#d08029':'#116d88'}"/><text x="${left+(r.shift_score||0)/max*(right-left)+8}" y="${y+17}">${fmt(r.shift_score)}</text>`});const tx=left+.5/max*(right-left);out+=`<line x1="${tx}" x2="${tx}" y1="18" y2="216" stroke="#8a623c" stroke-dasharray="4 4"/><text x="${tx}" y="237" text-anchor="middle">0.5 threshold</text></svg>`;return out}
function render(){$('data-scale').textContent=D.dataset_summary.event_rows.toLocaleString()+' materialized event rows aggregate to '+D.dataset_summary.daily_rows.toLocaleString()+' daily series observations. Models train on daily windows, not a million independent examples.';const series=$('series').value,scenario=$('scenario').value;const scope=r=>r.series_id===series&&r.scenario===scenario;const perf=D.performance.filter(scope).sort((a,b)=>a.window_number-b.window_number),latest=perf.at(-1),daily=D.daily.filter(scope).sort((a,b)=>a.run_date.localeCompare(b.run_date)),inputs=D.inputs.filter(r=>scope(r)&&r.window_number===latest.window_number);const exercise=scenario!=='Recorded replay';$('period').textContent=date(daily[0].run_date)+' – '+date(daily.at(-1).run_date)+' · '+daily.length+' matched daily records';$('notice').className='notice'+(exercise?' exercise':'');$('notice').textContent=exercise?'Controlled exercise: add 350 counts to outcomes after the reference window. Predictions and inputs stay frozen. These are scenario results.':'Recorded replay: the original synthetic counts and saved predictions. An input shift is a reason to investigate; deteriorating accuracy supplies separate evidence.';$('cards').innerHTML=[['Latest error',fmt(latest.mae)+' counts','Reference '+fmt(latest.reference_mae)+' · simple baseline '+fmt(latest.baseline_mae)],['Matched actuals',latest.n_actuals+' / '+latest.expected_actuals,'Latest window · '+date(latest.window_start)+' – '+date(latest.window_end)],['Review status',latest.status,'Two-window rule · no automatic model change']].map((v,i)=>`<div class="card"><div class="label">${v[0]}</div><div class="metric ${i===2?'badge '+(latest.status==='Within demo tolerance'?'good':'warn'):''}">${esc(v[1])}</div><p class="label">${esc(v[2])}</p></div>`).join('');$('daily-chart').innerHTML=line(daily,'run_date',[['evaluation_actual',exercise?'Exercise outcome':'Actual count'],['predicted_count','Frozen model forecast'],['trailing_mean_baseline','Trailing five-day mean']]);$('error-chart').innerHTML=line(perf,'window_end',[['mae','Model MAE'],['baseline_mae','Simple baseline MAE'],['review_threshold','Review threshold']]);$('input-chart').innerHTML=bars(inputs);$('performance-table').innerHTML=table(perf,[['window_number','Window',v=>v===0?'0 · Reference':v+' · Monitor'],['window_end','Through',date],['n_actuals','Actuals'],['mae','Model MAE',fmt],['baseline_mae','Baseline MAE',fmt],['bias','Bias (forecast − actual)',fmt],['status','Assessment']]);$('exercise-table').innerHTML=table(D.performance.filter(r=>r.series_id===series&&r.scenario==='Performance drift exercise'),[['window_number','Window'],['mae','Simulated MAE',fmt],['reference_mae','Reference MAE',fmt],['status','Assessment']]);$('decision-title').textContent=latest.status;$('decision-detail').textContent=latest.status==='Review for retraining'?'Two consecutive windows exceeded the demo error threshold. Open a review; verify data and causes before training a candidate.':latest.status==='Insufficient actuals'?'Evidence is insufficient or nonfinite. Resolve actuals and coverage before judging model health.':'This series has not met the two-window retraining-review rule. Continue monitoring and investigate any input or data-quality warnings.';$('review-table').innerHTML=table(D.reviews.filter(r=>r.series_id===series),[['episode_id','Episode'],['run_date','Started',date],['last_signal_date','Latest signal',date],['signal_windows','Flagged windows'],['disposition','Disposition']]);$('source').textContent='Model '+latest.model_type+' · '+latest.model_identity+' · Run '+latest.run_generated_at+' · Reference '+date(latest.reference_start)+' to '+date(latest.reference_end)+' · Source: notebook Cells 6, 8 and 21 · '+D.dataset+' · Frozen one-day RF, seed 42 · Historical replay, no live data refresh. Bias above zero means overprediction. Not a concept-drift diagnosis.'}
document.querySelectorAll('[data-tab]').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('[data-tab]').forEach(b=>b.setAttribute('aria-selected',String(b===button)));['overview','drift','exercise','decisions'].forEach(id=>$(id).hidden=id!==button.dataset.tab)}));$('activate-exercise').addEventListener('click',()=>{$('scenario').value='Performance drift exercise';render()});$('series').addEventListener('change',render);$('scenario').addEventListener('change',render);render();
</script></body></html>'''.replace("__DASHBOARD_DATA__", json.dumps(dashboard_payload, allow_nan=False).replace("<", "\\u003c"))
if "displayHTML" in globals():
    displayHTML(DASHBOARD_HTML)
else:
    print("Interactive HTML dashboard prepared in DASHBOARD_HTML; local build script exports the preview.")


# COMMAND ----------
# MAGIC %md
# MAGIC ## Closing talking points
# MAGIC 1. These are fictional counts and fixture metadata, never USCIS performance measurements.
# MAGIC 2. SPC flags and zones identify review candidates. The classifier imitates the historical rules; Isolation Forest offers a separate unusualness signal.
# MAGIC 3. Compare one-day and five-day forecasts with simple baselines. The five-day forecast includes dates beyond the last observed count.
# MAGIC 4. Office shifts, lineage and events support hypotheses. An analyst must verify cause and record a real decision.
# MAGIC 5. Historical assessment separates insufficient evidence, stale runs and possible retraining needs. Local lifecycle exercises demonstrate withdrawal.
# MAGIC 6. Replay notices are prepared or suppressed, never sent. A live schedule, real registry/Delta writes, external delivery and agency authorization remain separate checks.
