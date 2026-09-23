-- Replace demo_catalog.demo_schema with your actual writable Unity Catalog location.
-- These are read-only examples; run after OUTPUT_SCHEMA was set and the notebook succeeded.

-- Summary cards: data and model health for the latest synthetic run.
SELECT dataset_id, source_rows, missing_counts, rule_signal_rate,
       raw_signal_windows, review_episodes,
       classifier_accuracy, majority_baseline_accuracy, always_signal_accuracy,
       forecast_mae, trailing_mean_baseline_mae
FROM demo_catalog.demo_schema.monitoring_snapshot;

-- Time series for a control-chart visualization; filter to one series in the dashboard.
SELECT run_date, series_id, last_value, ucl, lcl,
       xmr_signal, cusum_signal, ewma_signal, signal_detected
FROM demo_catalog.demo_schema.spc_signals
WHERE series_id = 'Intake A'
ORDER BY run_date;

-- Human review candidates. This is a synthetic queue, not a case-management system.
SELECT series_id, run_date, last_signal_date, signal_windows,
       rule_fired, last_value, disposition, dataset_id
FROM demo_catalog.demo_schema.review_queue
ORDER BY last_signal_date DESC, series_id;

-- Forward-looking forecast: observed versus predicted on held-out dates.
SELECT run_date, series_id, daily_count, predicted_count, trailing_mean_baseline
FROM demo_catalog.demo_schema.count_forecasts
WHERE series_id = 'Intake A'
ORDER BY run_date;
