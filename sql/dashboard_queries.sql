-- Replace demo_catalog.demo_schema with the dedicated, verified demo schema.
-- Read-only examples. Dataset filters keep revisions separate; dates are ISO strings.

-- Summary scorecards.
SELECT source_rows, missing_counts, rule_signal_rate, raw_signal_windows, review_episodes,
       classifier_accuracy, always_signal_accuracy, forecast_mae, trailing_mean_baseline_mae
FROM demo_catalog.demo_schema.monitoring_snapshot
WHERE dataset_id = 'attainx_demo_v2_seed42';

-- Control-chart series.
SELECT run_date, series_id, last_value, ucl, lcl, xmr_signal, cusum_signal, ewma_signal, signal_detected
FROM demo_catalog.demo_schema.spc_signals
WHERE dataset_id = 'attainx_demo_v2_seed42' AND series_id = 'Intake A'
ORDER BY run_date;

-- Pending reviews, separate from the fictional review exercise.
SELECT series_id, run_date, last_signal_date, signal_windows, rule_fired, disposition
FROM demo_catalog.demo_schema.review_queue
WHERE dataset_id = 'attainx_demo_v2_seed42'
ORDER BY last_signal_date DESC;

-- One-day holdout predictions.
SELECT run_date, series_id, daily_count, predicted_count, trailing_mean_baseline
FROM demo_catalog.demo_schema.count_forecasts
WHERE dataset_id = 'attainx_demo_v2_seed42' AND series_id = 'Intake A'
ORDER BY run_date;

-- Future total and office forecasts; keep origin, method and office as filters.
SELECT series_id, office, model, origin_date, target_date, horizon, prediction
FROM demo_catalog.demo_schema.five_day_forecasts
WHERE dataset_id = 'attainx_demo_v2_seed42'
ORDER BY series_id, office, model, origin_date, target_date;

-- Forecast-error history across origins.
SELECT series_id, model, origin_date, n_actuals, mae, rmse, smape, bias, direction_accuracy, recommendation
FROM demo_catalog.demo_schema.forecast_assessment
WHERE dataset_id = 'attainx_demo_v2_seed42'
ORDER BY series_id, model, origin_date;

-- Investigation evidence. These are hypotheses, not verified causes.
SELECT series_id, episode_id, top_office, largest_abs_z, event_ids, hypothesis, confidence
FROM demo_catalog.demo_schema.investigation_hypotheses
WHERE dataset_id = 'attainx_demo_v2_seed42';

-- Prepared and suppressed notices; no delivery connector is enabled.
SELECT run_date, status, COUNT(*) AS notice_count
FROM demo_catalog.demo_schema.notification_outbox
WHERE dataset_id = 'attainx_demo_v2_seed42'
GROUP BY run_date, status ORDER BY run_date, status;

-- Candidate recommendations are advisory.
SELECT component, version, recommendation, reason
FROM demo_catalog.demo_schema.model_lifecycle
WHERE dataset_id = 'attainx_demo_v2_seed42';
