-- Replace ml_statistical_process_controls.demo_schema with the dedicated, verified demo schema.
-- Read-only examples. Dataset filters keep revisions separate; dates are ISO strings.

-- Summary scorecards.
SELECT source_rows, missing_counts, rule_signal_rate, raw_signal_windows, review_episodes,
       classifier_accuracy, always_signal_accuracy, forecast_mae, trailing_mean_baseline_mae
FROM ml_statistical_process_controls.demo_schema.monitoring_snapshot
WHERE dataset_id = 'attainx_applications_v3_seed42';

-- Control-chart series.
SELECT run_date, series_id, last_value, ucl, lcl, xmr_signal, cusum_signal, ewma_signal, signal_detected
FROM ml_statistical_process_controls.demo_schema.spc_signals
WHERE dataset_id = 'attainx_applications_v3_seed42' AND series_id = 'Intake A'
ORDER BY run_date;

-- Pending reviews, separate from the synthetic review exercise.
SELECT series_id, run_date, last_signal_date, signal_windows, rule_fired, disposition
FROM ml_statistical_process_controls.demo_schema.review_queue
WHERE dataset_id = 'attainx_applications_v3_seed42'
ORDER BY last_signal_date DESC;

-- One-day holdout predictions.
SELECT run_date, series_id, daily_count, predicted_count, trailing_mean_baseline
FROM ml_statistical_process_controls.demo_schema.count_forecasts
WHERE dataset_id = 'attainx_applications_v3_seed42' AND series_id = 'Intake A'
ORDER BY run_date;

-- Future total and office forecasts; keep origin, method and office as filters.
SELECT series_id, office, model, origin_date, target_date, horizon, prediction
FROM ml_statistical_process_controls.demo_schema.five_day_forecasts
WHERE dataset_id = 'attainx_applications_v3_seed42'
ORDER BY series_id, office, model, origin_date, target_date;

-- Forecast-error history across origins.
SELECT series_id, model, origin_date, n_actuals, mae, rmse, smape, bias, direction_accuracy, recommendation
FROM ml_statistical_process_controls.demo_schema.forecast_assessment
WHERE dataset_id = 'attainx_applications_v3_seed42'
ORDER BY series_id, model, origin_date;

-- Investigation evidence. These are hypotheses, not verified causes.
SELECT series_id, episode_id, top_office, largest_abs_z, event_ids, hypothesis, confidence
FROM ml_statistical_process_controls.demo_schema.investigation_hypotheses
WHERE dataset_id = 'attainx_applications_v3_seed42';

-- Prepared and suppressed notices; no delivery connector is enabled.
SELECT run_date, status, COUNT(*) AS notice_count
FROM ml_statistical_process_controls.demo_schema.notification_outbox
WHERE dataset_id = 'attainx_applications_v3_seed42'
GROUP BY run_date, status ORDER BY run_date, status;

-- Candidate recommendations are advisory.
SELECT component, version, recommendation, reason
FROM ml_statistical_process_controls.demo_schema.model_lifecycle
WHERE dataset_id = 'attainx_applications_v3_seed42';


-- Latest performance and freshness: historical data dates differ from execution time.
SELECT series_id, scenario, window_start, window_end, n_actuals, expected_actuals,
       mae, reference_mae, baseline_mae, review_threshold, status,
       model_type, model_identity, run_generated_at, reference_start, reference_end
FROM ml_statistical_process_controls.demo_schema.demo_drift_performance
WHERE dataset_id = 'attainx_applications_v3_seed42' AND window_number = 2
ORDER BY scenario, series_id;

-- Input change is separate evidence; insufficient variation/data is not a healthy score.
SELECT series_id, feature, reference_start, reference_end, window_start, window_end,
       reference_n, current_n, missing_rate, shift_score, threshold, status
FROM ml_statistical_process_controls.demo_schema.demo_drift_inputs
WHERE dataset_id = 'attainx_applications_v3_seed42' AND scenario = 'Recorded replay' AND window_number = 2
ORDER BY series_id, feature;
