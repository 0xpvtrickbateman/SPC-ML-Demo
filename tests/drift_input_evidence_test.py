"""Exercise the notebook's actual drift-row construction with nonfinite values and absent business days."""
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "notebooks/SPC_ML_Demo.py").read_text()
cell = source.split("# DBTITLE 1,Drift monitoring with a frozen forecast model\n")[1].split("# COMMAND ----------")[0]


def run_cell(inputs, results, expected_dates, training_end):
    ns = dict(np=np, pd=pd, DATASET_ID="finite-regression", forecast_df=inputs,
              forecast_train=inputs.run_date.le(training_end), forecast_results_df=results,
              forecast_test_dates=expected_dates, saved_forecast_manifest=None)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(cell, str(ROOT / "notebooks/SPC_ML_Demo.py"), "exec"), ns)
    return ns


def evidence(reference, current):
    """Current inputs occupy the first len(current) of 25 calendar dates; later dates are absent."""
    training_dates = pd.date_range("2026-01-01", periods=len(reference))
    current_dates = pd.date_range("2026-03-01", periods=25)
    values = np.concatenate([reference, current])
    inputs = pd.DataFrame({
        "series_id": "Test queue",
        "run_date": training_dates.append(current_dates[:len(current)]),
        **{feature: values for feature in ["last_value", "window_std", "trend"]},
    })
    results = pd.DataFrame({
        "series_id": "Test queue", "run_date": current_dates,
        "daily_count": 10., "predicted_count": 11., "trailing_mean_baseline": 12.,
    })
    rows = run_cell(inputs, results, current_dates, pd.Timestamp("2026-02-28"))["drift_inputs_df"]
    assert len(rows) == 6  # All three features in both scenarios use the same path.
    return rows


clean = np.arange(25, dtype=float)
cases = [
    ("clean", clean, clean, 25, 25, 0., True),
    ("six infinities", clean, np.r_[np.arange(19.), [np.inf, -np.inf] * 3], 25, 19, 6/25, False),
    ("mixed current 19", clean, np.r_[np.arange(19.), [np.nan, np.inf, -np.inf] * 2], 25, 19, 6/25, False),
    ("mixed current 20", clean, np.r_[np.arange(20.), np.nan, np.inf, -np.inf, np.nan, np.inf], 25, 20, 5/25, True),
    ("reference 19", np.r_[np.arange(19.), [np.nan, np.inf, -np.inf] * 2], clean, 19, 25, 0., False),
    ("both 20", np.r_[np.arange(20.), np.nan, np.inf, -np.inf, np.nan, np.inf], np.r_[np.arange(20.), -np.inf, np.nan, np.inf, -np.inf, np.nan], 20, 20, 5/25, True),
    ("all invalid", clean, np.resize([np.nan, np.inf, -np.inf], 25), 25, 0, 1., False),
    # Absent business days count as missing against the configured calendar, not just invalid values.
    ("five absent days", clean, np.arange(20.), 25, 20, 5/25, True),
    ("six absent days", clean, np.arange(19.), 25, 19, 6/25, False),
    ("empty current", clean, np.array([]), 25, 0, 1., False),
    ("empty reference", np.array([]), clean, 0, 25, 0., False),
]
for name, reference, current, reference_n, current_n, rate, eligible in cases:
    rows = evidence(reference, current)
    assert rows.reference_n.eq(reference_n).all(), name
    assert rows.current_n.eq(current_n).all(), name
    np.testing.assert_allclose(rows.missing_rate, rate, err_msg=name)
    if eligible:
        assert np.isfinite(rows.shift_score).all(), name
        assert not rows.status.eq("Insufficient variation or data").any(), name
    else:
        assert rows.shift_score.isna().all(), name
        assert rows.status.eq("Insufficient variation or data").all(), name
    if name in ("clean", "both 20"):
        assert rows.shift_score.eq(0.).all(), name

# Invalid values must not alter a score relative to the same finite observations.
contaminated = evidence(cases[3][1], cases[3][2])
finite_only = evidence(clean, np.arange(20.))
np.testing.assert_allclose(contaminated.shift_score, finite_only.shift_score)


def performance(absent):
    """75 calendar dates in three 25-day windows; outcome rows for `absent` dates never arrive."""
    calendar = pd.bdate_range("2026-03-02", periods=75)
    training = pd.date_range("2026-01-01", periods=25)
    inputs = pd.DataFrame({"series_id": "Test queue", "run_date": training.append(calendar),
                           **{feature: np.arange(100.) for feature in ["last_value", "window_std", "trend"]}})
    present = calendar.delete(absent)
    results = pd.DataFrame({"series_id": "Test queue", "run_date": present, "daily_count": 10.,
                            "predicted_count": 11., "trailing_mean_baseline": 12.})
    ns = run_cell(inputs, results, calendar, training[-1])
    frame = ns["drift_performance_df"]
    return frame[frame.scenario.eq("Recorded replay")].set_index("window_number"), ns["drift_daily_df"], calendar


# A five-day feed gap inside window 1 lowers its coverage; windows stay aligned to the calendar.
perf, daily, calendar = performance(list(range(30, 35)))
assert perf.expected_actuals.eq(25).all() and perf.n_actuals.tolist() == [25, 20, 25]
np.testing.assert_allclose(perf.coverage, [1., .8, 1.])
assert [(row.window_start, row.window_end) for row in perf.itertuples()] == [(calendar[i], calendar[i + 24]) for i in (0, 25, 50)]
assert perf.loc[1, "status"] != "Insufficient actuals"
gap = daily[daily.scenario.eq("Recorded replay") & daily.run_date.isin(calendar[30:35])]
assert len(gap) == 5 and gap.evaluation_actual.isna().all()  # Absent days stay visible as gaps.

# Six missing days leave 19 matched actuals: insufficient evidence, with 76% coverage rather than 19/19.
perf, _, _ = performance(list(range(56, 62)))
assert perf.loc[2, ["n_actuals", "expected_actuals"]].tolist() == [19, 25]
assert np.isclose(perf.loc[2, "coverage"], .76) and perf.loc[2, "status"] == "Insufficient actuals"
print(f"PASS: drift evidence for {len(cases)} input cases, finite-only score equivalence and calendar-aligned outcome coverage.")
