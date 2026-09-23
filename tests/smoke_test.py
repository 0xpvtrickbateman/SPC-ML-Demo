"""Execute the complete notebook as Python and check temporal/decision invariants."""
import contextlib
import io
import os
from pathlib import Path
import runpy

os.environ.setdefault("MPLBACKEND", "Agg")
root = Path(__file__).resolve().parents[1]
with contextlib.redirect_stdout(io.StringIO()) as output:
    state = runpy.run_path(str(root / "notebooks" / "SPC_ML_Demo.py"))

signals = state["signals_df"]
train = state["train_df"]
test = state["test_df"]
forecast = state["forecast_results_df"]
assert len(signals) == 915
assert signals["signal_detected"].eq(signals[["xmr_signal", "cusum_signal", "ewma_signal"]].any(axis=1)).all()
assert train["window_end"].max() < test["window_start"].min()
assert state["forecast_df"].loc[state["forecast_train"], "run_date"].max() < state["forecast_df"].loc[state["forecast_test"], "window_start"].min()
assert forecast["predicted_count"].ge(0).all()
assert state["review_df"]["disposition"].eq("Pending analyst review").all()
assert state["forecast_metrics"]["forecast_mae"] < state["forecast_metrics"]["trailing_mean_baseline_mae"]
assert state["metrics"]["accuracy"] < state["metrics"]["always_signal_accuracy"]
print("Smoke test passed; notebook outputs follow:")
print(output.getvalue())
