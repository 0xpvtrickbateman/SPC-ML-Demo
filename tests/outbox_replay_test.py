"""Run the actual daily replay with quiet days and repeated episode notices."""
import contextlib
import io
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import pandas as pd

root = Path(__file__).resolve().parents[1]
source = (root / "notebooks/SPC_ML_Demo.py").read_text()
cell = source.split("# DBTITLE 1,Daily replay notification outbox and human review\n", 1)[1].split("# COMMAND ----------", 1)[0]
dates = pd.bdate_range("2026-03-02", periods=6)
columns = ["run_date", "series_id", "episode_id"]


def replay(episodes):
    namespace = dict(
        pd=pd, plt=plt, DATASET_ID="quiet-replay",
        signals_df=pd.DataFrame({"run_date": dates, "signal_detected": False}),
        episode_rows=episodes,
        review_df=pd.DataFrame({"series_id": ["Intake A"], "episode_id": [1]}),
        total_review_episodes=1,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(cell, str(root / "notebooks/SPC_ML_Demo.py"), "exec"), namespace)
    plt.close("all")
    return namespace


for episodes in [pd.DataFrame(columns=columns), pd.DataFrame([[dates[0], "Intake A", 1]], columns=columns)]:
    state = replay(episodes)
    outbox = state["outbox_df"]
    assert outbox.empty
    assert list(outbox) == ["dataset_id", "run_date", "series_id", "episode_id", "status", "delivery"]
    assert str(outbox.run_date.dtype) == "datetime64[ns]" and str(outbox.episode_id.dtype) == "int64"
    counts = state["dashboard_df"].set_index("measure").value
    assert counts["Prepared notices (not sent)"] == counts["Repeated notices suppressed"] == 0

episodes = pd.DataFrame([[day, "Intake A", 1] for day in dates[-3:]], columns=columns)
first, second = replay(episodes)["outbox_df"], replay(episodes)["outbox_df"]
assert list(first.status) == ["prepared_not_sent", "suppressed_repeat", "suppressed_repeat"]
pd.testing.assert_frame_equal(first, second)
print("PASS: quiet replay keeps the outbox schema and zero notice counts; repeated episodes remain deterministic and unsent.")
