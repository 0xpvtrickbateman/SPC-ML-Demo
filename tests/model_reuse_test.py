"""Real serialization and fresh-interpreter inference; no mocked model persistence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from model_reuse import load_forecast_bundle, prepare_inputs, save_forecast_bundle, score_inputs

raw = pd.DataFrame({
    "series_id": ["queue_a", "queue_b"] * 20,
    "weekday": np.arange(40, dtype=np.int32) % 5,
    "last_value": np.linspace(100, 150, 40),
    "last_5_mean": np.linspace(101, 151, 40),
    "window_mean": np.linspace(99, 149, 40),
    "window_std": np.full(40, 5.0),
    "trend": np.linspace(-1, 1, 40),
    "baseline_mean": np.full(40, 110.0),
})
inputs = pd.get_dummies(raw, columns=["series_id"], dtype=float)
model = RandomForestRegressor(n_estimators=8, random_state=42).fit(inputs, np.arange(40) + 110)


def rejected(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError("Malformed input was accepted")


with tempfile.TemporaryDirectory(prefix="spc-model-reuse-") as tmp:
    manifest = save_forecast_bundle(model, inputs.iloc[:12], tmp)
    loaded, saved, identity = load_forecast_bundle(tmp)
    assert identity == "sha256:" + manifest["model_sha256"]
    expected = model.predict(inputs.iloc[:12])
    np.testing.assert_array_equal(score_inputs(loaded, inputs.iloc[:12], saved), expected)
    np.testing.assert_array_equal(score_inputs(loaded, inputs.iloc[:12, ::-1], saved), expected)
    pd.testing.assert_frame_equal(prepare_inputs(raw.iloc[:12], saved), inputs.iloc[:12])
    # A batch containing one queue must still retain both training dummy columns.
    pd.testing.assert_frame_equal(prepare_inputs(raw.iloc[[0]], saved), inputs.iloc[[0]])
    rejected(lambda: prepare_inputs(raw.iloc[:1].assign(series_id="new_queue"), saved))
    rejected(lambda: score_inputs(loaded, inputs.drop(columns="weekday"), saved))
    rejected(lambda: score_inputs(loaded, inputs.assign(extra=1.0), saved))
    rejected(lambda: score_inputs(loaded, inputs.assign(weekday=inputs.weekday.astype(float)), saved))
    rejected(lambda: score_inputs(loaded, inputs.assign(last_value="bad"), saved))
    rejected(lambda: score_inputs(loaded, inputs.assign(last_value=np.nan), saved))
    rejected(lambda: score_inputs(loaded, inputs.assign(last_value=np.inf), saved))
    rejected(lambda: score_inputs(loaded, pd.concat([inputs, inputs[["weekday"]]], axis=1), saved))
    # No training object is inherited by the child. Fail if scoring attempts fit.
    child = '''
import runpy
from unittest.mock import patch
from sklearn.ensemble import RandomForestRegressor
with patch.object(RandomForestRegressor, "fit", side_effect=AssertionError("Scoring called fit")):
    runpy.run_path("notebooks/SPC_Model_Scoring.py")
'''
    env = {**os.environ, "SPC_DEMO_MODEL_DIR": tmp, "SPC_DEMO_SCORE_BACKEND": "local"}
    result = subprocess.run([sys.executable, "-c", child], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert "Saved-model inference passed" in result.stdout
    artifact = Path(tmp) / "forecast.joblib"
    artifact.write_bytes(artifact.read_bytes() + b"corruption")
    rejected(lambda: load_forecast_bundle(tmp))

print("Real joblib persistence, fresh-process no-fit scoring, feature order/types, preprocessing and malformed-schema checks passed.")
