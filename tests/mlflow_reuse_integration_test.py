"""Optional REAL MLflow save/load test in a fresh process (requires mlflow-skinny).

Run with a separate environment when MLflow is not part of the local demo lock.
This tests an MLflow artifact on local disk, not workspace tracking or UC access.
"""
import importlib.util
import contextlib
import io
import runpy
import os
from pathlib import Path
import subprocess
import sys
import tempfile

if importlib.util.find_spec("mlflow") is None:
    raise SystemExit("MLflow missing: install mlflow-skinny in an isolated environment to run this integration test")

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"  # Isolated disposable test store only.
import mlflow.sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

with tempfile.TemporaryDirectory(prefix="spc-real-mlflow-") as tmp:
    mlflow.set_tracking_uri((Path(tmp) / "tracking").as_uri())
    mlflow.set_experiment("synthetic-model-reuse")
    bundle_path = Path(tmp) / "bundle"
    os.environ["SPC_DEMO_MODEL_DIR"] = str(bundle_path)
    os.environ["SPC_DEMO_LOCAL_TEST"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    notebook = ROOT / "notebooks" / "SPC_ML_Demo.py"
    with contextlib.redirect_stdout(io.StringIO()):
        state = runpy.run_path(str(notebook))
    # Execute the actual production logging cell with real MLflow, not a stand-in.
    cell = notebook.read_text().split(
        "# DBTITLE 1,Log the result in MLflow when available\n", 1
    )[1].split("# COMMAND ----------", 1)[0]
    state["LOCAL_TEST"] = False
    state["UC_MODEL_NAME"] = ""
    with contextlib.redirect_stdout(io.StringIO()):
        exec(cell, state)
    assert state["MLFLOW_STATUS"] == "logged", state["MLFLOW_STATUS"]
    assert all(state[name] is not None for name in (
        "classifier_model_info", "forecast_model_info", "logistic_model_info", "anomaly_model_info",
    ))
    assert state["saved_forecast_manifest"]["model_uri"] == state["forecast_model_info"].model_uri
    child = '''
import runpy
from unittest.mock import patch
from sklearn.ensemble import RandomForestRegressor
with patch.object(RandomForestRegressor, "fit", side_effect=AssertionError("Scoring called fit")):
    state = runpy.run_path("notebooks/SPC_Model_Scoring.py")
assert state["artifact_identity"] == state["manifest"]["model_uri"]
'''
    env = {**os.environ, "SPC_DEMO_MODEL_DIR": str(bundle_path), "SPC_DEMO_SCORE_BACKEND": "mlflow", "MLFLOW_TRACKING_URI": mlflow.get_tracking_uri()}
    result = subprocess.run([sys.executable, "-c", child], cwd=ROOT, env=env,
                            capture_output=True, text=True, check=True)
    assert "Saved-model inference passed" in result.stdout
print("Real MLflow sklearn artifact logged and loaded in a fresh process; predictions equal and fit forbidden. Workspace and registry not tested.")
