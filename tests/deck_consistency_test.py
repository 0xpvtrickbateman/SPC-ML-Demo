"""Check the delivered deck against the notebook settings, the saved evidence and a fresh execution.

Uses the standard library for the deck; the fresh execution runs the notebook in local-test mode.
"""
import contextlib
import io
import json
import math
import os
import re
import runpy
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
deck = zipfile.ZipFile(ROOT / "docs/SPC_ML_Demo_Walkthrough_Ready.pptx")
evidence = json.loads((ROOT / "docs/deck-evidence.json").read_text())
notebook = (ROOT / "notebooks/SPC_ML_Demo.py").read_text()
talking_points = (ROOT / "docs/talking-points.md").read_text()


def part(target, base="ppt/"):
    return target.lstrip("/") if target.startswith("/") else base + target.replace("../", "")


def text(element):
    return ["".join(t.text or "" for t in p.iter(f"{{{NS['a']}}}t")) for p in element.iter(f"{{{NS['a']}}}p")]


presentation = ET.fromstring(deck.read("ppt/presentation.xml"))
rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(deck.read("ppt/_rels/presentation.xml.rels"))}
slides, notes, tables = [], [], []
for slide_id in presentation.find("p:sldIdLst", NS):
    target = part(rels[slide_id.get(f"{{{NS['r']}}}id")])
    xml = ET.fromstring(deck.read(target))
    slides.append(" ".join(text(xml)))
    tables.append([[" ".join(text(cell)) for cell in row.findall("a:tc", NS)] for row in xml.iter(f"{{{NS['a']}}}tr")])
    slide_rels = ET.fromstring(deck.read(f"ppt/slides/_rels/{Path(target).name}.rels"))
    note = [part(r.get("Target")) for r in slide_rels if "notesSlide" in r.get("Target")]
    bodies = [shape for shape in ET.fromstring(deck.read(note[0])).iter(f"{{{NS['p']}}}sp")
              if (ph := shape.find(".//p:ph", NS)) is not None and ph.get("type") == "body"] if note else []
    notes.append(" ".join(" ".join(text(body)) for body in bodies))
assert all(notes), "Every slide needs speaker notes"

# Appendix B must state the configuration the notebook actually uses for the holdout and fold classifiers.
config = dict(re.findall(r"(\w+)=([\w\"]+)", re.search(r"CLASSIFIER_CONFIG = dict\((.*?)\)", notebook, re.S).group(1)))
expected = f"{config['n_estimators']} trees, depth {config['max_depth']}, leaf minimum {config['min_samples_leaf']}"
expected += ", balanced classes" if config.get("class_weight") == '"balanced"' else ""
assert "RandomForestClassifier(**CLASSIFIER_CONFIG)" in notebook.split("### Cell 11:")[1].split("### Cell 12:")[0]
appendix_b = next(t for s, t in zip(slides, tables) if "Appendix B" in s)
assert ["RF classifier (holdout and folds)", expected] in appendix_b, appendix_b
assert not any("60 trees" in s or "use different settings" in n for s, n in zip(slides, notes))

# The classifier slide shows the executed table, including an always-signal constant on the F1 metric.
classification = next(t for s, t in zip(slides, tables) if "The comparison determines the conclusion" in s)
assert classification == evidence["classificationTable"], classification
assert any("F1" in row[0] and "always signal" in row[1].lower() for row in classification[1:])
assert evidence["classificationConclusion"] in " ".join(slides)

# Wording that overstated bitwise-identical reuse or scoring of new data stays out.
assert not any("identical predictions" in s or "score later data" in s for s in slides)

# Talking points carry the same speaker notes as the delivered deck.
for note in notes:
    assert note.split(" Source: notebooks/")[0].strip() in talking_points, note[:80]

# The saved evidence must come from this notebook revision: check its pin, then rebuild it from a fresh execution.
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_deck_evidence import classification_conclusion, evidence_from_receipt, f1_outcome, receipt_from_state, source_identity

assert evidence["sourceIdentity"]["notebook_sha256"] == source_identity(evidence["dataset"])["notebook_sha256"], (
    "docs/deck-evidence.json was built from another notebook revision; rerun the smoke test and prepare_deck_evidence.py")
os.environ["SPC_DEMO_LOCAL_TEST"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["SPC_DEMO_MODEL_DIR"] = tempfile.mkdtemp(prefix="spc-deck-bundle-")
with contextlib.redirect_stdout(io.StringIO()):
    fresh = evidence_from_receipt(receipt_from_state(runpy.run_path(str(ROOT / "notebooks/SPC_ML_Demo.py"))))


def same(saved, current, path="evidence"):
    """Exact match except floats, which parallel tree prediction can perturb near machine precision."""
    if isinstance(saved, dict):
        assert set(saved) == set(current), (path, set(saved) ^ set(current))
        for key in saved:
            same(saved[key], current[key], f"{path}.{key}")
    elif isinstance(saved, list):
        assert len(saved) == len(current), path
        for index, (a, b) in enumerate(zip(saved, current)):
            same(a, b, f"{path}[{index}]")
    elif isinstance(saved, float) and not isinstance(saved, bool):
        assert math.isclose(saved, current, rel_tol=1e-9, abs_tol=1e-9), (path, saved, current)
    else:
        assert saved == current, (path, saved, current)


same(evidence, fresh)

# Wins, ties and losses against always-signal each produce an accurate conclusion.
assert (f1_outcome(0.9, 0.877), f1_outcome(0.8774, 0.8768), f1_outcome(0.8, 0.877)) == ("win", "tie", "loss")
assert classification_conclusion({"loss"}).startswith("Neither classifier beats")
assert "ties" in classification_conclusion({"tie", "loss"}) and "beats always-signal on F1 in at least one" in classification_conclusion({"win", "tie"})
print(f"PASS: {len(slides)} slides match notebook settings, saved evidence, a fresh execution and the talking points.")
