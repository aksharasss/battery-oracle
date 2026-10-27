"""predict() contract on the real Severson samples (needs the fitted model)."""
import io
from pathlib import Path

import pytest

from oracle.grading import MODEL_DIR
from oracle.predict import GradeResult, predict

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
pytestmark = pytest.mark.skipif(not (MODEL_DIR / "grading_severson.joblib").exists(),
                                reason="run scripts/fit_grading.py first")


def test_fresh_cell_is_grade_a():
    r = predict(SAMPLES / "severson_c0_cycle11.csv")
    assert isinstance(r, GradeResult)
    assert r.grade == "A" and r.dominant == "none"
    assert set(r.mechanisms) == {"LLI", "LAM_PE", "LAM_NE"}
    assert 0 <= r.confidence <= 1 and r.reference is not None


def test_aged_cell_grades_worse_than_fresh():
    fresh = predict(SAMPLES / "severson_c0_cycle11.csv")
    aged = predict(SAMPLES / "severson_c0_cycle311.csv")
    assert aged.grade == "C"
    assert aged.features["severity"] > fresh.features["severity"]
    assert not any(n.startswith("Out of range") for n in aged.notes)


def test_accepts_file_like():
    buf = io.StringIO((SAMPLES / "severson_c0_cycle201.csv").read_text())
    assert predict(buf).grade in "ABC"


def test_bad_input_raises_readable_valueerror():
    with pytest.raises(ValueError, match="voltage"):
        predict(io.StringIO("a,b\n1,2\n3,4\n"))
    with pytest.raises(ValueError, match="window"):
        predict(io.StringIO("voltage_V,capacity_Ah\n" + "\n".join(f"{3.6 - i * 1e-3},{i * 1e-3}" for i in range(50))))


def test_slow_rate_upload_uses_twin_path():
    fresh = predict(SAMPLES / "wheeler_6a_cycle0_C25.csv")
    aged = predict(SAMPLES / "wheeler_6a_cycle3300_C25.csv")
    assert fresh.features["path"] == aged.features["path"] == "slow"
    assert fresh.grade == "A"
    assert aged.grade == "C" and aged.mechanisms["LLI"] > 0.2


def test_fast_upload_stays_on_4c_path():
    assert predict(SAMPLES / "severson_c0_cycle11.csv").features["path"] == "fast"
