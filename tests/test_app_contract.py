"""The UI's side of the predict() contract, and a headless run of the page."""
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))


def _predict():
    """Same choice the app makes: the fitted pipeline if available, else the stub."""
    try:
        from oracle.predict import load_model, predict
        load_model()
        return predict
    except (ImportError, FileNotFoundError):
        from stub_predict import predict
        return predict


@pytest.mark.parametrize("name", ["mild_lli", "lam_pe", "severe"])
def test_predict_returns_contract_fields(name):
    res = _predict()(APP / "samples" / f"{name}.csv")
    assert res.grade in {"A", "B", "C"}
    assert 0.0 <= res.confidence <= 1.0
    assert set(res.mechanisms) == {"LLI", "LAM_PE", "LAM_NE"}
    assert all(0.0 <= v <= 1.0 for v in res.mechanisms.values())
    assert res.dominant in {*res.mechanisms, "none"}
    assert res.ica.voltage.shape == res.ica.dqdv.shape
    assert isinstance(res.second_life_use, str) and res.second_life_use
    assert isinstance(res.notes, list)


SEVERSON = sorted((APP.parent / "data" / "samples").glob("*.csv"))


@pytest.mark.skipif(not SEVERSON, reason="no pipeline samples yet")
@pytest.mark.parametrize("path", SEVERSON, ids=lambda p: p.stem)
def test_pipeline_samples_grade(path):
    res = _predict()(path)
    assert res.grade in {"A", "B", "C"}
    assert res.dominant in {*res.mechanisms, "none"}


def test_page_renders_a_sample():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("Second-life grade" in m.value for m in at.markdown)
