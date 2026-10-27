"""ICA and peak extraction on synthetic curves with known peaks."""
import numpy as np
import pandas as pd
import pytest

from oracle.features import extract_peaks, relative_to_reference
from oracle.ica import ICAConfig, compute_ica
from oracle.schema import DischargeCurve

PEAKS = [(3.34, 0.30, 0.012), (3.22, 0.45, 0.015), (3.08, 0.20, 0.020)]  # (V, Ah, width)


def synthetic_discharge(peaks=PEAKS, n=4000, noise_v=0.0005, seed=0, cycle=0):
    """Q(V) as a sum of logistic steps -> dQ/dV has a peak of area A at each V_k."""
    rng = np.random.default_rng(seed)
    v = np.linspace(3.45, 2.95, n)  # time order: voltage falls
    q = sum(a / (1 + np.exp((v - v0) / w)) for v0, a, w in peaks)
    q = q - q[0]
    v_meas = np.round(v + rng.normal(0, noise_v, n), 3)  # 1 mV cycler resolution
    return DischargeCurve(v_meas, q, cell_id="syn", cycle=cycle, source="synthetic")


def test_recovers_peak_positions_and_areas():
    feats = extract_peaks(compute_ica(synthetic_discharge()))
    for name, (v0, area, _) in zip(["p1", "p2", "p3"], PEAKS):
        assert feats[f"{name}_pos"] == pytest.approx(v0, abs=0.006)
        assert feats[f"{name}_area"] == pytest.approx(area, rel=0.15)


def test_dqdv_is_nonnegative_on_uniform_grid():
    ica = compute_ica(synthetic_discharge())
    assert (ica.dqdv >= 0).all()
    assert np.allclose(np.diff(ica.voltage), ICAConfig().dv)


def test_lli_like_fade_shows_as_area_ratio():
    fresh = [synthetic_discharge(seed=s, cycle=s) for s in range(10)]
    aged_peaks = [(v, a * 0.8, w) for v, a, w in PEAKS]
    aged = [synthetic_discharge(aged_peaks, seed=s, cycle=100 + s) for s in range(3)]
    df = pd.DataFrame([extract_peaks(compute_ica(c)) for c in fresh + aged])
    rel = relative_to_reference(df, ref_cycles=10)
    last = rel.sort_values("cycle").iloc[-1]
    assert last["r_p2_area"] == pytest.approx(0.8, abs=0.05)
    assert abs(last["d_p2_pos"]) < 0.005


def test_window_with_no_data_raises():
    with pytest.raises(ValueError):
        compute_ica(synthetic_discharge(), ICAConfig(v_min=3.6, v_max=3.7))
