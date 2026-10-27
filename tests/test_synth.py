"""Synthetic half-cell model: calibration and the physics each mode should show."""
import numpy as np
import pytest

from oracle import synth
from oracle.ica import ICAConfig, compute_ica
from oracle.schema import DischargeCurve

SLOW = 0.05


def test_fresh_cell_matches_severson():
    c = synth.generate(seed=0)
    assert isinstance(c, DischargeCurve)
    assert c.source == "synthetic"
    assert (c.current < 0).all()
    assert c.voltage.max() < synth.V_MAX and c.voltage.min() >= synth.V_MIN - 0.01
    assert c.total_capacity == pytest.approx(1.07, abs=0.03)       # Severson cycle 10
    ica = compute_ica(c, ICAConfig(v_min=2.85, v_max=3.40))
    assert ica.voltage[ica.dqdv.argmax()] == pytest.approx(3.13, abs=0.04)


def test_capacity_loss_follows_each_mode():
    q0 = synth.generate(c_rate=SLOW).total_capacity
    assert synth.generate(lli=0.10, c_rate=SLOW).total_capacity == pytest.approx(q0 - 0.10 * q0, abs=0.01)
    # The fitted A123 balance has LFP nearly limiting, so even small LAM_PE
    # costs capacity and makes LFP the limiting electrode.
    small = synth.generate(lam_pe=0.05, c_rate=SLOW)
    assert small.meta["limiting_electrode"] == "PE" and small.total_capacity < q0 - 0.03
    # LAM_NE is hidden: graphite has spare capacity.
    assert synth.generate(lam_ne=0.10, c_rate=SLOW).total_capacity == pytest.approx(q0, abs=0.01)


def test_lli_removes_high_voltage_peak_and_lam_pe_the_low_one():
    cfg = ICAConfig(v_min=3.0, v_max=3.45, savgol_window_v=0.02)

    def area(curve, lo, hi):
        ica = compute_ica(curve, cfg)
        m = (ica.voltage >= lo) & (ica.voltage < hi)
        return np.trapezoid(ica.dqdv[m], ica.voltage[m])

    fresh = synth.generate(c_rate=SLOW)
    lli = synth.generate(lli=0.2, c_rate=SLOW)
    lam_pe = synth.generate(lam_pe=0.3, c_rate=SLOW)
    hi_band, lo_band = (3.32, 3.45), (3.0, 3.22)
    assert area(lli, *hi_band) < 0.75 * area(fresh, *hi_band)
    assert area(lli, *lo_band) > 0.8 * area(fresh, *lo_band)
    assert area(lam_pe, *lo_band) < 0.5 * area(fresh, *lo_band)
    assert area(lam_pe, *hi_band) > 0.8 * area(fresh, *hi_band)


def test_resistance_shifts_without_losing_capacity():
    a, b = synth.generate(), synth.generate(r_ohm=0.02)
    assert b.total_capacity == pytest.approx(a.total_capacity, abs=0.02)
    assert np.median(b.voltage) < np.median(a.voltage) - 0.06


def test_partial_window_is_rezeroed():
    c = synth.generate(window=(3.30, 3.00))
    assert c.voltage.max() <= 3.30 and c.voltage.min() >= 3.00
    assert c.capacity[0] == 0


def test_dominant_mode_labels():
    assert synth.dominant_mode(0.2, 0.01, 0.0) == "lli"
    assert synth.dominant_mode(0.1, 0.1, 0.1) == "mixed"
    assert synth.dominant_mode(0, 0, 0) == "fresh"


def test_dataset_roundtrip(tmp_path):
    curves, labels = synth.generate_dataset(n_cells_per_profile=1, cycle_step=500, seed=1)
    assert len(curves) == len(labels) == 4 * 3
    assert set(labels.profile) == set(synth.PROFILES)
    path = synth.save_dataset(curves, labels, tmp_path / "ds")
    back, lab2 = synth.load_dataset(path)
    assert len(back) == len(curves)
    np.testing.assert_allclose(back[5].voltage, curves[5].voltage)
    assert back[5].meta["lli"] == pytest.approx(labels.lli[5])
