"""Phase 4: grading UI. Run from the battery-oracle folder with:

    streamlit run app/streamlit_app.py

Upload a partial discharge CSV (or pick a bundled sample) and the page shows
the second-life grade, the mechanism breakdown (LLI / LAM_PE / LAM_NE) and
the dQ/dV curve against a fresh-cell reference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for p in (ROOT, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))



def _pick_predict():
    """The real pipeline once it is importable and fitted, else the stub."""
    try:
        from oracle.predict import load_model, predict
        load_model()
        return predict, False
    except (ImportError, FileNotFoundError):
        from stub_predict import predict
        return predict, True


predict, USING_STUB = _pick_predict()

# Real Severson cells from the pipeline first, then the synthetic slow-rate
# files (their peaks sit outside the 4C model's range, which the app flags).
SAMPLES = {}
for f in sorted((ROOT / "data" / "samples").glob("severson_*.csv")):
    _, cell, cycle = f.stem.split("_", 2)
    SAMPLES[f"Severson cell {cell[1:]}, cycle {cycle.removeprefix('cycle')}"] = f
SAMPLES.update({
    "Synthetic slow-rate: mild lithium loss": APP_DIR / "samples" / "mild_lli.csv",
    "Synthetic slow-rate: cathode material loss": APP_DIR / "samples" / "lam_pe.csv",
    "Synthetic slow-rate: severe fade": APP_DIR / "samples" / "severe.csv",
})

# Grade colours read on both the light and dark Streamlit themes.
GRADE_COLOR = {"A": "#2e9e6e", "B": "#d08a1a", "C": "#d9483e"}
MECH_LABEL = {
    "LLI": "Loss of lithium inventory",
    "LAM_PE": "Loss of active material, cathode (LFP)",
    "LAM_NE": "Loss of active material, anode (graphite)",
}
MECH_COLOR = {"LLI": "#3b6fb6", "LAM_PE": "#c0612b", "LAM_NE": "#7a5bb5"}
PEAK_COLOR = "#8a8f98"
GRID = "rgba(128,128,128,0.2)"

st.set_page_config(page_title="Battery Oracle", page_icon="🔋", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      .grade-card {border-radius: 14px; padding: 22px 26px; margin-bottom: 8px;}
      .grade-card .letter {font-size: 64px; font-weight: 700; line-height: 1;}
      .grade-card .label {font-size: 13px; letter-spacing: .06em; text-transform: uppercase; opacity: .8;}
      .grade-card .use {font-size: 17px; margin-top: 10px; font-weight: 500;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Battery Oracle 2.0")
st.caption(
    "Second-life grading for retired LFP cells from a single partial discharge, "
    "using Incremental Capacity Analysis (dQ/dV) to read the degradation mechanism."
)

if USING_STUB:
    st.info(
        "Running on the placeholder grader. ICA and peak extraction are real, but the grade "
        "and mechanism split come from simple rules until the fitted pipeline is connected.",
        icon="🧪",
    )

with st.sidebar:
    st.header("Input")
    upload = st.file_uploader(
        "Partial discharge CSV",
        type=["csv"],
        help="Needs a voltage column plus either a capacity column, or current and time columns.",
    )
    sample = st.selectbox("…or try a sample", ["(none)", *SAMPLES], index=0 if upload else 1)
    with st.expander("File format"):
        st.markdown(
            "One row per sample, in time order, covering roughly **3.45 V down to 3.0 V**.\n\n"
            "Columns are matched by name: `voltage`, and `capacity`/`Ah`, "
            "or `current` + `time` (seconds)."
        )
        st.code("time_s,voltage_V,capacity_Ah\n0,3.450,0.000\n1.2,3.449,0.0004\n…", language="text")

source = upload if upload is not None else SAMPLES.get(sample)
if source is None:
    st.write("Upload a partial discharge CSV in the sidebar to grade a cell.")
    st.stop()

name = upload.name if upload is not None else sample


@st.cache_data(show_spinner=False)
def run(data: bytes, using_stub: bool, model_mtime: float):
    # using_stub and model_mtime only key the cache, so a refit model or the
    # switch from stub to real pipeline never shows a stale result.
    import io
    return predict(io.BytesIO(data))


def _model_mtime() -> float:
    files = list((ROOT / "models").glob("*.joblib"))
    return max((f.stat().st_mtime for f in files), default=0.0)


data = upload.getvalue() if upload is not None else Path(source).read_bytes()
try:
    with st.spinner("Computing dQ/dV and grading…"):
        res = run(data, USING_STUB, _model_mtime())
except ValueError as e:
    st.error(f"Couldn't grade **{name}**: {e}")
    st.stop()

st.subheader(name)
left, right = st.columns([1, 2], gap="large")

with left:
    fg = GRADE_COLOR.get(res.grade, "#888888")
    st.markdown(
        f"""<div class="grade-card" style="background:{fg}1f;border:1px solid {fg}66">
              <div class="label" style="color:{fg}">Second-life grade</div>
              <div class="letter" style="color:{fg}">{res.grade}</div>
              <div class="use">{res.second_life_use}</div>
            </div>""",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.metric("Confidence", f"{res.confidence:.0%}")
    dominant = "None" if res.dominant == "none" else res.dominant.replace("_", " ")
    c2.metric("Dominant mechanism", dominant)
    # Notes about the mechanism split sit under the bars they qualify;
    # out-of-range input is an error, everything else a warning.
    mech_notes = [n for n in res.notes if n.startswith(("Mechanism split", "Outside the synthetic"))]
    ood = res.features.get("ood_ratio")
    if ood is not None and np.isfinite(ood) and ood > 1:
        st.markdown(
            f"<span style='display:inline-block;padding:2px 10px;border-radius:999px;"
            f"background:#d08a1a26;color:#d08a1a;font-size:13px;font-weight:600'>"
            f"Outside calibration range · {ood:.1f}× training spread</span>",
            unsafe_allow_html=True,
        )
    for note in res.notes:
        if note in mech_notes:
            continue
        (st.error if note.startswith("Out of range") else st.warning)(note)

with right:
    st.markdown("**Mechanism breakdown**  \nEstimated loss against a fresh cell, by mechanism.")
    for note in mech_notes:
        # extrapolation is a warning; the calibration note is context
        (st.warning if note.startswith("Outside") else st.info)(note, icon="⚠️")
    keys = ["LLI", "LAM_PE", "LAM_NE"]
    vals = [res.mechanisms.get(k, 0.0) for k in keys]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=[MECH_LABEL[k] for k in keys],
            orientation="h",
            marker_color=[MECH_COLOR[k] for k in keys],
            text=[f"{v:.0%}" for v in vals],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=40, t=10, b=10),
        xaxis=dict(tickformat=".0%", range=[0, max(0.5, max(vals) * 1.25)], gridcolor=GRID),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown("**Incremental capacity curve (dQ/dV)**")
fig = go.Figure()
if res.reference is not None:
    fig.add_trace(go.Scatter(
        x=res.reference.voltage, y=res.reference.dqdv, name="Fresh reference",
        line=dict(color="#9aa0a6", width=2, dash="dash"),
        hovertemplate="%{x:.3f} V<br>%{y:.2f} Ah/V<extra>Fresh</extra>",
    ))
fig.add_trace(go.Scatter(
    x=res.ica.voltage, y=res.ica.dqdv, name="This cell",
    line=dict(color="#1f4e8c", width=2.5),
    hovertemplate="%{x:.3f} V<br>%{y:.2f} Ah/V<extra>This cell</extra>",
))
peaks = sorted({k.split("_")[0] for k in res.features if k.endswith("_pos")})
px, py, pt = [], [], []
for p in peaks:
    pos, h = res.features.get(f"{p}_pos"), res.features.get(f"{p}_height")
    if pos is not None and h is not None and np.isfinite(pos) and np.isfinite(h):
        px.append(pos); py.append(h); pt.append(p.upper())
fig.add_trace(go.Scatter(
    x=px, y=py, mode="markers+text", text=pt, textposition="top center", name="Peaks",
    marker=dict(color=PEAK_COLOR, size=8, symbol="diamond"), showlegend=False,
    hovertemplate="%{text}: %{x:.3f} V<extra></extra>",
))
fig.update_layout(
    height=420,
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(title="Voltage (V)", autorange="reversed", gridcolor=GRID),
    yaxis=dict(title="dQ/dV (Ah/V)", gridcolor=GRID, rangemode="tozero"),

    legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Voltage runs high to low, as on discharge. Peaks that shrink together point to lithium loss; "
    "a single peak shrinking on its own points to loss of active material on one electrode."
)

with st.expander("Peak features"):
    rows = []
    for p in peaks:
        rows.append({
            "Peak": p.upper(),
            "Position (V)": res.features.get(f"{p}_pos"),
            "Height (Ah/V)": res.features.get(f"{p}_height"),
            "Area (Ah)": res.features.get(f"{p}_area"),
            "FWHM (V)": res.features.get(f"{p}_fwhm"),
        })
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, use_container_width=True)
    st.caption(f"Capacity in window: {res.features.get('q_window', float('nan')):.3f} Ah")
