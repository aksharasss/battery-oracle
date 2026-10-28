"""Phase 4: grading UI. Run from the battery-oracle folder with:

    streamlit run app/streamlit_app.py

Upload a partial discharge CSV (or pick a bundled sample) and the page shows
the second-life grade, the mechanism breakdown (LLI / LAM_PE / LAM_NE) and
the dQ/dV curve against a fresh-cell reference, drawn as one lab-notebook
sheet: graphite ink on a millimetre grid, with a red pen for what changed.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import numpy as np
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

# Real cells from the pipeline first (Severson 4C, then Wheeler C/25, which
# takes the slower twin-fit path), then the app's own synthetic files.
SAMPLES = {}
for f in sorted((ROOT / "data" / "samples").glob("severson_*.csv")):
    _, cell, cycle = f.stem.split("_", 2)
    SAMPLES[f"Severson cell {cell[1:]}, cycle {cycle.removeprefix('cycle')} (4C)"] = f
for f in sorted((ROOT / "data" / "samples").glob("wheeler_*.csv"),
                key=lambda f: int(f.stem.split("_")[2].removeprefix("cycle"))):
    _, cell, cycle, _rate = f.stem.split("_", 3)
    SAMPLES[f"Wheeler cell {cell}, cycle {cycle.removeprefix('cycle')} (C/25)"] = f
SAMPLES.update({
    "Synthetic slow-rate: mild lithium loss": APP_DIR / "samples" / "mild_lli.csv",
    "Synthetic slow-rate: cathode material loss": APP_DIR / "samples" / "lam_pe.csv",
    "Synthetic slow-rate: severe fade": APP_DIR / "samples" / "severe.csv",
})

MECH_LABEL = {
    "LLI": "Loss of lithium inventory",
    "LAM_PE": "Loss of active material, cathode (LFP)",
    "LAM_NE": "Loss of active material, anode (graphite)",
}

# Two sheets of the same notebook. Three ink values on each (ground, graphite,
# red pen) plus the grid and a muted graphite for secondary text.
SHEETS = {
    "dark": dict(ground="#0d1611", grid="#1b2c22", grid2="#2d4a37", ink="#eaf2ec", muted="#a3bcab",
                 rule="#55705f", pen="#ff7a66", panel="#101b14", fresh="#8fae9b", select="#29473a"),
    "light": dict(ground="#f6faf7", grid="#d6e8dc", grid2="#afd0ba", ink="#1d2227", muted="#4b5a54",
                  rule="#1d2227", pen="#c42a22", panel="#eef5f0", fresh="#5d6b65", select="#cfe5d6"),
}

st.set_page_config(page_title="Battery Oracle", page_icon=":material/battery_charging_full:", layout="wide")

with st.sidebar:
    st.markdown('<div class="bo-mast">Battery Oracle <span>2.0</span></div>', unsafe_allow_html=True)
    upload = st.file_uploader(
        "Partial discharge CSV",
        type=["csv"],
        help="Needs a voltage column plus either a capacity column, or current and time columns.",
    )
    default = next((i + 1 for i, k in enumerate(SAMPLES) if "cycle 311" in k), 1)
    sample = st.selectbox("Or pick a sample", ["(none)", *SAMPLES], index=0 if upload else default)
    dark = st.toggle("Dark sheet", value=True, help="Switch between the dark and light notebook sheet.")
    with st.expander("File format"):
        st.markdown(
            "One row per sample, in time order, covering roughly **3.45 V down to 2.9 V**. "
            "A slow (about C/25) discharge is graded by fitting the half-cell model directly, "
            "which takes up to a minute; a fast (4C) one uses the feature model and is instant.\n\n"
            "Columns are matched by name: `voltage`, and `capacity`/`Ah`, "
            "or `current` + `time` (seconds)."
        )
        st.code("time_s,voltage_V,capacity_Ah\n0,3.450,0.000\n1.2,3.449,0.0004\n…", language="text")

T = SHEETS["dark" if dark else "light"]
st.html(
    f"""
<style>
@import url("https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Archivo:wght@900&family=Caveat:wght@600&display=swap");
:root {{
  --ground: {T['ground']}; --grid: {T['grid']}; --grid2: {T['grid2']}; --ink: {T['ink']}; --muted: {T['muted']};
  --rule: {T['rule']}; --pen: {T['pen']}; --panel: {T['panel']}; --select: {T['select']};
  --ui: "Atkinson Hyperlegible", "Segoe UI", system-ui, sans-serif;
  --display: "Archivo", "Segoe UI", sans-serif;
  --hand: "Caveat", "Segoe Print", cursive;
}}
/* the sheet: millimetre grid with a heavier line every 5 mm */
.stApp {{ background-color: var(--ground); color: var(--ink); font-family: var(--ui);
  background-image: linear-gradient(var(--grid2) 1px, transparent 1px), linear-gradient(90deg, var(--grid2) 1px, transparent 1px),
    linear-gradient(var(--grid) 1px, transparent 1px), linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size: 50px 50px, 50px 50px, 10px 10px, 10px 10px; }}
.stApp p, .stApp label, .stApp li, [data-testid="stMarkdownContainer"] {{ color: var(--ink); }}
.stApp h1, .stApp h2, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp dt, .stApp dd, .stApp td, .stApp th,
[data-testid="stMarkdownContainer"] *:not([data-testid="stIconMaterial"]), [data-baseweb="select"] div, [data-baseweb="popover"] li,
[data-testid="stSpinner"] p {{ font-family: var(--ui) !important; }}
.stApp .bo-letter {{ font-family: var(--display) !important; }}
::selection {{ background: var(--pen); color: var(--ground); }}
*:focus-visible {{ outline: 2px solid var(--pen) !important; outline-offset: 2px; }}
* {{ scrollbar-color: var(--rule) var(--ground); }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1240px; padding-top: 1.2rem; padding-bottom: 3rem; }}
/* sidebar: the notebook's margin column */
[data-testid="stSidebar"] {{ background: var(--panel); border-right: 1px solid var(--rule); }}
[data-testid="stSidebar"] * {{ color: var(--ink); }}
.bo-mast {{ font-family: var(--ui); font-weight: 700; font-size: 1.35rem; padding: .2rem 0 1rem;
  border-bottom: 1px solid var(--rule); margin-bottom: .6rem; }}
.bo-mast span {{ color: var(--pen); }}
[data-testid="stWidgetLabel"] p {{ font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; color: var(--muted) !important; }}
[data-testid="stFileUploaderDropzone"] {{ background: transparent; border: 1px dashed var(--rule); border-radius: 0; }}
[data-testid="stFileUploaderDropzone"] button, .stButton button {{ background: transparent; color: var(--ink);
  border: 1px solid var(--rule); border-radius: 0; }}
[data-testid="stFileUploaderDropzone"] button:hover {{ border-color: var(--pen); color: var(--pen); }}
[data-baseweb="select"] > div {{ background: var(--ground) !important; border: 1px solid var(--rule) !important; border-radius: 0 !important; color: var(--ink) !important; }}
[data-baseweb="select"] svg {{ fill: var(--ink); }}
[data-baseweb="select"] input {{ color: var(--ink) !important; background: transparent !important; }}
[data-baseweb="select"] input:focus-visible {{ outline: none !important; }}
[data-baseweb="select"] > div:focus-within {{ border-color: var(--pen) !important; }}
[data-testid="stCheckbox"] label > div:first-child {{ border: 1px solid var(--rule); }}
[data-baseweb="popover"] ul, [data-baseweb="popover"] li {{ background: var(--panel) !important; color: var(--ink) !important; font-family: var(--ui); }}
[data-baseweb="popover"] li[aria-selected="true"], [data-baseweb="popover"] li:hover {{ background: var(--select) !important; }}
[data-testid="stExpander"] details {{ border: 1px solid var(--rule); border-radius: 0; background: transparent; }}
[data-testid="stExpander"] summary:hover p {{ color: var(--pen); }}
[data-testid="stCode"] pre, .stCode pre {{ background: var(--ground) !important; border: 1px solid var(--rule); border-radius: 0; }}
[data-testid="stCode"] code {{ color: var(--ink) !important; }}
[data-testid="stSpinner"] p, [data-testid="stSpinner"] {{ color: var(--muted); font-size: 1rem; }}
[data-testid="stSpinner"] i {{ border-top-color: var(--pen) !important; }}
.stApp code {{ color: var(--ink); background: transparent; }}
/* the result sheet */
.bo-head {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem; flex-wrap: wrap;
  border-bottom: 1px solid var(--rule); padding-bottom: .7rem; margin-bottom: 1.4rem; }}
.bo-head h1 {{ font-family: var(--ui); font-weight: 700; font-size: 1.9rem; line-height: 1.15; margin: 0; padding: 0; color: var(--ink); }}
.bo-head .sub {{ color: var(--muted); font-size: 1rem; margin-top: .3rem; }}
.bo-stamp {{ display: inline-block; border: 2px solid var(--pen); color: var(--pen); font-weight: 700; font-size: .78rem;
  letter-spacing: .1em; text-transform: uppercase; padding: .3rem .6rem; transform: rotate(-3deg); white-space: nowrap; }}
.bo-grade {{ padding: .2rem 0 0; }}
.bo-letter {{ font-family: var(--display); font-weight: 900; font-size: 8.5rem; line-height: .82; letter-spacing: -.02em; margin: 0 0 .6rem; }}
.bo-letter.C {{ color: var(--pen); }}
.bo-use {{ font-size: 1.25rem; font-weight: 700; line-height: 1.3; margin: 0 0 1rem; }}
.bo-facts {{ display: grid; grid-template-columns: auto 1fr; gap: .35rem 1.2rem; font-size: 1rem; margin: 0; }}
.bo-facts dt {{ color: var(--muted); }} .bo-facts dd {{ margin: 0; font-weight: 700; font-variant-numeric: tabular-nums; }}
.bo-mech {{ border-left: 1px solid var(--rule); padding: .2rem 0 .4rem 2rem; }}
.bo-mech h2, .bo-plot h2 {{ font-family: var(--ui); font-size: 1.15rem; font-weight: 700; margin: 0 0 .25rem; padding: 0; color: var(--ink); }}
.bo-mech .lead, .bo-plot .lead {{ color: var(--muted); margin: 0 0 1.1rem; font-size: .95rem; }}
.bo-bar {{ margin: 1.05rem 0; font-size: 1.05rem; }}
.bo-bar .row {{ display: flex; justify-content: space-between; gap: 1rem; margin-bottom: .35rem; }}
.bo-bar.dom .name {{ font-weight: 700; text-decoration: underline 2px var(--pen); text-underline-offset: 5px; }}
.bo-bar .track {{ display: block; height: 1.25rem; border: 1px solid var(--rule); position: relative;
  background-image: repeating-linear-gradient(90deg, var(--rule) 0 1px, transparent 1px var(--tick)); }}
.bo-bar .fill {{ position: absolute; inset: 0 auto 0 0; background: var(--ink); }}
.bo-bar.dom .fill {{ background: var(--pen); }}
.bo-bar .val {{ font-weight: 700; font-variant-numeric: tabular-nums; }}
.bo-note {{ border: 1.5px solid var(--rule); background: transparent; padding: .6rem .85rem; margin: .8rem 0 0; font-size: .95rem; line-height: 1.45; }}
.bo-note b {{ display: inline-block; font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; margin: 0 .5rem .15rem 0; padding: 0 .35rem; border: 1px solid currentColor; color: var(--muted); }}
.bo-note.pen {{ border: 2px solid var(--pen); }} .bo-note.pen b {{ color: var(--pen); }}
.bo-plot {{ margin-top: 1.4rem; padding-left: 2rem; }}
.bo-caption {{ color: var(--muted); font-size: .92rem; margin: .4rem 0 0; max-width: 75ch; }}
.bo-table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: .95rem; }}
.bo-table th {{ text-align: left; font-weight: 700; color: var(--muted); border-bottom: 1px solid var(--rule); padding: .4rem .5rem; }}
.bo-table td {{ border-bottom: 1px solid var(--grid2); padding: .4rem .5rem; }}
.bo-empty {{ border: 1px dashed var(--rule); padding: 2rem; max-width: 40rem; font-size: 1.1rem; }}
[data-testid="stPlotlyChart"] {{ border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); background: transparent; margin-left: 2rem; }}
@media (max-width: 760px) {{
  .bo-letter {{ font-size: 6rem; }} .bo-mech {{ border-left: 0; border-top: 1px solid var(--rule); padding: 1rem 0 0; }} .bo-plot {{ padding-left: 0; }} [data-testid="stPlotlyChart"] {{ margin-left: 0; }}
}}
</style>
""",
)


def note(title: str, body: str, pen: bool = False) -> str:
    return (f'<div class="bo-note{" pen" if pen else ""}"><b>{html.escape(title)}</b>'
            f'{html.escape(body)}</div>')


if USING_STUB:
    st.markdown(note("Placeholder grader",
                     "ICA and peak extraction are real, but the grade and mechanism split come from "
                     "simple rules until the fitted pipeline is connected.", pen=True),
                unsafe_allow_html=True)

source = upload if upload is not None else SAMPLES.get(sample)
if source is None:
    st.markdown('<div class="bo-empty">Upload a partial-discharge CSV in the sidebar, or pick one of the '
                'sample cells, to grade a cell.</div>', unsafe_allow_html=True)
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
    with st.spinner("Computing dQ/dV and grading… a slow-rate (C/25) file runs the full "
                    "half-cell fit and can take up to a minute."):
        res = run(data, USING_STUB, _model_mtime())
except ValueError as e:
    st.markdown(note(f"Couldn't grade {name}", f"{e}. Check the file has a voltage column and a "
                     "capacity column (or current and time), covering the discharge.", pen=True),
                unsafe_allow_html=True)
    st.stop()

path = res.features.get("path")
if path == "slow":
    stamp, method = "Slow path · twin fit", "Half-cell twin fit (C/25)"
elif path == "fast":
    stamp, method = "Fast path · 4C model", "4C feature model"
else:
    stamp, method = "Placeholder", "Rule-based placeholder"
st.markdown(
    f'<div class="bo-head"><div><h1>{html.escape(name)}</h1>'
    f'<div class="sub">Second-life grading from one partial discharge, read through its dQ/dV curve</div></div>'
    f'<span class="bo-stamp">{html.escape(stamp)}</span></div>',
    unsafe_allow_html=True,
)

mech_notes = [n for n in res.notes if n.startswith(("Mechanism split", "Outside the synthetic"))]
other_notes = [n for n in res.notes if n not in mech_notes]
left, right = st.columns([5, 7], gap="large")

with left:
    dominant = "None (fresh)" if res.dominant == "none" else res.dominant.replace("_", " ")
    facts = [("Confidence", f"{res.confidence:.0%}"), ("Dominant mechanism", dominant), ("Method", method)]
    rms = res.features.get("fit_rms")
    if path == "slow" and rms is not None and np.isfinite(rms):
        facts.append(("Fit RMS", f"{rms:.3f}"))
    ood = res.features.get("ood_ratio")
    if ood is not None and np.isfinite(ood):
        facts.append(("Calibration distance", f"{ood:.1f}× training spread"))
    dl = "".join(f"<dt>{k}</dt><dd>{html.escape(v)}</dd>" for k, v in facts)
    notes_html = "".join(note("Outside calibration range" if n.startswith("Outside") else "Mechanism split",
                              n, pen=n.startswith("Outside")) for n in mech_notes)
    for n in other_notes:
        if n.startswith("Out of range"):
            notes_html += note("Out of range", n, pen=True)
        elif n.startswith("Slow-rate path"):
            notes_html += note("Method", n)
        else:
            notes_html += note("Caveat", n)
    st.markdown(
        f'<div class="bo-grade"><div class="bo-letter {res.grade}" role="img" aria-label="Second-life grade {res.grade}">{res.grade}</div>'
        f'<p class="bo-use">{html.escape(res.second_life_use)}</p><dl class="bo-facts">{dl}</dl></div>{notes_html}',
        unsafe_allow_html=True,
    )

with right:
    keys = ["LLI", "LAM_PE", "LAM_NE"]
    vals = [float(res.mechanisms.get(k, 0.0)) for k in keys]
    scale = max(0.4, max(vals) * 1.25)
    rows = "".join(
        f'<div class="bo-bar{" dom" if k == res.dominant else ""}"><div class="row"><span class="name">{MECH_LABEL[k]}</span>'
        f'<span class="val">{v:.0%}</span></div>'
        f'<span class="track" style="--tick:{0.1 / scale * 100:.3f}%"><span class="fill" style="width:{min(v / scale, 1) * 100:.1f}%"></span></span></div>'
        for k, v in zip(keys, vals)
    )
    st.markdown(
        f'<div class="bo-mech"><h2>Mechanism breakdown</h2>'
        f'<p class="lead">Estimated loss against a fresh cell. One scale, ticks every 10%, up to {scale:.0%}.</p>'
        f'{rows}</div>',
        unsafe_allow_html=True,
    )

    # dQ/dV on the graph paper, with the red-pen note on the biggest change from fresh
    fig = go.Figure()
    pen_note = None
    if res.reference is not None:
        fig.add_trace(go.Scatter(
            x=res.reference.voltage, y=res.reference.dqdv, name="Fresh reference",
            line=dict(color=T["fresh"], width=2, dash="dash"),
            hovertemplate="%{x:.3f} V<br>%{y:.2f} Ah/V<extra>Fresh</extra>",
        ))
        # the red-pen note marks how far the main peak fell from the fresh cell's
        v, y = np.asarray(res.ica.voltage), np.asarray(res.ica.dqdv)
        ref_peak = float(np.nanmax(res.reference.dqdv))
        if y.size and ref_peak > 0:
            i = int(np.nanargmax(y))
            fall = 1 - float(y[i]) / ref_peak
            if fall > 0.10:  # only changes worth a note
                # aim at the rising (high-voltage) flank at 85% of the peak, on this cell's curve alone
                hv = np.where((v > v[i]) & (y <= 0.85 * y[i]))[0]
                j = int(hv[np.argmin(v[hv] - v[i])]) if hv.size else i
                pen_note = (float(v[j]), float(y[j]), fall, float(np.nanmin(v)), float(np.nanmax(v)))
    fig.add_trace(go.Scatter(
        x=res.ica.voltage, y=res.ica.dqdv, name="This cell",
        line=dict(color=T["ink"], width=3),
        hovertemplate="%{x:.3f} V<br>%{y:.2f} Ah/V<extra>This cell</extra>",
    ))
    if pen_note:
        vx, vy, frac, vmin, vmax = pen_note
        # the note is written above the plot's top-right corner, clear of both curves,
        # and a short red-pen arrow points at this cell's peak flank
        pen = dict(family="Caveat, Segoe Print, cursive", size=24, color=T["pen"])
        fig.add_annotation(xref="paper", yref="paper", x=1, y=1.0, xanchor="right", yanchor="bottom", xshift=-8,
                           text=f"peak down {frac:.0%} from fresh", showarrow=False, font=pen)
        fig.add_annotation(x=vx, y=vy, text="", showarrow=True, arrowhead=2, arrowwidth=2,
                           arrowcolor=T["pen"], ax=-45, ay=30)
    axis = dict(gridcolor=T["grid2"], minor=dict(showgrid=True, gridcolor=T["grid"]), linecolor=T["rule"], zeroline=False, tickfont=dict(size=14, color=T["muted"]),
                title_font=dict(size=14, color=T["muted"]), showline=True, mirror=False, automargin=True, title_standoff=14)
    fig.update_layout(
        height=340,
        margin=dict(l=20, r=24, t=44, b=20),
        paper_bgcolor=T["ground"], plot_bgcolor=T["ground"],
        font=dict(family="Atkinson Hyperlegible, Segoe UI, sans-serif", color=T["ink"]),
        xaxis=dict(title="Voltage (V), high to low as on discharge", autorange="reversed", **axis),
        yaxis=dict(title="dQ/dV (Ah/V)", rangemode="tozero", **axis),
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor=T["panel"], font=dict(color=T["ink"])),
    )
    st.markdown(
        '<div class="bo-plot"><h2>Incremental capacity curve, dQ/dV</h2>'
        '<p class="lead">This cell (solid) against a fresh cell (dashed). Peaks shrinking together mean lithium loss; one alone, electrode material loss.</p></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, theme=None)


with st.expander("Peak features"):
    peaks = sorted({k.split("_")[0] for k in res.features if k.endswith("_pos") and not k.startswith("d_")})

    def cell(x, fmt):
        return fmt.format(x) if isinstance(x, (int, float)) and np.isfinite(x) else "not found"

    body = "".join(
        f"<tr><td>{p.upper()}</td><td>{cell(res.features.get(f'{p}_pos'), '{:.3f}')}</td>"
        f"<td>{cell(res.features.get(f'{p}_height'), '{:.2f}')}</td><td>{cell(res.features.get(f'{p}_area'), '{:.4f}')}</td>"
        f"<td>{cell(res.features.get(f'{p}_fwhm'), '{:.3f}')}</td></tr>"
        for p in peaks
    )
    q = res.features.get("q_window", float("nan"))
    st.markdown(
        '<table class="bo-table"><thead><tr><th>Peak</th><th>Position (V)</th><th>Height (Ah/V)</th>'
        f'<th>Area (Ah)</th><th>FWHM (V)</th></tr></thead><tbody>{body}</tbody></table>'
        f'<p class="bo-caption">Capacity in the voltage window: {cell(q, "{:.3f}")} Ah</p>',
        unsafe_allow_html=True,
    )
