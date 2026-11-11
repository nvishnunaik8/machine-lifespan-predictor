"""Predictive Maintenance web app: enter machine information, get the remaining lifespan."""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CRITICAL_RUL, RUL_CAP, WARNING_RUL  # noqa: E402
from src.data_loader import parse_uploaded  # noqa: E402
from src.make_dashboard import COLORS, donut_chart, ranking_chart  # noqa: E402
from src.predict import load_bundle, predict_history  # noqa: E402

ARTIFACT_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "artifacts"))

st.set_page_config(page_title="Machine Lifespan Predictor", page_icon="🛠️", layout="wide")

# Friendly names for the NASA C-MAPSS columns (so users know what to type).
NAMES = {
    "setting_1": "Altitude setting", "setting_2": "Mach number setting", "setting_3": "Throttle angle setting",
    "sensor_1": "Fan inlet temperature (T2)", "sensor_2": "Low-pressure compressor outlet temp (T24)",
    "sensor_3": "High-pressure compressor outlet temp (T30)", "sensor_4": "Low-pressure turbine outlet temp (T50)",
    "sensor_5": "Fan inlet pressure (P2)", "sensor_6": "Bypass-duct pressure (P15)",
    "sensor_7": "High-pressure compressor outlet pressure (P30)", "sensor_8": "Physical fan speed (Nf)",
    "sensor_9": "Physical core speed (Nc)", "sensor_10": "Engine pressure ratio (epr)",
    "sensor_11": "Static pressure, HPC outlet (Ps30)", "sensor_12": "Fuel flow ratio (phi)",
    "sensor_13": "Corrected fan speed (NRf)", "sensor_14": "Corrected core speed (NRc)",
    "sensor_15": "Bypass ratio (BPR)", "sensor_16": "Burner fuel-air ratio (farB)",
    "sensor_17": "Bleed enthalpy (htBleed)", "sensor_18": "Demanded fan speed (Nf_dmd)",
    "sensor_19": "Demanded corrected fan speed (PCNfR_dmd)", "sensor_20": "HPT coolant bleed (W31)",
    "sensor_21": "LPT coolant bleed (W32)",
}

st.markdown(
    """
<style>
.block-container{padding-top:4rem;max-width:1200px}
.hero{background:linear-gradient(120deg,#0f172a,#1e3a8a);color:#fff;border-radius:16px;padding:22px 26px;margin-bottom:18px}
.hero h1{margin:0 0 4px;font-size:28px;color:#fff}.hero p{margin:0;opacity:.85}
.result{border-left:10px solid var(--c);background:#f8fafc;border-radius:14px;padding:18px 24px;margin:10px 0}
.result .big{font-size:48px;font-weight:800;color:#0f172a;line-height:1.1}
.result .badge{display:inline-block;background:var(--c);color:#fff;padding:3px 14px;border-radius:99px;font-weight:700}
.result .sub{color:#475569;margin-top:8px}
</style>
<div class="hero"><h1>🛠️ Machine Lifespan Predictor</h1>
<p>Enter your machine's sensor information and the model estimates how many operating cycles it can still run.</p></div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def get_bundle(name: str):
    path = ARTIFACT_DIR / name
    return load_bundle(str(path)) if path.exists() else None


@st.cache_data(show_spinner=False)
def read_csv_if_exists(name: str) -> pd.DataFrame | None:
    path = ARTIFACT_DIR / name
    return pd.read_csv(path) if path.exists() else None


def status_of(rul: float) -> str:
    return "Critical" if rul <= CRITICAL_RUL else "Warning" if rul <= WARNING_RUL else "Healthy"


ADVICE = {
    "Critical": "Failure is close. Inspect or service this machine as soon as possible.",
    "Warning": "Wear is visible. Plan maintenance soon and keep monitoring.",
    "Healthy": "No action needed right now.",
}


def show_result(rul: float, cycle: int, cycles_per_day: float, typical_error: float | None, note: str) -> None:
    status = status_of(rul)
    lines = [ADVICE[status]]
    if cycles_per_day > 0:
        days = rul / cycles_per_day
        lines.append(f"At {cycles_per_day:g} cycles per day that is about <b>{days:,.0f} days</b> "
                     f"(around <b>{(date.today() + timedelta(days=days)).strftime('%d %b %Y')}</b>).")
    if typical_error:
        lines.append(f"Typical error of this model: about ±{typical_error:.0f} cycles.")
    if rul >= RUL_CAP - 0.5:
        lines.append(f"The model never predicts above {RUL_CAP} cycles, because healthy machines show little wear yet.")
    lines.append(note)
    st.markdown(
        f"<div class='result' style='--c:{COLORS[status]}'><div class='sub'>Estimated remaining life at cycle {cycle}</div>"
        f"<div class='big'>{rul:.0f} <span style='font-size:20px'>cycles</span></div>"
        f"<span class='badge'>{status}</span><div class='sub'>{'<br>'.join(lines)}</div></div>",
        unsafe_allow_html=True,
    )
    st.progress(min(rul, RUL_CAP) / RUL_CAP, text=f"Health: {min(rul, RUL_CAP) / RUL_CAP:.0%} of the {RUL_CAP}-cycle scale")


# ---------------------------------------------------------------- quick check
def pick_examples(full, sample: pd.DataFrame | None) -> dict[str, dict]:
    """Example machines (last observed row) near failure / mid-life / early life."""
    if full is None or sample is None:
        return {}
    last = predict_history(full, sample)
    rows = sample.sort_values(["unit", "cycle"]).groupby("unit").tail(1).set_index("unit")
    order = last.sort_values("predicted_rul")["unit"].tolist()
    picks = {"Example: machine close to failure": order[0], "Example: machine in mid-life": order[len(order) // 2],
             "Example: machine early in life": order[-1]}
    return {label: rows.loc[u].to_dict() for label, u in picks.items()}


def tab_quick(quick, examples: dict[str, dict]) -> None:
    ranges = quick["input_ranges"]
    base_cols = quick["base_cols"]
    for col in base_cols:
        st.session_state.setdefault(f"in_{col}", float(ranges[col]["median"]))
    st.session_state.setdefault("in_cycle", int(ranges["cycle"]["median"]))

    def apply_example() -> None:
        row = examples.get(st.session_state["example"])
        if row:
            for col in base_cols:
                st.session_state[f"in_{col}"] = float(row[col])
            st.session_state["in_cycle"] = int(row["cycle"])

    if examples:
        st.selectbox("Start from an example (optional)", ["(type my own values)", *examples], key="example",
                     on_change=apply_example, help="Fills the form with the last readings of a NASA test machine.")
    with st.form("quick"):
        st.markdown("**1. How long has the machine been running?**")
        c1, c2 = st.columns(2)
        cycle = c1.number_input("Operating cycles completed so far", min_value=1, step=1, key="in_cycle")
        per_day = c2.number_input("Cycles per day (optional, gives days left)", min_value=0.0, step=0.5, value=0.0)
        st.markdown("**2. Current sensor readings**")
        cols = st.columns(3)
        values = {}
        for i, col in enumerate(base_cols):
            r = ranges[col]
            step = max((r["max"] - r["min"]) / 200, 1e-6)
            values[col] = cols[i % 3].number_input(f"{col} · {NAMES.get(col, '')}", step=step, format="%.4f", key=f"in_{col}")
        submitted = st.form_submit_button("Predict remaining life", type="primary")

    if submitted:
        row = pd.DataFrame([{"unit": 1, "cycle": int(cycle), **values}])
        rul = float(predict_history(quick, row).iloc[0]["predicted_rul"])
        outside = [c for c in base_cols if not ranges[c]["min"] <= values[c] <= ranges[c]["max"]]
        mae = quick["metadata"].get("test_metrics", {}).get("test_MAE")
        show_result(rul, int(cycle), per_day, mae,
                    "Quick check uses only the current readings. Upload sensor history for a more accurate estimate.")
        if outside:
            st.warning("These values are outside the range the model saw in training, so treat the result with "
                       f"care: {', '.join(outside[:6])}{'...' if len(outside) > 6 else ''}")


# -------------------------------------------------------------- history upload
def fleet_view(full, history: pd.DataFrame) -> None:
    latest = predict_history(full, history)
    counts = latest["status"].value_counts()
    c = st.columns(4)
    c[0].metric("Machines", len(latest)); c[1].metric("Critical", int(counts.get("Critical", 0)))
    c[2].metric("Warning", int(counts.get("Warning", 0))); c[3].metric("Healthy", int(counts.get("Healthy", 0)))
    left, right = st.columns([1, 2])
    left.plotly_chart(donut_chart(latest), width="stretch")
    right.plotly_chart(ranking_chart(latest), width="stretch")
    st.dataframe(latest.sort_values("predicted_rul").rename(columns={"cycle": "latest_cycle"}),
                 width="stretch", hide_index=True)
    st.markdown("#### Look at one machine")
    unit = st.selectbox("Machine", sorted(history["unit"].unique()))
    traj = predict_history(full, history[history["unit"] == unit], every_cycle=True)
    fig = go.Figure(go.Scatter(x=traj["cycle"], y=traj["predicted_rul"], mode="lines", line=dict(color="#2563eb", width=3)))
    fig.add_hline(y=CRITICAL_RUL, line_dash="dash", line_color=COLORS["Critical"])
    fig.add_hline(y=WARNING_RUL, line_dash="dash", line_color=COLORS["Warning"])
    fig.update_layout(height=320, margin=dict(l=40, r=10, t=10, b=40), xaxis_title="Operating cycle",
                      yaxis_title="Predicted remaining life (cycles)")
    st.plotly_chart(fig, width="stretch")
    if len(traj) < 20:
        st.info("Short history: predictions get more reliable once a machine has 20 or more cycles of data.")


def tab_upload(full) -> None:
    st.write("Upload a CSV with a `cycle` column, a `unit` column (machine number; optional for one machine) "
             "and the sensor columns. Many machines in one file are fine.")
    up = st.file_uploader("Sensor history file (CSV or NASA .txt)", type=["csv", "txt"])
    if up is None:
        return
    try:
        fleet_view(full, parse_uploaded(up))
    except ValueError as exc:
        st.error(f"Problem with the file: {exc}")


# --------------------------------------------------------------------- pages
def page_check(full, quick, examples) -> None:
    labels = (["⚡ Quick check (type values)"] if quick else []) + ["📄 Upload sensor history"]
    tabs = st.tabs(labels)
    if quick:
        with tabs[0]:
            tab_quick(quick, examples)
    with tabs[-1]:
        tab_upload(full)


def page_example_fleet(full) -> None:
    sample = read_csv_if_exists("sample_history.csv")
    if sample is None:
        st.info("No example data included.")
        return
    st.caption("Example fleet: 10 NASA test engines, observed up to a point before failure.")
    fleet_view(full, sample)


def page_performance() -> None:
    results = read_csv_if_exists("results.csv")
    if results is None:
        st.info("results.csv not found.")
        return
    st.dataframe(results.round(3), width="stretch", hide_index=True)
    pred = read_csv_if_exists("test_predictions.csv")
    if pred is not None:
        fig = go.Figure(go.Scatter(x=pred["actual_rul"], y=pred["predicted_rul"], mode="markers",
                                   marker=dict(color="#2563eb", size=8)))
        top = float(max(pred["actual_rul"].max(), pred["predicted_rul"].max()))
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color="#94a3b8", dash="dash")))
        fig.update_layout(height=420, showlegend=False, xaxis_title="Actual remaining life",
                          yaxis_title="Predicted remaining life", margin=dict(l=40, r=10, t=10, b=40))
        st.plotly_chart(fig, width="stretch")
    stage = read_csv_if_exists("stage_mae.csv")
    if stage is not None:
        st.markdown("**Error by life stage**")
        st.dataframe(stage.round(2), width="stretch", hide_index=True)
    st.caption("Tested on the official NASA test engines, which the model never saw in training. "
               "The data is simulated, so this shows benchmark performance, not proven real-factory performance.")


def page_explain() -> None:
    imp = read_csv_if_exists("importance.csv")
    if imp is None:
        st.info("importance.csv not found.")
        return
    top = imp.groupby("sensor")["importance_mean"].sum().sort_values().tail(12)
    labels = ["cycle (machine age)" if s == "cycle" else s for s in top.index]
    fig = go.Figure(go.Bar(x=top.values, y=labels, orientation="h", marker_color="#2563eb"))
    fig.update_layout(height=420, xaxis_title="How much the error grows if this input is scrambled",
                      margin=dict(l=40, r=10, t=10, b=40))
    st.plotly_chart(fig, width="stretch")
    st.caption("This shows what the model relies on. It does not prove these sensors cause the wear.")


def main() -> None:
    full, quick = get_bundle("model_bundle.joblib"), get_bundle("model_quick.joblib")
    if full is None:
        st.error("Model file not found. Put model_bundle.joblib in the artifacts folder.")
        st.stop()
    sample = read_csv_if_exists("sample_history.csv")
    examples = pick_examples(full, sample)
    page = st.sidebar.radio("Menu", ["Check a machine", "Example fleet", "Model performance", "How it decides"])
    st.sidebar.caption("Status rule: Critical = 30 cycles or fewer left, Warning = 60 or fewer. "
                       "These are demonstration limits, not safety limits.")
    if page == "Check a machine":
        page_check(full, quick, examples)
    elif page == "Example fleet":
        page_example_fleet(full)
    elif page == "Model performance":
        page_performance()
    else:
        page_explain()
    with st.expander("Model details"):
        st.code(json.dumps(full["metadata"], indent=2))


main()
