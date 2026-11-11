"""Build a self-contained HTML dashboard showing every machine (engine).

The output is ONE file (charts + data + styling are embedded, no internet or
server needed). Open it by double-clicking, or download it from Colab.

Usage (from the project root):
    python -m src.make_dashboard --data-dir data/raw --artifacts artifacts --out artifacts/dashboard.html
"""
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import CRITICAL_RUL, RUL_CAP, WARNING_RUL
from .data_loader import load_cmapss
from .predict import load_bundle, predict_history

COLORS = {"Critical": "#dc2626", "Warning": "#f59e0b", "Healthy": "#16a34a"}
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, Roboto, sans-serif"
PLOT_CONFIG = {"displaylogo": False, "responsive": True}


def _style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=50, r=20, t=30, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color="#334155", size=13),
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(gridcolor="#e5e7eb", zeroline=False)
    return fig


def _div(fig: go.Figure, first: bool = False) -> str:
    # The first chart embeds the plotly library once; the others reuse it.
    return fig.to_html(full_html=False, include_plotlyjs=first, config=PLOT_CONFIG)


def donut_chart(latest: pd.DataFrame) -> go.Figure:
    counts = latest["status"].value_counts().reindex(["Critical", "Warning", "Healthy"], fill_value=0)
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.62, sort=False,
        marker=dict(colors=[COLORS[s] for s in counts.index], line=dict(color="white", width=3)),
        textinfo="label+percent", hovertemplate="%{label}: %{value} machines<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{len(latest)}</b><br>machines", showarrow=False, font=dict(size=20))
    fig.update_layout(showlegend=False)
    return _style(fig, 340)


def ranking_chart(latest: pd.DataFrame) -> go.Figure:
    d = latest.sort_values("predicted_rul").reset_index(drop=True)
    fig = go.Figure(go.Bar(
        x=d["unit"].astype(str), y=d["predicted_rul"],
        marker_color=[COLORS[s] for s in d["status"]],
        customdata=d[["status", "cycle"]],
        hovertemplate="Machine %{x}<br>Predicted RUL: %{y:.0f} cycles<br>Status: %{customdata[0]}"
                      "<br>Latest cycle: %{customdata[1]}<extra></extra>",
    ))
    fig.add_hline(y=CRITICAL_RUL, line_dash="dash", line_color=COLORS["Critical"], annotation_text="critical")
    fig.add_hline(y=WARNING_RUL, line_dash="dash", line_color=COLORS["Warning"], annotation_text="warning")
    fig.update_xaxes(title="Machine ID (sorted: most urgent first)", type="category", tickangle=-60, tickfont_size=9)
    fig.update_yaxes(title="Predicted remaining life (cycles)")
    return _style(fig, 340)


def accuracy_chart(table: pd.DataFrame) -> go.Figure:
    top = max(table["actual_rul"].max(), table["predicted_rul"].max()) * 1.05
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color="#94a3b8", dash="dash"),
                             name="perfect prediction", hoverinfo="skip"))
    for status, grp in table.groupby("status"):
        fig.add_trace(go.Scatter(
            x=grp["actual_rul"], y=grp["predicted_rul"], mode="markers", name=status,
            marker=dict(color=COLORS[status], size=9, line=dict(color="white", width=1)),
            text=grp["unit"], hovertemplate="Machine %{text}<br>Actual %{x}<br>Predicted %{y:.0f}<extra></extra>",
        ))
    fig.update_xaxes(title="Actual remaining life (NASA answer)")
    fig.update_yaxes(title="Predicted remaining life")
    return _style(fig, 340)


def trajectory_chart(traj: pd.DataFrame, test: pd.DataFrame, sensor: str, actual: dict[int, float]) -> go.Figure:
    units = sorted(traj["unit"].unique())
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09, row_heights=[0.6, 0.4],
                        subplot_titles=("Predicted remaining life", f"{sensor} (key sensor)"))
    for unit in units:
        t = traj[traj["unit"] == unit]
        s = test[test["unit"] == unit]
        shown = unit == units[0]
        fig.add_trace(go.Scatter(x=t["cycle"], y=t["predicted_rul"], mode="lines", name="Predicted RUL",
                                 line=dict(color="#2563eb", width=3), visible=shown, legendgroup="p"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[t["cycle"].iloc[-1]], y=[actual.get(unit)], mode="markers",
                                 name="Actual RUL (NASA)", marker=dict(color="#0f172a", size=12, symbol="diamond"),
                                 visible=shown, legendgroup="a"), row=1, col=1)
        fig.add_trace(go.Scatter(x=s["cycle"], y=s[sensor], mode="lines", name=sensor,
                                 line=dict(color="#7c3aed", width=2), visible=shown, showlegend=False), row=2, col=1)
    fig.add_hline(y=CRITICAL_RUL, line_dash="dash", line_color=COLORS["Critical"], row=1, col=1)
    fig.add_hline(y=WARNING_RUL, line_dash="dash", line_color=COLORS["Warning"], row=1, col=1)
    buttons = []
    for i, unit in enumerate(units):
        vis = [False] * (3 * len(units))
        vis[3 * i: 3 * i + 3] = [True, True, True]
        buttons.append(dict(label=f"Machine {unit}", method="restyle", args=[{"visible": vis}]))
    fig.update_layout(updatemenus=[dict(buttons=buttons, direction="down", x=0.0, xanchor="left", y=1.22,
                                        yanchor="top", bgcolor="white", bordercolor="#cbd5e1")])
    fig.update_xaxes(title="Operating cycle", row=2, col=1)
    fig.update_yaxes(title="Cycles left", row=1, col=1)
    return _style(fig, 520)


def importance_chart(artifacts: Path) -> str:
    path = artifacts / "importance.csv"
    if not path.exists():
        return "<p class='muted'>importance.csv not found.</p>"
    imp = pd.read_csv(path).groupby("sensor")["importance_mean"].sum().sort_values().tail(10)
    labels = ["cycle (engine age)" if s == "cycle" else s for s in imp.index]
    fig = go.Figure(go.Bar(x=imp.values, y=labels, orientation="h", marker_color="#2563eb",
                           hovertemplate="%{y}: error grows by %{x:.2f} if scrambled<extra></extra>"))
    fig.update_xaxes(title="How much the error grows if this input is scrambled")
    return _div(_style(fig, 380))


def model_chart(results: pd.DataFrame) -> str:
    r = results.sort_values("test_MAE", ascending=False)
    fig = go.Figure(go.Bar(x=r["test_MAE"], y=r["model"], orientation="h",
                           marker_color=["#16a34a" if m == r["model"].iloc[-1] else "#94a3b8" for m in r["model"]],
                           text=r["test_MAE"].round(1), textposition="outside", cliponaxis=False))
    fig.update_xaxes(range=[0, r["test_MAE"].max() * 1.18])
    fig.update_xaxes(title="Average error in cycles (lower is better)")
    return _div(_style(fig, 300))


def machine_table(table: pd.DataFrame, cycle_hours: float | None) -> str:
    has_actual = "actual_rul" in table
    head = ["Machine", "Latest cycle", "Predicted RUL", *(["Actual RUL", "Error"] if has_actual else []),
            *(["Time left (h)"] if cycle_hours else []), "Health", "Status"]
    rows = []
    for r in table.sort_values("predicted_rul").itertuples():
        pct = min(r.predicted_rul, RUL_CAP) / RUL_CAP * 100
        cells = [f"<td data-v='{r.unit}'><b>Machine {r.unit}</b></td>", f"<td>{r.cycle}</td>",
                 f"<td data-v='{r.predicted_rul}'><b>{r.predicted_rul:.0f}</b> cycles</td>"]
        if has_actual:
            cells += [f"<td>{r.actual_rul:.0f}</td>", f"<td data-v='{r.error}'>{r.error:+.0f}</td>"]
        if cycle_hours:
            cells.append(f"<td>{r.predicted_rul * cycle_hours:,.0f}</td>")
        cells.append(f"<td data-v='{pct:.0f}'><div class='bar'><span style='width:{pct:.0f}%;"
                     f"background:{COLORS[r.status]}'></span></div></td>")
        cells.append(f"<td><span class='badge {r.status}'>{r.status}</span></td>")
        rows.append(f"<tr data-status='{r.status}'>{''.join(cells)}</tr>")
    ths = "".join(f"<th>{h} ⇅</th>" for h in head)
    return f"<table id='mt'><thead><tr>{ths}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def small_table(df: pd.DataFrame) -> str:
    return "<div class='xs'>" + df.round(2).to_html(index=False, classes="plain", border=0) + "</div>"


def build(args) -> Path:
    artifacts = Path(args.artifacts)
    bundle = load_bundle(str(artifacts / "model_bundle.joblib"))
    meta = bundle["metadata"]
    _, test, rul = load_cmapss(args.data_dir, meta.get("subset", "FD001"))

    latest = predict_history(bundle, test)
    traj = predict_history(bundle, test, every_cycle=True)
    table = latest.copy()
    actual: dict[int, float] = {}
    if len(rul) == len(latest):  # NASA's true answers, in engine order
        table["actual_rul"] = rul["RUL"].to_numpy().astype(float)
        table["error"] = (table["predicted_rul"] - table["actual_rul"]).round(1)
        actual = dict(zip(table["unit"], table["actual_rul"]))

    sensor = bundle["base_cols"][0]
    imp_path = artifacts / "importance.csv"
    if imp_path.exists():
        by_sensor = pd.read_csv(imp_path).groupby("sensor")["importance_mean"].sum().sort_values(ascending=False)
        sensor = next((s for s in by_sensor.index if s in bundle["base_cols"] and s.startswith("sensor")), sensor)

    counts = latest["status"].value_counts()
    tm = meta.get("test_metrics", {})
    kpis = [
        ("Machines monitored", f"{len(latest)}", "#2563eb"),
        ("Critical", f"{counts.get('Critical', 0)}", COLORS["Critical"]),
        ("Warning", f"{counts.get('Warning', 0)}", COLORS["Warning"]),
        ("Healthy", f"{counts.get('Healthy', 0)}", COLORS["Healthy"]),
        ("Average predicted life", f"{latest['predicted_rul'].mean():.0f} cycles", "#7c3aed"),
        ("Model average error", f"{tm.get('test_MAE', float('nan')):.1f} cycles", "#0891b2"),
    ]
    kpi_html = "".join(
        f"<div class='kpi' style='border-top-color:{c}'><div class='kv'>{v}</div><div class='kl'>{html.escape(k)}</div></div>"
        for k, v, c in kpis)

    donut = _div(donut_chart(latest), first=True)  # must come first in the page (loads plotly)
    ranking = _div(ranking_chart(latest))
    accuracy = _div(accuracy_chart(table)) if actual else ""
    trajectory = _div(trajectory_chart(traj, test, sensor, actual))
    results_path = artifacts / "results.csv"
    results = pd.read_csv(results_path) if results_path.exists() else None
    model_section = ""
    if results is not None:
        cols = ["model", "test_MAE", "test_RMSE", "test_R2", "test_NASA_score"]
        model_section = (f"<div class='card'><h3>Model comparison</h3>{model_chart(results)}</div>"
                         f"<div class='card'><h3>All metrics</h3>{small_table(results[cols])}</div>")
    stage_path = artifacts / "stage_mae.csv"
    stage_section = (f"<div class='card'><h3>Error by life stage</h3>{small_table(pd.read_csv(stage_path))}"
                     "<p class='muted'>Is the model accurate when it matters most, close to failure?</p></div>"
                     if stage_path.exists() else "")

    page = TEMPLATE
    for key, value in {
        "%%TITLE%%": "Predictive Maintenance Dashboard",
        "%%SUBTITLE%%": html.escape(f"Model: {meta.get('best_model')} · Data: NASA C-MAPSS {meta.get('subset')} (simulated engines)"),
        "%%KPIS%%": kpi_html, "%%DONUT%%": donut, "%%RANKING%%": ranking,
        "%%ACCURACY%%": f"<div class='card'><h3>Predicted vs actual</h3>{accuracy}</div>" if accuracy else "",
        "%%TRAJ%%": trajectory, "%%TABLE%%": machine_table(table, args.cycle_hours),
        "%%IMPORTANCE%%": importance_chart(artifacts), "%%MODELS%%": model_section, "%%STAGE%%": stage_section,
        "%%CRIT%%": str(CRITICAL_RUL), "%%WARN%%": str(WARNING_RUL),
        "%%NOW%%": datetime.now().strftime("%d %b %Y, %H:%M"),
        "%%SENSOR%%": html.escape(sensor),
    }.items():
        page = page.replace(key, value)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%%TITLE%%</title>
<style>
:root{--bg:#f1f5f9;--card:#fff;--ink:#0f172a;--muted:#64748b;--line:#e2e8f0}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,'Segoe UI',system-ui,-apple-system,Roboto,sans-serif}
header{background:linear-gradient(120deg,#0f172a,#1e3a8a);color:#fff;padding:28px 24px 64px}
header h1{margin:0 0 6px;font-size:28px}
header p{margin:0;opacity:.8}
main{max-width:1240px;margin:-44px auto 40px;padding:0 16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-bottom:18px}
.kpi{background:var(--card);border-radius:14px;padding:16px 18px;border-top:5px solid;box-shadow:0 4px 14px rgba(15,23,42,.08)}
.kv{font-size:26px;font-weight:800;white-space:nowrap}.kl{color:var(--muted);font-size:13px;margin-top:2px}
.grid{display:grid;gap:18px;margin-bottom:18px}
.g2{grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr))}
.card{background:var(--card);border-radius:14px;padding:18px 20px;box-shadow:0 4px 14px rgba(15,23,42,.06);min-width:0}
.card h3{margin:0 0 6px;font-size:16px}
.card{overflow:hidden}
.card .plotly-graph-div{width:100%!important}
.xs{overflow-x:auto}
.muted{color:var(--muted);font-size:13px}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0 12px}
.toolbar input{flex:1;min-width:160px;padding:9px 12px;border:1px solid #cbd5e1;border-radius:10px;font-size:14px}
.chip{border:1px solid #cbd5e1;background:#fff;border-radius:999px;padding:7px 14px;cursor:pointer;font-size:13px}
.chip.on{background:#0f172a;color:#fff;border-color:#0f172a}
.scroll{max-height:520px;overflow:auto;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{position:sticky;top:0;background:#f8fafc;cursor:pointer;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
tr:hover td{background:#f8fafc}
.bar{width:120px;height:8px;background:#e2e8f0;border-radius:99px;overflow:hidden}
.bar span{display:block;height:100%;border-radius:99px}
.badge{padding:3px 10px;border-radius:99px;font-size:12px;font-weight:700;color:#fff}
.badge.Critical{background:#dc2626}.badge.Warning{background:#f59e0b}.badge.Healthy{background:#16a34a}
table.plain th{cursor:default;position:static}
footer{color:var(--muted);font-size:12.5px;line-height:1.6;padding:6px 4px}
</style></head><body>
<header><h1>🛠️ %%TITLE%%</h1><p>%%SUBTITLE%%</p></header>
<main>
<div class="kpis">%%KPIS%%</div>
<div class="grid g2">
  <div class="card"><h3>Machine health</h3>%%DONUT%%</div>
  <div class="card"><h3>Which machines need attention first?</h3>%%RANKING%%</div>
</div>
<div class="grid">
  <div class="card"><h3>Watch one machine: remaining life over time</h3>
  <p class="muted">Pick a machine from the dropdown. The diamond is NASA's true remaining life at the last observed cycle.</p>%%TRAJ%%</div>
</div>
<div class="grid g2">%%ACCURACY%%<div class="card"><h3>What the model relies on</h3>%%IMPORTANCE%%</div></div>
<div class="grid">
  <div class="card"><h3>All machines</h3>
  <div class="toolbar"><input id="q" placeholder="Search machine, status..."><button class="chip on" data-s="">All</button>
  <button class="chip" data-s="Critical">Critical</button><button class="chip" data-s="Warning">Warning</button>
  <button class="chip" data-s="Healthy">Healthy</button></div>
  <div class="scroll">%%TABLE%%</div>
  <p class="muted">RUL = remaining useful life, counted in operating cycles. Click a column title to sort.</p></div>
</div>
<div class="grid g2">%%MODELS%%%%STAGE%%</div>
<footer>Status rule: Critical = %%CRIT%% cycles or fewer left, Warning = %%WARN%% or fewer, otherwise Healthy.
These limits are demonstration choices, not safety limits. Data is simulated (NASA C-MAPSS), so results show
benchmark performance, not proven performance on real factory machines. Key sensor shown: %%SENSOR%%. Generated %%NOW%%.</footer>
</main>
<script>
(function(){
  var tbl=document.getElementById('mt'),q=document.getElementById('q'),chips=document.querySelectorAll('.chip'),st='';
  function apply(){var s=q.value.toLowerCase();Array.prototype.forEach.call(tbl.tBodies[0].rows,function(r){
    var ok=(!st||r.dataset.status===st)&&r.textContent.toLowerCase().indexOf(s)>-1;r.style.display=ok?'':'none';});}
  q.addEventListener('input',apply);
  chips.forEach(function(c){c.addEventListener('click',function(){chips.forEach(function(x){x.classList.remove('on')});
    c.classList.add('on');st=c.dataset.s;apply();});});
  Array.prototype.forEach.call(tbl.tHead.rows[0].cells,function(th,i){th.addEventListener('click',function(){
    var dir=th.dataset.dir==='asc'?'desc':'asc';th.dataset.dir=dir;
    var rows=Array.prototype.slice.call(tbl.tBodies[0].rows);
    rows.sort(function(a,b){var x=a.cells[i].dataset.v||a.cells[i].textContent,y=b.cells[i].dataset.v||b.cells[i].textContent;
      var nx=parseFloat(x),ny=parseFloat(y);var c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);return dir==='asc'?c:-c;});
    rows.forEach(function(r){tbl.tBodies[0].appendChild(r)});});});
})();
window.addEventListener('load',function(){setTimeout(function(){window.dispatchEvent(new Event('resize'))},300)});
</script></body></html>
"""


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder with test_FD001.txt and RUL_FD001.txt")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--out", default="artifacts/dashboard.html")
    p.add_argument("--cycle-hours", type=float, default=None,
                   help="optional: hours of real operation per cycle, adds a 'time left (h)' column")
    out = build(p.parse_args(argv))
    print(f"Dashboard written to {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
