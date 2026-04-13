from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from dashboard.data_loading import CycleRecord, list_available_cells, load_discharge_cycles
from dashboard.modeling import (
    TwinArtifacts, _FitResult,
    fit_twin_models, compute_eol_artifacts,
)
from dashboard.simulation import USAGE_PROFILES, simulate_what_if
from dashboard.visualization import (
    plot_anomalies,
    plot_capacity_fade,
    plot_health_indicators,
    plot_rul,
    plot_signals,
    plot_what_if,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config & custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Battery Digital Twin",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ── Google font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Dark background ── */
    .stApp { background: #0d1117; color: #e6edf3; }
    section[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }

    /* ── KPI Cards ── */
    .kpi-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 20px 22px 16px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.45);
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 14px 14px 0 0;
    }
    .kpi-soh::before    { background: linear-gradient(90deg, #22c55e, #16a34a); }
    .kpi-resist::before { background: linear-gradient(90deg, #ef4444, #dc2626); }
    .kpi-rul::before    { background: linear-gradient(90deg, #6366f1, #4f46e5); }
    .kpi-eol::before    { background: linear-gradient(90deg, #f59e0b, #d97706); }

    .kpi-label  { font-size: 0.72rem; font-weight: 500; color: #8b949e; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }
    .kpi-value  { font-size: 1.85rem; font-weight: 700; color: #e6edf3; }
    .kpi-delta  { font-size: 0.78rem; color: #8b949e; margin-top: 4px; }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.05rem; font-weight: 600; color: #c9d1d9;
        margin: 28px 0 12px; padding-bottom: 8px;
        border-bottom: 1px solid #21262d;
        display: flex; align-items: center; gap: 8px;
    }

    /* ── Health bar ── */
    .health-bar-outer {
        background: #21262d; border-radius: 8px; height: 10px;
        margin-top: 6px; overflow: hidden;
    }
    .health-bar-inner {
        height: 100%; border-radius: 8px;
        transition: width 0.6s ease;
    }

    /* ── Quality badge ── */
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em;
    }
    .badge-green  { background: #0f4c24; color: #3fb950; }
    .badge-yellow { background: #4a3500; color: #f0a93a; }
    .badge-red    { background: #4a0c0c; color: #f85149; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        color: #8b949e; padding: 8px 18px; font-size: 0.85rem; font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1f3a5f, #1e3a8a) !important;
        border-color: #3b82f6 !important; color: #93c5fd !important;
    }

    /* ── Warning pill ── */
    .warn-pill {
        background: #3d2800; border: 1px solid #7a4a00; border-radius: 8px;
        padding: 6px 12px; font-size: 0.8rem; color: #fbbf24;
        margin: 4px 0; display: flex; align-items: center; gap: 6px;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Plotly theme helper
# ─────────────────────────────────────────────────────────────────────────────

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#c9d1d9", size=12),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor="#30363d", borderwidth=1),
    hoverlabel=dict(bgcolor="#1c2128", bordercolor="#30363d", font=dict(color="#e6edf3")),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
)


def _apply_theme(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=14, color="#c9d1d9")), height=height, **_PLOTLY_LAYOUT)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Cached helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_load_cycles(cell_id: str, data_dir: str) -> list[CycleRecord]:
    return load_discharge_cycles(cell_id=cell_id, data_dir=data_dir)


@st.cache_resource(show_spinner=False)
def cached_fit_models(cell_id: str, data_dir: str) -> _FitResult:
    """Heavy step: load cycles + fit GBR. Keyed on cell only – never reruns on EOL change."""
    cycles = load_discharge_cycles(cell_id=cell_id, data_dir=data_dir)
    return fit_twin_models(cycles)


def clear_model_caches() -> None:
    cached_load_cycles.clear()
    cached_fit_models.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Plotly chart builders
# ─────────────────────────────────────────────────────────────────────────────

def plotly_capacity_fade(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["cycle"], y=df["real_capacity_ah"],
        name="Measured",
        line=dict(color="#3b82f6", width=2.5),
        hovertemplate="Cycle %{x}<br>Capacity: %{y:.3f} Ah<extra>Measured</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["cycle"], y=df["twin_capacity_ah"],
        name="Digital Twin",
        line=dict(color="#f97316", width=2.5, dash="dash"),
        hovertemplate="Cycle %{x}<br>Twin: %{y:.3f} Ah<extra>Digital Twin</extra>",
    ))
    _apply_theme(fig, "⚡ Capacity Fade — Measured vs Digital Twin", height=360)
    fig.update_layout(yaxis_title="Capacity (Ah)", xaxis_title="Cycle")
    return fig


def plotly_soh_resistance(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df["cycle"], y=df["real_soh"] * 100,
        name="SOH (%)",
        line=dict(color="#22c55e", width=2.5),
        hovertemplate="Cycle %{x}<br>SOH: %{y:.2f} %<extra>SOH</extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df["cycle"], y=df["internal_resistance_ohm"],
        name="Internal Resistance (Ω)",
        line=dict(color="#ef4444", width=2.0, dash="dot"),
        hovertemplate="Cycle %{x}<br>R₀: %{y:.5f} Ω<extra>R₀</extra>",
    ), secondary_y=True)
    _apply_theme(fig, "🏥 Health Indicators", height=360)
    fig.update_yaxes(title_text="SOH (%)", secondary_y=False, gridcolor="#21262d")
    fig.update_yaxes(title_text="R₀ (Ω)", secondary_y=True, gridcolor="#21262d")
    fig.update_layout(xaxis_title="Cycle")
    return fig


def plotly_anomaly(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["cycle"], y=df["capacity_deviation_ah"],
        name="Deviation (Measured − Twin)",
        line=dict(color="#6366f1", width=1.8),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.08)",
        hovertemplate="Cycle %{x}<br>Δ Capacity: %{y:.4f} Ah<extra>Deviation</extra>",
    ))
    anomaly_df = df[df["is_anomaly"]]
    if not anomaly_df.empty:
        fig.add_trace(go.Scatter(
            x=anomaly_df["cycle"], y=anomaly_df["capacity_deviation_ah"],
            mode="markers",
            name="Anomaly",
            marker=dict(color="#ef4444", size=9, symbol="x", line=dict(width=2, color="#dc2626")),
            hovertemplate="Cycle %{x}<br>Anomaly: %{y:.4f} Ah<extra>⚠ Anomaly</extra>",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="#4b5563", line_width=1)
    _apply_theme(fig, "🔍 Anomaly Detection — Capacity Residuals", height=300)
    fig.update_layout(xaxis_title="Cycle", yaxis_title="Capacity Deviation (Ah)")
    return fig


def plotly_rul(snapshot: dict[str, float]) -> go.Figure:
    current_cycle = snapshot["cycle_number"]
    mean_eol      = current_cycle + snapshot["rul_mean_cycles"]
    low_eol       = current_cycle + snapshot["rul_ci_lower_90"]
    high_eol      = current_cycle + snapshot["rul_ci_upper_90"]

    fig = go.Figure()
    # CI band
    fig.add_shape(type="rect",
        x0=min(low_eol, high_eol), x1=max(low_eol, high_eol), y0=0, y1=1,
        fillcolor="rgba(99,102,241,0.18)", line_width=0)
    # Current cycle line
    fig.add_vline(x=current_cycle, line_dash="dot", line_color="#c9d1d9", line_width=1.5,
                  annotation_text="Now", annotation_font_color="#c9d1d9")
    # Mean EOL
    fig.add_vline(x=mean_eol, line_color="#818cf8", line_width=2.5,
                  annotation_text=f"EOL ~{mean_eol:.0f}", annotation_font_color="#818cf8")
    # Build a layout dict without the generic 'yaxis' key from _PLOTLY_LAYOUT
    # to avoid duplicate keyword arguments.
    rul_layout = {k: v for k, v in _PLOTLY_LAYOUT.items() if k != "yaxis"}
    fig.update_layout(
        title=dict(text="🔋 RUL Prediction (90% CI band)", font=dict(size=14, color="#c9d1d9")),
        height=180,
        yaxis=dict(visible=False, range=[0, 1], gridcolor="#21262d", linecolor="#30363d"),
        xaxis_title="Cycle Number",
        **rul_layout,
    )
    return fig


def plotly_signals(cycle: CycleRecord, r0: float = 0.0, r1: float = 0.0, r2: float = 0.0) -> go.Figure:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         subplot_titles=["Voltage (V)", "Current (A)", "Temperature (°C)", "ECM Resistances (Ω)"])
    t = cycle.time_s
    fig.add_trace(go.Scatter(x=t, y=cycle.voltage_v, name="Voltage",
                              line=dict(color="#3b82f6", width=1.5),
                              hovertemplate="t=%{x:.1f}s  V=%{y:.4f} V<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=cycle.current_a, name="Current",
                              line=dict(color="#14b8a6", width=1.5),
                              hovertemplate="t=%{x:.1f}s  I=%{y:.4f} A<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=cycle.temperature_c, name="Temperature",
                              line=dict(color="#f97316", width=1.5),
                              hovertemplate="t=%{x:.1f}s  T=%{y:.2f} °C<extra></extra>"), row=3, col=1)
    
    # Plot scalar resistances as horizontal lines stretching across the cycle time
    fig.add_trace(go.Scatter(x=[t[0], t[-1]], y=[r0, r0], name="R₀ (Ohmic)",
                              line=dict(color="#ef4444", width=2.0),
                              hovertemplate="R₀ = %{y:.5f} Ω<extra></extra>"), row=4, col=1)
    fig.add_trace(go.Scatter(x=[t[0], t[-1]], y=[r1, r1], name="R₁ (SEI)",
                              line=dict(color="#d97706", width=2.0, dash="dash"),
                              hovertemplate="R₁ = %{y:.5f} Ω<extra></extra>"), row=4, col=1)
    fig.add_trace(go.Scatter(x=[t[0], t[-1]], y=[r2, r2], name="R₂ (Transfer)",
                              line=dict(color="#8b5cf6", width=2.0, dash="dot"),
                              hovertemplate="R₂ = %{y:.5f} Ω<extra></extra>"), row=4, col=1)

    _apply_theme(fig, f"📈 Cycle {cycle.cycle_number} — Raw Signals & ECM", height=550)
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    for i in [1, 2, 3, 4]:
        fig.update_xaxes(gridcolor="#21262d", row=i, col=1)
        fig.update_yaxes(gridcolor="#21262d", row=i, col=1)
    return fig


def plotly_what_if(sim_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         subplot_titles=["Capacity (Ah)", "SOH", "Internal Resistance (Ω)"])
    fig.add_trace(go.Scatter(x=sim_df["cycle"], y=sim_df["predicted_capacity_ah"],
                              name="Capacity", line=dict(color="#3b82f6", width=2.2),
                              hovertemplate="Cycle %{x}<br>Cap: %{y:.3f} Ah<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sim_df["cycle"], y=sim_df["predicted_soh"] * 100,
                              name="SOH", line=dict(color="#22c55e", width=2.2),
                              hovertemplate="Cycle %{x}<br>SOH: %{y:.2f} %<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=sim_df["cycle"], y=sim_df["predicted_internal_resistance_ohm"],
                              name="R₀", line=dict(color="#ef4444", width=2.2),
                              hovertemplate="Cycle %{x}<br>R₀: %{y:.5f} Ω<extra></extra>"), row=3, col=1)
    _apply_theme(fig, "🔮 What-If Simulation", height=480)
    fig.update_xaxes(title_text="Future Cycle", row=3, col=1)
    for i in [1, 2, 3]:
        fig.update_xaxes(gridcolor="#21262d", row=i, col=1)
        fig.update_yaxes(gridcolor="#21262d", row=i, col=1)
    return fig


def plotly_temperature_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=df["mean_temperature_c"],
        nbinsx=25,
        marker=dict(color="#6366f1", opacity=0.8, line=dict(color="#818cf8", width=0.5)),
        hovertemplate="Temp: %{x:.1f} °C<br>Count: %{y}<extra></extra>",
    ))
    _apply_theme(fig, "🌡 Temperature Distribution", height=260)
    fig.update_layout(xaxis_title="Mean Temperature (°C)", yaxis_title="Cycles")
    return fig


def plotly_current_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=df["mean_abs_current_a"],
        nbinsx=25,
        marker=dict(color="#14b8a6", opacity=0.8, line=dict(color="#5eead4", width=0.5)),
        hovertemplate="Current: %{x:.3f} A<br>Count: %{y}<extra></extra>",
    ))
    _apply_theme(fig, "⚡ Discharge Current Distribution", height=260)
    fig.update_layout(xaxis_title="Mean Abs Current (A)", yaxis_title="Cycles")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# KPI card HTML
# ─────────────────────────────────────────────────────────────────────────────

def _kpi_html(label: str, value: str, delta: str, css_class: str) -> str:
    return f"""
    <div class="kpi-card {css_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-delta">{delta}</div>
    </div>
    """


def _health_bar(soh: float) -> str:
    pct = int(soh * 100)
    if pct >= 80:
        color = "linear-gradient(90deg, #22c55e, #16a34a)"
    elif pct >= 60:
        color = "linear-gradient(90deg, #f59e0b, #d97706)"
    else:
        color = "linear-gradient(90deg, #ef4444, #dc2626)"
    return f"""
    <div style="margin: 4px 0 12px;">
      <div style="font-size:0.72rem;color:#8b949e;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:4px;">
        Battery Health — {pct}%
      </div>
      <div class="health-bar-outer">
        <div class="health-bar-inner" style="width:{pct}%;background:{color};"></div>
      </div>
    </div>
    """


def _quality_badge(score: float) -> str:
    if score >= 80:
        cls, label = "badge-green", "Good"
    elif score >= 50:
        cls, label = "badge-yellow", "Fair"
    else:
        cls, label = "badge-red", "Poor"
    return f'<span class="badge {cls}">Data Quality: {label} ({score:.0f}/100)</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Formula popover helper
# ─────────────────────────────────────────────────────────────────────────────

_FORMULA_BTN_CSS = """
<style>
/* Formula popover trigger button */
[data-testid="stPopover"] > button {
    background: rgba(99,102,241,0.12) !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    border-radius: 8px !important;
    color: #818cf8 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    padding: 3px 10px !important;
    width: 100% !important;
    margin-top: 6px !important;
    letter-spacing: 0.05em !important;
    transition: background 0.2s ease, border-color 0.2s ease !important;
}
[data-testid="stPopover"] > button:hover {
    background: rgba(99,102,241,0.25) !important;
    border-color: #6366f1 !important;
    color: #a5b4fc !important;
}
/* Popover panel */
[data-testid="stPopoverBody"] {
    background: #1c2128 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    padding: 16px !important;
    min-width: 320px !important;
}
</style>
"""

_FORMULA_DEFS: dict[str, dict] = {
    "soh": {
        "title": "State of Health (SOH)",
        "latex": [
            r"\text{SOH}(n) = \frac{Q_{\text{actual}}(n)}{Q_{\text{nominal}}}",
            r"Q_{\text{nominal}} = \widetilde{Q}_{\text{IQR}}\bigl(Q_1, \ldots, Q_{10\%}\bigr)",
        ],
        "vars": [
            ("Q_actual(n)",   "Discharge capacity measured at cycle n (Ah) via trapezoidal integration of |I(t)|"),
            ("Q_nominal",     "IQR-trimmed median of the first 10 % of discharge cycles — used as the pristine baseline"),
            ("SOH = 1.0",     "Fully healthy; SOH ≤ EOL threshold → end of life"),
        ],
    },
    "r0": {
        "title": "Internal Resistance (R₀)",
        "latex": [
            r"R_0^{\text{proxy}}(n) = 0.012 + 0.025 \times \bigl(1 - \text{SOH}(n)\bigr)^{1.2}",
            r"R_0^{\text{proxy}}(n) \;=\; \max\!\bigl(R_0^{\text{proxy}}(n-1),\; R_0^{\text{proxy}}(n)\bigr) \quad \text{(monotonic)}",
            r"\hat{R}_0(n) = a_0 n^2 + a_1 n + a_2 \quad \text{(IQR-filtered least-squares polyfit)}",
        ],
        "vars": [
            ("0.012 Ω",       "Nominal (fresh battery) internal resistance offset"),
            ("0.025",         "Empirical scaling coefficient for capacity-to-resistance mapping"),
            ("exponent 1.2",  "Sub-linear degradation exponent (fitted to NASA battery aging data)"),
            ("a₀, a₁, a₂",   "Polynomial coefficients from degree-2 least-squares fit on resistance proxy"),
        ],
    },
    "rul": {
        "title": "Remaining Useful Life (RUL)",
        "latex": [
            r"\hat{Q}(n) = \underbrace{c_0\bigl(1 - \beta \cdot n^{\,\alpha}\bigr)}_{\text{physics}} + \underbrace{\text{GBR}(\mathbf{x}_n)}_{\text{ML residual}}",
            r"\hat{n}_{\text{EOL}} = \min\bigl\{n : \hat{Q}(n) \leq Q_{\text{EOL}}\bigr\}",
            r"\text{RUL} = \hat{n}_{\text{EOL}} - n_{\text{current}}",
            r"\text{CI}_{90\%}: \hat{Q}(n) \pm 1.64\,\hat{\sigma}_{\text{residual}}",
        ],
        "vars": [
            ("c₀",            "Initial capacity parameter — multi-start curve_fit (4 initial guesses, TRF solver)"),
            ("β",             "Fade rate coefficient (bounds: [0, 0.5])"),
            ("α",             "Fade exponent, controls curvature (bounds: [0.15, 1.3])"),
            ("GBR",           "Gradient Boosting Regressor (200 trees, lr=0.05, depth=3) trained on 10-feature vector: [n, R₀, T, Q̂_physics, rolling stats, R₀ rate, thermal stress]"),
            ("σ_residual",    "Inflated train-residual std: σ × √(1 + 1/N) for honest out-of-sample uncertainty"),
        ],
    },
    "eol": {
        "title": "Predicted EOL Cycle",
        "latex": [
            r"n_{\text{EOL}} = n_{\text{current}} + \text{RUL}_{\text{mean}}",
            r"n_{\text{EOL}}^{\text{low}} = n_{\text{current}} + \text{RUL}_{\text{low}}, \quad n_{\text{EOL}}^{\text{high}} = n_{\text{current}} + \text{RUL}_{\text{high}}",
            r"\text{where } \text{RUL}_{\text{low/high}} = \hat{n}_{\text{EOL}}^{\text{pess/opt}} - n_{\text{current}}",
        ],
        "vars": [
            ("n_current",              "Current (latest observed) cycle number"),
            ("RUL_mean",              "Mean RUL from nominal fade prediction Q̂(n)"),
            ("n_EOL_pess / opt",      "Pessimistic EOL from Q̂ − 1.64σ; Optimistic from Q̂ + 1.64σ"),
            ("Q_EOL",                 "User-configurable EOL capacity threshold (Ah) — adjustable in sidebar"),
        ],
    },
    "r1": {
        "title": "SEI Resistance (R₁)",
        "latex": [
            r"R_1^{\text{proxy}}(n) = 0.008 + 0.015 \times \bigl(1 - \text{SOH}(n)\bigr)^{1.3}",
            r"R_1^{\text{proxy}}(n) \;=\; \max\!\bigl(R_1^{\text{proxy}}(n-1),\; R_1^{\text{proxy}}(n)\bigr) \quad \text{(monotonic)}",
        ],
        "vars": [
            ("Solid Electrolyte Interphase", "R1 models the resistive growth of the SEI layer mapping exactly to the RC pairs in the Dual-Polarization ECM model."),
            ("0.008 Ω", "Nominal healthy SEI baseline derived from DE bounds"),
            ("1.3 Exponent", "Sub-linear capacity-to-SEI decay mapping"),
        ],
    },
    "r2": {
        "title": "Transfer Resistance (R₂)",
        "latex": [
            r"R_2^{\text{proxy}}(n) = 0.015 + 0.030 \times \bigl(1 - \text{SOH}(n)\bigr)^{1.5}",
            r"R_2^{\text{proxy}}(n) \;=\; \max\!\bigl(R_2^{\text{proxy}}(n-1),\; R_2^{\text{proxy}}(n)\bigr) \quad \text{(monotonic)}",
        ],
        "vars": [
            ("Charge Transfer", "R2 corresponds to the ionic transfer rate limitations and double-layer capacitive boundaries."),
            ("0.015 Ω", "Baseline transfer resistance from offline mapping"),
            ("1.5 Exponent", "Super-linear exponent accounting for severe terminal deterioration of mass transfer."),
        ],
    },
}


def _kpi_with_formula(col, label: str, value: str, delta: str, css_class: str, formula_key: str) -> None:
    """Render a KPI card + a formula popover button inside the given Streamlit column."""
    with col:
        st.markdown(_kpi_html(label, value, delta, css_class), unsafe_allow_html=True)
        st.markdown(_FORMULA_BTN_CSS, unsafe_allow_html=True)
        defn = _FORMULA_DEFS[formula_key]
        with st.popover("ƒ  Formula", use_container_width=True):
            st.markdown(
                f'<div style="font-size:1rem;font-weight:700;color:#e6edf3;margin-bottom:10px;">'
                f'{defn["title"]}</div>',
                unsafe_allow_html=True,
            )
            for eq in defn["latex"]:
                st.latex(eq)
            st.markdown(
                '<div style="font-size:0.75rem;font-weight:700;color:#8b949e;'
                'text-transform:uppercase;letter-spacing:0.07em;margin:12px 0 6px;">'
                'Variable definitions</div>',
                unsafe_allow_html=True,
            )
            for var, desc in defn["vars"]:
                st.markdown(
                    f'<div style="font-size:0.8rem;color:#c9d1d9;margin-bottom:5px;">'
                    f'<code style="background:#2d333b;border-radius:4px;padding:1px 5px;'
                    f'color:#79c0ff;font-size:0.78rem;">{var}</code> '
                    f'<span style="color:#8b949e;">{desc}</span></div>',
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div style="font-size:1.3rem;font-weight:700;color:#e6edf3;margin-bottom:4px;">⚡ Battery Digital Twin</div>'
            '<div style="font-size:0.78rem;color:#8b949e;margin-bottom:20px;">Physics-informed ML monitoring</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        data_dir = st.text_input("📁 Data Directory", value="data/raw")
        eol_capacity_ah = st.number_input(
            "🔴 EOL Threshold (Ah)", min_value=0.5, max_value=2.0, value=1.4, step=0.05,
            help="Battery considered end-of-life below this capacity.",
        )

        cells = list_available_cells(data_dir)
        if not cells:
            st.error(f"No MAT files found in '{data_dir}'.")
            st.stop()

        cell_id = st.selectbox("🔋 Battery Cell", options=cells, index=min(3, len(cells) - 1))

        auto_refresh = st.toggle("🔄 Auto-refresh (30 s)", value=False)
        if auto_refresh:
            st.rerun() if st.session_state.get("_trig") else None

        st.divider()
        col_a, col_b = st.columns(2)
        if col_a.button("🔁 Retrain", use_container_width=True):
            clear_model_caches()
            st.rerun()
        if col_b.button("🗑 Clear Cache", use_container_width=True):
            clear_model_caches()
            st.info("Cache cleared.")

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        # Stage 1: heavy fit – cached; only reruns on Retrain or new cell
        with st.spinner("⚙ Loading cycles and fitting Digital Twin models…"):
            cycles = cached_load_cycles(cell_id=cell_id, data_dir=data_dir)
            fit    = cached_fit_models(cell_id=cell_id, data_dir=data_dir)

        # Stage 2: EOL post-processing – fast (< 200 ms), called every render
        # NOT cached: avoids shared-object mutation bugs and float hash collisions.
        artifacts = compute_eol_artifacts(fit, float(eol_capacity_ah))
    except Exception as exc:
        st.error(f"**Failed to initialise dashboard:** {exc}")
        st.stop()

    if not cycles:
        st.warning("No discharge cycles available.")
        st.stop()

    summary_df = artifacts.summary_df
    latest     = artifacts.snapshot
    metrics    = artifacts.training_metrics
    quality    = metrics.quality_report

    # Detect new data
    prev_count = st.session_state.get("cycle_count")
    st.session_state["cycle_count"] = len(cycles)
    if prev_count is not None and len(cycles) > int(prev_count):
        st.toast(f"🆕 {len(cycles) - int(prev_count)} new cycles loaded!", icon="⚡")

    # ── Header ────────────────────────────────────────────────────────────────
    header_col, badge_col = st.columns([3, 1])
    with header_col:
        st.markdown(
            f'<div style="font-size:1.6rem;font-weight:700;color:#e6edf3;margin-bottom:2px;">'
            f'⚡ Battery Digital Twin Dashboard</div>'
            f'<div style="font-size:0.82rem;color:#8b949e;">Cell: <b style="color:#c9d1d9">{cell_id}</b> &nbsp;|&nbsp; '
            f'{len(cycles)} discharge cycles &nbsp;|&nbsp; {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>',
            unsafe_allow_html=True,
        )
    with badge_col:
        st.markdown(
            f'<div style="margin-top:14px;text-align:right;">{_quality_badge(quality.quality_score)}</div>',
            unsafe_allow_html=True,
        )

    # ── Quality warnings ──────────────────────────────────────────────────────
    if quality.warnings:
        with st.expander(f"⚠ {len(quality.warnings)} data quality notice(s)", expanded=False):
            for w in quality.warnings:
                st.markdown(f'<div class="warn-pill">⚠ {w}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Health bar ────────────────────────────────────────────────────────────
    st.markdown(_health_bar(latest["soh"]), unsafe_allow_html=True)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    soh_pct       = latest["soh"] * 100
    r0_val        = latest["internal_resistance_ohm"]
    r1_val        = latest.get("sei_resistance_ohm", 0.0)
    r2_val        = latest.get("transfer_resistance_ohm", 0.0)
    rul_mean      = latest["rul_mean_cycles"]
    rul_lo        = latest["rul_ci_lower_90"]
    rul_hi        = latest["rul_ci_upper_90"]
    eol_cycle     = latest["predicted_eol_cycle_mean"]
    current_cycle = int(latest["cycle_number"])

    # Row 1: Health & Predictions
    k1, k2, k3 = st.columns(3)
    _kpi_with_formula(k1, "State of Health",       f"{soh_pct:.1f}%",        f"Cycle {current_cycle}",                    "kpi-soh",    "soh")
    _kpi_with_formula(k2, "Remaining Useful Life", f"{rul_mean:.0f} cyc",    f"90% CI: [{rul_lo:.0f}, {rul_hi:.0f}]",     "kpi-rul",    "rul")
    _kpi_with_formula(k3, "Predicted EOL Cycle",   f"{eol_cycle:.0f}",       f"Current: {current_cycle}",                 "kpi-eol",    "eol")

    st.markdown("<br/>", unsafe_allow_html=True)

    # Row 2: ECM Physics Parameters
    k4, k5, k6 = st.columns(3)
    _kpi_with_formula(k4, "Ohmic Resistance (R0)",    f"{r0_val*1000:.2f} mΩ", "Internal bulk resistance",                   "kpi-resist", "r0")
    _kpi_with_formula(k5, "SEI Resistance (R1)",      f"{r1_val*1000:.2f} mΩ", "Solid Electrolyte Interphase layer",         "kpi-resist", "r1")
    _kpi_with_formula(k6, "Transfer Resistance (R2)", f"{r2_val*1000:.2f} mΩ", "Charge transfer polarization",               "kpi-resist", "r2")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_fade, tab_health, tab_signals, tab_sim, tab_diagnostics, tab_export, tab_datalab = st.tabs([
        "🔋 Capacity Fade",
        "🏥 Health",
        "📈 Signals",
        "🔮 What-If",
        "🧪 ML Diagnostics",
        "📤 Export",
        "🔬 Data Lab",
    ])

    # ────────────────────────────────────────────────────────────
    # Tab 1 – Capacity Fade
    # ────────────────────────────────────────────────────────────
    with tab_fade:
        st.plotly_chart(plotly_capacity_fade(summary_df), use_container_width=True)

        st.markdown('<div class="section-header">🔍 Anomaly Detection</div>', unsafe_allow_html=True)
        st.plotly_chart(plotly_anomaly(summary_df), use_container_width=True)

        anomaly_rows = summary_df[summary_df["is_anomaly"]]
        if anomaly_rows.empty:
            st.info("✅ No anomalies detected in this cell's history.")
        else:
            st.warning(f"⚠ {len(anomaly_rows)} anomalous cycles detected.")
            st.dataframe(
                anomaly_rows[["cycle", "real_capacity_ah", "twin_capacity_ah", "capacity_deviation_ah"]]
                .rename(columns={
                    "cycle": "Cycle",
                    "real_capacity_ah": "Measured (Ah)",
                    "twin_capacity_ah": "Twin (Ah)",
                    "capacity_deviation_ah": "Deviation (Ah)",
                })
                .style.format("{:.4f}", subset=["Measured (Ah)", "Twin (Ah)", "Deviation (Ah)"]),
                use_container_width=True,
                hide_index=True,
            )

        # Distribution plots
        dist_c1, dist_c2 = st.columns(2)
        with dist_c1:
            st.plotly_chart(plotly_temperature_distribution(summary_df), use_container_width=True)
        with dist_c2:
            st.plotly_chart(plotly_current_distribution(summary_df), use_container_width=True)

    # ────────────────────────────────────────────────────────────
    # Tab 2 – Health
    # ────────────────────────────────────────────────────────────
    with tab_health:
        st.plotly_chart(plotly_soh_resistance(summary_df), use_container_width=True)

        rul_col, gap_col = st.columns([2, 1])
        with rul_col:
            st.plotly_chart(plotly_rul(latest), use_container_width=True)
        with gap_col:
            st.markdown("<br>", unsafe_allow_html=True)
            rul_fraction = rul_mean / max(rul_hi, 1)
            st.metric("📏 RUL Mean", f"{rul_mean:.0f} cycles")
            st.metric("📉 Capacity @ Latest Cycle", f"{latest['capacity_ah']:.4f} Ah")
            st.metric("⚗ EOL Threshold", f"{eol_capacity_ah:.2f} Ah")
            st.metric("🌡 Last Temp", f"{float(summary_df['mean_temperature_c'].iloc[-1]):.1f} °C")

    # ────────────────────────────────────────────────────────────
    # Tab 3 – Raw Signals
    # ────────────────────────────────────────────────────────────
    with tab_signals:
        st.markdown('<div class="section-header">📈 Voltage, Current, Temperature Curves</div>', unsafe_allow_html=True)
        cycle_nums = [c.cycle_number for c in cycles]
        selected_cycle_number = st.select_slider(
            "Select Cycle",
            options=cycle_nums,
            value=cycle_nums[-1],
            label_visibility="collapsed",
        )
        selected_cycle = next(c for c in cycles if c.cycle_number == selected_cycle_number)
        selected_record = summary_df[summary_df["cycle"] == selected_cycle_number]
        
        r0 = 0.0; r1 = 0.0; r2 = 0.0
        if not selected_record.empty:
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Capacity", f"{selected_cycle.capacity_ah:.4f} Ah")
            sc2.metric("Avg Temp", f"{float(selected_cycle.temperature_c.mean()):.1f} °C")
            sc3.metric("Avg |Current|", f"{float(np.abs(selected_cycle.current_a).mean()):.3f} A")
            sc4.metric("Duration", f"{selected_cycle.time_s[-1]:.0f} s")
            
            r0 = float(selected_record["internal_resistance_ohm"].iloc[0])
            r1 = float(selected_record["sei_resistance_ohm"].iloc[0]) if "sei_resistance_ohm" in selected_record.columns else 0.0
            r2 = float(selected_record["transfer_resistance_ohm"].iloc[0]) if "transfer_resistance_ohm" in selected_record.columns else 0.0

        st.plotly_chart(plotly_signals(selected_cycle, r0, r1, r2), use_container_width=True)

    # ────────────────────────────────────────────────────────────
    # Tab 4 – What-If Simulator
    # ────────────────────────────────────────────────────────────
    with tab_sim:
        st.markdown('<div class="section-header">🔮 What-If Future Scenario Simulator</div>', unsafe_allow_html=True)
        sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
        profile_name        = sim_c1.selectbox("Usage Profile", options=sorted(USAGE_PROFILES.keys()))
        horizon_cycles      = sim_c2.slider("Future Cycles", min_value=10, max_value=500, value=200, step=10)
        temperature_delta_c = sim_c3.slider("Temp Delta (°C)", min_value=-10.0, max_value=20.0, value=0.0, step=1.0)
        current_multiplier  = sim_c4.slider("Current Multiplier", min_value=0.5, max_value=2.0, value=1.0, step=0.05)

        sim_df = simulate_what_if(
            artifacts=artifacts,
            horizon_cycles=int(horizon_cycles),
            profile_name=str(profile_name),
            temperature_delta_c=float(temperature_delta_c),
            current_multiplier=float(current_multiplier),
            eol_capacity_ah=float(eol_capacity_ah),
        )
        if not sim_df.empty:
            st.plotly_chart(plotly_what_if(sim_df), use_container_width=True)
            fin = sim_df.iloc[-1]
            ms1, ms2, ms3 = st.columns(3)
            ms1.metric("Final Projected RUL", f"{fin['predicted_rul_cycles']:.0f} cyc")
            ms2.metric("Final Projected SOH",  f"{fin['predicted_soh']*100:.1f}%")
            ms3.metric("Final Projected R₀",   f"{fin['predicted_internal_resistance_ohm']*1000:.2f} mΩ")
        else:
            st.warning("No simulation data to display.")

    # ────────────────────────────────────────────────────────────
    # Tab 5 – ML Diagnostics
    # ────────────────────────────────────────────────────────────
    with tab_diagnostics:
        st.markdown('<div class="section-header">🧪 Training & Model Diagnostics</div>', unsafe_allow_html=True)

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Cycles",         f"{quality.total_cycles}")
        d2.metric("Filtered Cycles",      f"{quality.dropped_cycles}")
        d3.metric("Capacity Outliers",    f"{quality.capacity_outliers}")
        d4.metric("Anomalies Detected",   f"{metrics.n_anomalies}")

        st.markdown("---")
        md1, md2, md3 = st.columns(3)
        md1.metric("Physics Fit SSE",     f"{metrics.empirical_fit_sse:.4f}" if np.isfinite(metrics.empirical_fit_sse) else "N/A")
        md2.metric("ML CV-RMSE",          f"{metrics.ml_cv_rmse:.4f}" if np.isfinite(metrics.ml_cv_rmse) else "N/A (< 8 cycles)")
        md3.metric("Residual Std Used",   f"{metrics.combined_residual_std:.4f}")

        with st.expander("📋 Data Quality Details"):
            st.write(f"**Temperature range:** {quality.temperature_range[0]:.1f} – {quality.temperature_range[1]:.1f} °C")
            st.write(f"**Capacity range:** {quality.capacity_range[0]:.4f} – {quality.capacity_range[1]:.4f} Ah")
            st.write(f"**Quality score:** {quality.quality_score:.1f} / 100")
            if quality.warnings:
                st.markdown("**Warnings:**")
                for w in quality.warnings:
                    st.warning(w)
            else:
                st.success("No data quality issues detected.")

        # Residual scatter plot
        st.markdown('<div class="section-header">📊 Prediction Residuals</div>', unsafe_allow_html=True)
        residual_fig = go.Figure()
        residual_fig.add_trace(go.Scatter(
            x=summary_df["cycle"],
            y=summary_df["capacity_deviation_ah"],
            mode="markers+lines",
            marker=dict(color="#818cf8", size=5, opacity=0.7),
            line=dict(color="#6366f1", width=1),
            name="Residual (Measured − Twin)",
            hovertemplate="Cycle %{x}<br>Residual: %{y:.5f} Ah<extra></extra>",
        ))
        residual_fig.add_hline(y=0, line_dash="dash", line_color="#4b5563")
        # ±1σ band
        std_val = metrics.combined_residual_std
        residual_fig.add_hrect(y0=-std_val, y1=std_val,
                                fillcolor="rgba(99,102,241,0.07)", line_width=0,
                                annotation_text="±1σ", annotation_font_color="#818cf8")
        _apply_theme(residual_fig, "Capacity Residuals vs Cycle", height=300)
        residual_fig.update_layout(xaxis_title="Cycle", yaxis_title="Residual (Ah)")
        st.plotly_chart(residual_fig, use_container_width=True)

        # Capacity evolution (all models stacked)
        st.markdown('<div class="section-header">📉 Model Regression Summary</div>', unsafe_allow_html=True)
        reg_fig = go.Figure()
        reg_fig.add_trace(go.Scatter(
            x=summary_df["cycle"], y=summary_df["real_capacity_ah"],
            name="Measured", mode="markers",
            marker=dict(color="#3b82f6", size=4, opacity=0.6),
        ))
        reg_fig.add_trace(go.Scatter(
            x=summary_df["cycle"], y=summary_df["twin_capacity_ah"],
            name="Digital Twin", line=dict(color="#f97316", width=2.5, dash="dash"),
        ))
        # CI band
        upper = summary_df["twin_capacity_ah"] + 1.64 * std_val
        lower = summary_df["twin_capacity_ah"] - 1.64 * std_val
        reg_fig.add_trace(go.Scatter(
            x=pd.concat([summary_df["cycle"], summary_df["cycle"][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(249,115,22,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="90% CI Band",
            hoverinfo="skip",
        ))
        _apply_theme(reg_fig, "Regression Fit + 90% Uncertainty Band", height=340)
        reg_fig.update_layout(xaxis_title="Cycle", yaxis_title="Capacity (Ah)")
        st.plotly_chart(reg_fig, use_container_width=True)

    # ────────────────────────────────────────────────────────────
    # Tab 6 – Export
    # ────────────────────────────────────────────────────────────
    with tab_export:
        st.markdown('<div class="section-header">📤 Export Data & Reports</div>', unsafe_allow_html=True)

        report = {
            "cell_id":          cell_id,
            "generated_utc":    datetime.now(timezone.utc).isoformat(),
            "cycle_count":      len(cycles),
            "anomaly_count":    int(summary_df["is_anomaly"].sum()),
            "latest_snapshot":  latest,
            "data_quality": {
                "quality_score":     quality.quality_score,
                "total_cycles":      quality.total_cycles,
                "dropped_cycles":    quality.dropped_cycles,
                "capacity_outliers": quality.capacity_outliers,
                "temperature_range": list(quality.temperature_range),
                "capacity_range":    list(quality.capacity_range),
                "warnings":          quality.warnings,
            },
            "model_metrics": {
                "empirical_fit_sse":      metrics.empirical_fit_sse,
                "ml_cv_rmse":             metrics.ml_cv_rmse,
                "combined_residual_std":  metrics.combined_residual_std,
                "n_anomalies":            metrics.n_anomalies,
            },
            "settings": {
                "data_dir":         data_dir,
                "eol_capacity_ah":  float(eol_capacity_ah),
            },
        }

        ec1, ec2 = st.columns(2)
        ec1.download_button(
            label="⬇ Download Predictions (CSV)",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{cell_id.lower()}_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
        ec2.download_button(
            label="⬇ Download Report (JSON)",
            data=json.dumps(report, indent=2, default=str).encode("utf-8"),
            file_name=f"{cell_id.lower()}_report.json",
            mime="application/json",
            use_container_width=True,
        )

        with st.expander("🗂 Raw Report JSON"):
            st.json(report)


        with st.expander("📊 Full Predictions Table"):
            st.dataframe(summary_df.style.format({
                "real_capacity_ah":        "{:.4f}",
                "twin_capacity_ah":        "{:.4f}",
                "real_soh":                "{:.4f}",
                "twin_soh":                "{:.4f}",
                "internal_resistance_ohm": "{:.5f}",
                "mean_temperature_c":      "{:.2f}",
                "mean_abs_current_a":      "{:.3f}",
                "capacity_deviation_ah":   "{:.5f}",
            }), use_container_width=True, hide_index=True)

    # ────────────────────────────────────────────────────────────
    # Tab 7 – Data Lab
    # ────────────────────────────────────────────────────────────
    with tab_datalab:
        st.markdown('<div class="section-header">🔬 Data Remediation Lab</div>', unsafe_allow_html=True)
        st.write("Run the end-to-end data pipeline to detect anomalies, enforce physics-based constraints, and synthesize balanced data.")
        
        lab_col1, lab_col2 = st.columns([1, 2], gap="large")
        with lab_col1:
            st.markdown("### 🎛️ Control Panel")
            if st.button("🚀 Execute Remediation Pipeline", use_container_width=True, type="primary"):
                with st.spinner("Executing 5-step robust remediation..."):
                    from data.remediation_pipeline import run_remediation
                    run_remediation(
                        raw_dir=data_dir, 
                        eol_capacity_ah=float(eol_capacity_ah),
                        out_csv="data/remediated_dataset.csv", 
                        out_json="data/remediated_dataset.json"
                    )
                    st.session_state["remediation_done"] = True
                    st.success("✨ Remediation completed successfully!")
                    # Use st.rerun() if available, else omit to avoid errors
                    if hasattr(st, "rerun"):
                        st.rerun()
            
            st.markdown("<hr style='margin: 15px 0; border-color: #30363d;'/>", unsafe_allow_html=True)
            
            if Path("data/remediated_dataset.csv").exists():
                st.info("✅ **Dataset Ready**")
                
                # Dynamic Metrics
                try:
                    df_rem = pd.read_csv("data/remediated_dataset.csv")
                    n_orig = len(df_rem[df_rem.get("is_synthetic", False) == False])
                    n_synth = len(df_rem[df_rem.get("is_synthetic", False) == True])
                    
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Clean Rows", n_orig)
                    mc2.metric("Synthetic", n_synth, f"+{int(n_synth/max(1,n_orig)*100)}%")
                except Exception:
                    pass

                try:
                    with open("data/remediation_report.md", "r", encoding="utf-8") as f:
                        report_md = f.read()
                    st.download_button(
                        label="⬇ Download Audit Report (MD)",
                        data=report_md.encode("utf-8"),
                        file_name="remediation_report.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                except Exception:
                    pass
            else:
                st.warning("No remediated dataset found. Execute pipeline first.")
                
        with lab_col2:
            st.markdown("### 📊 Live Audit Report")
            if st.session_state.get("remediation_done") or Path("data/remediation_report.md").exists():
                try:
                    with open("data/remediation_report.md", "r", encoding="utf-8") as f:
                        report_text = f.read()
                    with st.container(border=True):
                        st.markdown(report_text)
                except FileNotFoundError:
                    st.info("Report file not generated yet.")
            else:
                st.info("👈 Please execute the pipeline to generate the audit report.")


if __name__ == "__main__":
    main()
