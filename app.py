from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.data_loading import CycleRecord, list_available_cells, load_discharge_cycles
from dashboard.modeling import TwinArtifacts, train_twin_model
from dashboard.simulation import USAGE_PROFILES, simulate_what_if
from dashboard.visualization import (
    plot_anomalies,
    plot_capacity_fade,
    plot_health_indicators,
    plot_rul,
    plot_signals,
    plot_what_if,
)


@st.cache_data(show_spinner=False)
def cached_load_cycles(cell_id: str, data_dir: str) -> list[CycleRecord]:
    return load_discharge_cycles(cell_id=cell_id, data_dir=data_dir)


@st.cache_resource(show_spinner=False)
def cached_train(cell_id: str, data_dir: str, eol_capacity_ah: float) -> TwinArtifacts:
    cycles = load_discharge_cycles(cell_id=cell_id, data_dir=data_dir)
    return train_twin_model(cycles=cycles, eol_capacity_ah=eol_capacity_ah)


def clear_model_caches() -> None:
    cached_load_cycles.clear()
    cached_train.clear()


def main() -> None:
    st.set_page_config(page_title="Battery Digital Twin", layout="wide")
    st.title("Battery Digital Twin Dashboard")
    st.caption("Capacity fade, cycle signals, RUL, health indicators, anomalies, and what-if simulation.")

    with st.sidebar:
        st.header("Settings")
        data_dir = st.text_input("Data Directory", value="data/raw")
        eol_capacity_ah = st.number_input("EOL Capacity Threshold (Ah)", min_value=0.5, max_value=2.0, value=1.4, step=0.05)

        cells = list_available_cells(data_dir)
        if not cells:
            st.error(f"No MAT files found in '{data_dir}'.")
            st.stop()

        cell_id = st.selectbox("Battery Cell", options=cells, index=min(3, len(cells) - 1))
        if st.button("Retrain Model"):
            clear_model_caches()
            st.rerun()

    try:
        cycles = cached_load_cycles(cell_id=cell_id, data_dir=data_dir)
        artifacts = cached_train(cell_id=cell_id, data_dir=data_dir, eol_capacity_ah=float(eol_capacity_ah))
    except Exception as exc:
        st.error(f"Failed to initialize dashboard data: {exc}")
        st.stop()

    if not cycles:
        st.warning("No discharge cycles available.")
        st.stop()

    summary_df = artifacts.summary_df
    latest = artifacts.snapshot

    previous_count = st.session_state.get("cycle_count")
    st.session_state["cycle_count"] = len(cycles)
    if previous_count is not None and len(cycles) > int(previous_count):
        st.success(f"New data detected: {len(cycles) - int(previous_count)} additional cycles loaded.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest SOH", f"{latest['soh']:.3f}")
    col2.metric("Internal Resistance", f"{latest['internal_resistance_ohm']:.4f} Ω")
    col3.metric(
        "RUL (Mean)",
        f"{latest['rul_mean_cycles']:.1f} cycles",
        delta=f"90% CI: [{latest['rul_ci_lower_90']:.1f}, {latest['rul_ci_upper_90']:.1f}]",
    )
    col4.metric("Predicted EOL Cycle", f"{latest['predicted_eol_cycle_mean']:.1f}")

    st.pyplot(plot_capacity_fade(summary_df), clear_figure=True)

    st.subheader("Voltage, Current, Temperature Curves")
    selected_cycle_number = st.select_slider(
        "Cycle",
        options=[cycle.cycle_number for cycle in cycles],
        value=cycles[-1].cycle_number,
    )
    selected_cycle = next(cycle for cycle in cycles if cycle.cycle_number == selected_cycle_number)
    st.pyplot(plot_signals(selected_cycle), clear_figure=True)

    rul_col, health_col = st.columns(2)
    with rul_col:
        st.pyplot(plot_rul(latest), clear_figure=True)
    with health_col:
        st.pyplot(plot_health_indicators(summary_df), clear_figure=True)

    st.subheader("Anomaly Detection")
    st.pyplot(plot_anomalies(summary_df), clear_figure=True)
    anomaly_rows = summary_df[summary_df["is_anomaly"]]
    if anomaly_rows.empty:
        st.info("No anomalies detected.")
    else:
        st.dataframe(anomaly_rows[["cycle", "real_capacity_ah", "twin_capacity_ah", "capacity_deviation_ah"]], use_container_width=True)

    st.subheader("What-If Simulator")
    sim_col_1, sim_col_2, sim_col_3, sim_col_4 = st.columns(4)
    profile_name = sim_col_1.selectbox("Usage Profile", options=sorted(USAGE_PROFILES.keys()))
    horizon_cycles = sim_col_2.slider("Future Cycles", min_value=10, max_value=500, value=200, step=10)
    temperature_delta_c = sim_col_3.slider("Temperature Delta (°C)", min_value=-10.0, max_value=20.0, value=0.0, step=1.0)
    current_multiplier = sim_col_4.slider("Current Multiplier", min_value=0.5, max_value=2.0, value=1.0, step=0.05)

    simulation_df = simulate_what_if(
        artifacts=artifacts,
        horizon_cycles=int(horizon_cycles),
        profile_name=str(profile_name),
        temperature_delta_c=float(temperature_delta_c),
        current_multiplier=float(current_multiplier),
        eol_capacity_ah=float(eol_capacity_ah),
    )
    if not simulation_df.empty:
        st.pyplot(plot_what_if(simulation_df), clear_figure=True)
        st.caption(
            "Projected final RUL: "
            f"{simulation_df['predicted_rul_cycles'].iloc[-1]:.1f} cycles | "
            f"Profile: {simulation_df['profile'].iloc[-1]}"
        )

    st.subheader("Export")
    report = {
        "cell_id": cell_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cycle_count": len(cycles),
        "anomaly_count": int(summary_df["is_anomaly"].sum()),
        "latest_snapshot": latest,
        "settings": {
            "data_dir": data_dir,
            "eol_capacity_ah": float(eol_capacity_ah),
        },
    }

    st.download_button(
        label="Download Predictions (CSV)",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{cell_id.lower()}_predictions.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download Report (JSON)",
        data=json.dumps(report, indent=2).encode("utf-8"),
        file_name=f"{cell_id.lower()}_report.json",
        mime="application/json",
    )

    with st.expander("Raw Report JSON"):
        st.json(report)


if __name__ == "__main__":
    main()

