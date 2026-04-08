from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dashboard.data_loading import CycleRecord


def plot_capacity_fade(summary_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(summary_df["cycle"], summary_df["real_capacity_ah"], label="Real", linewidth=2.0, color="#2563EB")
    ax.plot(
        summary_df["cycle"],
        summary_df["twin_capacity_ah"],
        label="Digital Twin",
        linewidth=2.0,
        linestyle="--",
        color="#F97316",
    )
    ax.set_title("Real vs Digital Twin Capacity Fade")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity (Ah)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_signals(cycle: CycleRecord) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(cycle.time_s, cycle.voltage_v, color="#1D4ED8", linewidth=1.3)
    axes[0].set_ylabel("Voltage (V)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(cycle.time_s, cycle.current_a, color="#0D9488", linewidth=1.3)
    axes[1].set_ylabel("Current (A)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(cycle.time_s, cycle.temperature_c, color="#DC2626", linewidth=1.3)
    axes[2].set_ylabel("Temp (°C)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Voltage, Current, Temperature | Cycle {cycle.cycle_number}")
    fig.tight_layout()
    return fig


def plot_rul(snapshot: dict[str, float]) -> plt.Figure:
    current_cycle = snapshot["cycle_number"]
    mean_eol = current_cycle + snapshot["rul_mean_cycles"]
    low_eol = current_cycle + snapshot["rul_ci_lower_90"]
    high_eol = current_cycle + snapshot["rul_ci_upper_90"]

    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.axvline(current_cycle, color="#111827", linestyle=":", linewidth=1.4, label="Current")
    ax.axvline(mean_eol, color="#4F46E5", linewidth=2.0, label="Mean EOL")
    ax.fill_betweenx([0.0, 1.0], min(low_eol, high_eol), max(low_eol, high_eol), alpha=0.28, color="#818CF8", label="90% CI")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Cycle")
    ax.set_title("RUL Prediction")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_health_indicators(summary_df: pd.DataFrame) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    l1 = ax1.plot(summary_df["cycle"], summary_df["real_soh"], color="#059669", linewidth=2.0, label="SOH")[0]
    l2 = ax2.plot(
        summary_df["cycle"],
        summary_df["internal_resistance_ohm"],
        color="#B91C1C",
        linewidth=1.8,
        linestyle="--",
        label="Internal Resistance",
    )[0]
    ax1.set_xlabel("Cycle")
    ax1.set_ylabel("SOH")
    ax2.set_ylabel("R0 (Ohm)")
    ax1.set_title("Health Indicators")
    ax1.grid(alpha=0.3)
    ax1.legend([l1, l2], ["SOH", "Internal Resistance"], loc="best")
    fig.tight_layout()
    return fig


def plot_anomalies(summary_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(summary_df["cycle"], summary_df["capacity_deviation_ah"], color="#374151", linewidth=1.7, label="Deviation (Real - Twin)")
    anomalies = summary_df[summary_df["is_anomaly"]]
    if not anomalies.empty:
        ax.scatter(anomalies["cycle"], anomalies["capacity_deviation_ah"], color="#DC2626", s=28, label="Anomaly")
    ax.axhline(0.0, color="#6B7280", linestyle="--", linewidth=1.0)
    ax.set_title("Anomaly Detection")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity Deviation (Ah)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_what_if(simulation_df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(simulation_df["cycle"], simulation_df["predicted_capacity_ah"], color="#2563EB", linewidth=2.0)
    axes[0].set_ylabel("Capacity (Ah)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(simulation_df["cycle"], simulation_df["predicted_soh"], color="#059669", linewidth=2.0)
    axes[1].set_ylabel("SOH")
    axes[1].grid(alpha=0.3)

    axes[2].plot(
        simulation_df["cycle"],
        simulation_df["predicted_internal_resistance_ohm"],
        color="#B91C1C",
        linewidth=2.0,
    )
    axes[2].set_ylabel("R0 (Ohm)")
    axes[2].set_xlabel("Future Cycle")
    axes[2].grid(alpha=0.3)

    fig.suptitle("What-If Simulation")
    fig.tight_layout()
    return fig

