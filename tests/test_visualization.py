"""
tests/test_visualization.py
============================
Tests for dashboard/visualization.py chart generators
and the inline Plotly charting functions from app.py.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless backend — no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dashboard.data_loading import CycleRecord
from dashboard.visualization import (
    plot_anomalies,
    plot_capacity_fade,
    plot_health_indicators,
    plot_rul,
    plot_signals,
    plot_what_if,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_summary_df() -> pd.DataFrame:
    n = 30
    cycles = np.arange(n)
    return pd.DataFrame({
        "cycle":                   cycles,
        "real_capacity_ah":        2.0 - 0.01 * cycles,
        "twin_capacity_ah":        2.0 - 0.0095 * cycles,
        "real_soh":                (2.0 - 0.01 * cycles) / 2.0,
        "twin_soh":                (2.0 - 0.0095 * cycles) / 2.0,
        "internal_resistance_ohm": 0.012 + 0.0003 * cycles,
        "sei_resistance_ohm":      0.008 + 0.0002 * cycles,
        "transfer_resistance_ohm": 0.015 + 0.0004 * cycles,
        "mean_temperature_c":      25.0 + 0.1 * cycles,
        "mean_abs_current_a":      np.full(n, 1.96),
        "capacity_deviation_ah":   np.random.normal(0, 0.003, n),
        "is_anomaly":              np.zeros(n, dtype=bool),
    })


@pytest.fixture
def sample_cycle() -> CycleRecord:
    n = 200
    return CycleRecord(
        cycle_number=10,
        time_s=np.linspace(0, 3600, n),
        voltage_v=np.linspace(4.2, 3.0, n),
        current_a=-2.0 * np.ones(n),
        temperature_c=25 + 5 * np.sin(np.linspace(0, 2 * np.pi, n)),
        dt_s=18.0,
        capacity_ah=1.85,
    )


@pytest.fixture
def sample_snapshot() -> dict:
    return {
        "cycle_number": 150.0,
        "rul_mean_cycles": 200.0,
        "rul_ci_lower_90": 150.0,
        "rul_ci_upper_90": 280.0,
    }


@pytest.fixture
def sample_sim_df() -> pd.DataFrame:
    n = 50
    cycles = np.arange(150, 150 + n)
    return pd.DataFrame({
        "cycle":                                cycles,
        "predicted_capacity_ah":                1.8 - 0.005 * np.arange(n),
        "predicted_soh":                        (1.8 - 0.005 * np.arange(n)) / 2.0,
        "predicted_internal_resistance_ohm":    0.02 + 0.0002 * np.arange(n),
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPlotCapacityFade:
    def test_returns_figure(self, sample_summary_df):
        fig = plot_capacity_fade(sample_summary_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_axes(self, sample_summary_df):
        fig = plot_capacity_fade(sample_summary_df)
        assert len(fig.axes) >= 1
        plt.close(fig)


class TestPlotHealthIndicators:
    def test_returns_figure(self, sample_summary_df):
        fig = plot_health_indicators(sample_summary_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_twin_axes(self, sample_summary_df):
        fig = plot_health_indicators(sample_summary_df)
        # Should have at least 2 axes (left + right y-axis)
        # Twinx creates a second axes sharing the x-axis
        assert len(fig.axes) >= 2
        plt.close(fig)


class TestPlotSignals:
    def test_returns_figure(self, sample_cycle):
        fig = plot_signals(sample_cycle)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotRul:
    def test_returns_figure(self, sample_snapshot):
        fig = plot_rul(sample_snapshot)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotAnomalies:
    def test_returns_figure(self, sample_summary_df):
        fig = plot_anomalies(sample_summary_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_handles_no_anomalies(self, sample_summary_df):
        """Should not crash when no anomalies are present."""
        df = sample_summary_df.copy()
        df["is_anomaly"] = False
        fig = plot_anomalies(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_handles_all_anomalies(self, sample_summary_df):
        df = sample_summary_df.copy()
        df["is_anomaly"] = True
        fig = plot_anomalies(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotWhatIf:
    def test_returns_figure(self, sample_sim_df):
        fig = plot_what_if(sample_sim_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
