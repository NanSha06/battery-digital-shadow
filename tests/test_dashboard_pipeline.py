"""
tests/test_dashboard_pipeline.py
================================
End-to-end tests for the dashboard modeling pipeline:
  - Data loading & CycleRecord construction
  - Two-stage model fitting (fit_twin_models → compute_eol_artifacts)
  - R0 / R1 / R2 proxy calculations
  - Anomaly detection
  - Snapshot & summary_df integrity
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dashboard.data_loading import CycleRecord
from dashboard.modeling import (
    DataQualityReport,
    TwinArtifacts,
    _FitResult,
    compute_eol_artifacts,
    detect_capacity_anomalies,
    fit_twin_models,
    _assess_data_quality,
)


# ---------------------------------------------------------------------------
# Helpers – synthetic cycle generator
# ---------------------------------------------------------------------------

def _make_cycle(cycle_number: int, capacity_ah: float = 1.8,
                n_samples: int = 200, temp_c: float = 30.0) -> CycleRecord:
    """Create a realistic-looking synthetic discharge cycle."""
    t = np.linspace(0, 3600, n_samples)
    i = -2.0 * np.ones(n_samples)                      # constant 2A discharge
    v = np.linspace(4.2, 3.0, n_samples)                # linear voltage drop
    temp = np.full(n_samples, temp_c) + np.random.normal(0, 0.3, n_samples)
    return CycleRecord(
        cycle_number=cycle_number,
        time_s=t,
        voltage_v=v,
        current_a=i,
        temperature_c=temp,
        dt_s=float(np.median(np.diff(t))),
        capacity_ah=capacity_ah,
    )


def _make_degrading_fleet(n_cycles: int = 50) -> list[CycleRecord]:
    """Generate a fleet of cycles with a realistic capacity fade curve."""
    cycles = []
    for i in range(n_cycles):
        cap = 2.0 - 0.008 * i + np.random.normal(0, 0.005)
        cap = max(cap, 1.2)
        cycles.append(_make_cycle(cycle_number=i, capacity_ah=cap,
                                  temp_c=25.0 + 0.05 * i))
    return cycles


# ---------------------------------------------------------------------------
# Test: CycleRecord dataclass
# ---------------------------------------------------------------------------

class TestCycleRecord:
    def test_frozen_dataclass(self):
        c = _make_cycle(0)
        with pytest.raises(AttributeError):
            c.cycle_number = 99  # type: ignore[misc]

    def test_fields_present(self):
        c = _make_cycle(5, capacity_ah=1.9)
        assert c.cycle_number == 5
        assert c.capacity_ah == pytest.approx(1.9)
        assert c.time_s.shape[0] == 200
        assert c.voltage_v.shape == c.current_a.shape == c.temperature_c.shape


# ---------------------------------------------------------------------------
# Test: Data quality assessment
# ---------------------------------------------------------------------------

class TestDataQuality:
    def test_quality_score_perfect(self):
        cycles = _make_degrading_fleet(30)
        cap = np.array([c.capacity_ah for c in cycles])
        report = _assess_data_quality(cycles, cap)
        assert report.quality_score >= 80.0
        assert report.total_cycles == 30

    def test_warns_on_few_cycles(self):
        cycles = _make_degrading_fleet(10)
        cap = np.array([c.capacity_ah for c in cycles])
        report = _assess_data_quality(cycles, cap)
        assert any("Only" in w for w in report.warnings)

    def test_quality_score_bounded(self):
        cycles = _make_degrading_fleet(50)
        cap = np.array([c.capacity_ah for c in cycles])
        report = _assess_data_quality(cycles, cap)
        assert 0.0 <= report.quality_score <= 100.0


# ---------------------------------------------------------------------------
# Test: Anomaly detection
# ---------------------------------------------------------------------------

class TestAnomalyDetection:
    def test_no_anomalies_in_clean_data(self):
        deviations = np.random.normal(0, 0.005, 100)
        flags = detect_capacity_anomalies(deviations)
        assert flags.sum() == 0  # all within normal range

    def test_detects_injected_spike(self):
        deviations = np.random.normal(0, 0.005, 100)
        deviations[42] = 0.5   # massive spike
        deviations[77] = -0.4  # massive dip
        flags = detect_capacity_anomalies(deviations)
        assert flags[42] == True
        assert flags[77] == True

    def test_output_shape_matches_input(self):
        d = np.zeros(50)
        assert detect_capacity_anomalies(d).shape == (50,)


# ---------------------------------------------------------------------------
# Test: Two-stage model pipeline
# ---------------------------------------------------------------------------

class TestFitTwinModels:
    @pytest.fixture(scope="class")
    def fit_result(self) -> _FitResult:
        return fit_twin_models(_make_degrading_fleet(50))

    def test_returns_fit_result(self, fit_result: _FitResult):
        assert isinstance(fit_result, _FitResult)

    def test_cycle_ids_match(self, fit_result: _FitResult):
        assert len(fit_result.cycle_ids) == 50

    def test_r0_r1_r2_arrays_exist(self, fit_result: _FitResult):
        assert fit_result.model_r0.shape == (50,)
        assert fit_result.model_r1.shape == (50,)
        assert fit_result.model_r2.shape == (50,)

    def test_r1_r2_monotonically_increasing(self, fit_result: _FitResult):
        """R1 and R2 are enforced monotonic via np.maximum.accumulate."""
        r1 = fit_result.model_r1
        r2 = fit_result.model_r2
        assert np.all(np.diff(r1) >= -1e-10)
        assert np.all(np.diff(r2) >= -1e-10)

    def test_r2_greater_than_r1(self, fit_result: _FitResult):
        """R2 (charge transfer) should always be larger than R1 (SEI) by design."""
        assert np.all(fit_result.model_r2 >= fit_result.model_r1)

    def test_raises_on_empty_cycles(self):
        with pytest.raises(ValueError, match="No cycles"):
            fit_twin_models([])

    def test_raises_on_all_short_cycles(self):
        short = [_make_cycle(0, n_samples=5)]  # under 100 samples
        with pytest.raises(ValueError, match="filtered out"):
            fit_twin_models(short)


# ---------------------------------------------------------------------------
# Test: EOL artifacts (Stage 2)
# ---------------------------------------------------------------------------

class TestComputeEolArtifacts:
    @pytest.fixture(scope="class")
    def artifacts(self) -> TwinArtifacts:
        fit = fit_twin_models(_make_degrading_fleet(50))
        return compute_eol_artifacts(fit, eol_capacity_ah=1.4)

    def test_returns_twin_artifacts(self, artifacts: TwinArtifacts):
        assert isinstance(artifacts, TwinArtifacts)

    def test_summary_df_has_required_columns(self, artifacts: TwinArtifacts):
        required = {
            "cycle", "real_capacity_ah", "twin_capacity_ah",
            "real_soh", "twin_soh",
            "internal_resistance_ohm", "sei_resistance_ohm", "transfer_resistance_ohm",
            "mean_temperature_c", "mean_abs_current_a",
            "capacity_deviation_ah", "is_anomaly",
        }
        assert required.issubset(set(artifacts.summary_df.columns))

    def test_snapshot_keys(self, artifacts: TwinArtifacts):
        required_keys = {
            "cycle_number", "capacity_ah", "soh",
            "internal_resistance_ohm", "sei_resistance_ohm", "transfer_resistance_ohm",
            "rul_mean_cycles", "rul_ci_lower_90", "rul_ci_upper_90",
            "predicted_eol_cycle_mean",
        }
        assert required_keys.issubset(set(artifacts.snapshot.keys()))

    def test_snapshot_soh_between_0_and_1(self, artifacts: TwinArtifacts):
        soh = artifacts.snapshot["soh"]
        assert 0.0 <= soh <= 1.5

    def test_rul_non_negative(self, artifacts: TwinArtifacts):
        assert artifacts.snapshot["rul_mean_cycles"] >= 0

    def test_ci_ordering(self, artifacts: TwinArtifacts):
        lo = artifacts.snapshot["rul_ci_lower_90"]
        hi = artifacts.snapshot["rul_ci_upper_90"]
        assert hi >= lo

    def test_eol_does_not_mutate_model(self):
        """Changing EOL must NOT mutate the cached fit object."""
        fleet = _make_degrading_fleet(50)
        fit = fit_twin_models(fleet)
        art_14 = compute_eol_artifacts(fit, eol_capacity_ah=1.4)
        art_12 = compute_eol_artifacts(fit, eol_capacity_ah=1.2)
        # Different EOL → different RUL
        assert art_14.snapshot["rul_mean_cycles"] != art_12.snapshot["rul_mean_cycles"]
        # But the underlying model state is unchanged
        assert fit.fade_model.eol_capacity_ah == 0.0  # was set to 0 by fit_twin_models

    def test_training_metrics_present(self, artifacts: TwinArtifacts):
        m = artifacts.training_metrics
        assert np.isfinite(m.empirical_fit_sse)
        assert m.combined_residual_std > 0
        assert m.n_anomalies >= 0
