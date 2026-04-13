"""
tests/test_remediation_pipeline.py
===================================
Tests for the data/remediation_pipeline.py module:
  - Ingestion from raw .mat files
  - Anomaly detection (Z-score, IQR, Isolation Forest, Domain Rules)
  - Capping & imputation fixes
  - Synthetic data generation with physics constraints
  - Report file generation
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Skip entire module if raw .mat files are not present
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
_HAS_MAT = RAW_DIR.exists() and any(RAW_DIR.glob("B*.mat"))

pytestmark = pytest.mark.skipif(
    not _HAS_MAT,
    reason="Raw .mat files not found in data/raw/ — skipping remediation tests.",
)


# ---------------------------------------------------------------------------
# Fixture: run the pipeline once for the module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def remediation_result():
    from data.remediation_pipeline import run_remediation
    return run_remediation(
        raw_dir="data/raw",
        eol_capacity_ah=1.4,
        out_csv="data/_test_remediated.csv",
        out_json="data/_test_remediated.json",
    )


@pytest.fixture(scope="module", autouse=True)
def cleanup_test_artifacts():
    """Remove temporary test files after the module finishes."""
    yield
    for f in ["data/_test_remediated.csv", "data/_test_remediated.json",
              "data/_test_remediated.b64.txt"]:
        try:
            Path(f).unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test: Ingestion
# ---------------------------------------------------------------------------

class TestIngestion:
    def test_original_df_not_empty(self, remediation_result):
        assert len(remediation_result.df_original) > 0

    def test_has_expected_columns(self, remediation_result):
        expected = {"cell_id", "cycle_number", "capacity_ah",
                    "mean_voltage_v", "mean_temperature_c", "duration_s", "soh"}
        assert expected.issubset(set(remediation_result.df_original.columns))

    def test_capacity_positive(self, remediation_result):
        assert (remediation_result.df_original["capacity_ah"] > 0).all()


# ---------------------------------------------------------------------------
# Test: Anomaly detection
# ---------------------------------------------------------------------------

class TestAnomalyDetection:
    def test_anomaly_log_is_list(self, remediation_result):
        assert isinstance(remediation_result.anomaly_log, list)

    def test_anomaly_entries_have_required_fields(self, remediation_result):
        if remediation_result.anomaly_log:
            entry = remediation_result.anomaly_log[0]
            assert hasattr(entry, "row_idx")
            assert hasattr(entry, "col")
            assert hasattr(entry, "method")
            assert hasattr(entry, "value")
            assert hasattr(entry, "reason")

    def test_anomaly_methods_are_known(self, remediation_result):
        known_methods = {"Z-score", "IQR", "IsolationForest", "DomainRule"}
        for entry in remediation_result.anomaly_log:
            assert entry.method in known_methods, f"Unknown method: {entry.method}"

    def test_anomaly_df_property(self, remediation_result):
        df = remediation_result.anomaly_df
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert set(df.columns) == {"Row", "Column", "Method", "Value", "Reason"}


# ---------------------------------------------------------------------------
# Test: Fixing / Remediation
# ---------------------------------------------------------------------------

class TestRemediation:
    def test_fix_log_not_empty(self, remediation_result):
        """Pipeline should always apply at least one fix (IQR capping, imputation)."""
        assert len(remediation_result.fix_log) > 0

    def test_clean_df_no_nan_capacity(self, remediation_result):
        assert remediation_result.df_clean["capacity_ah"].isna().sum() == 0

    def test_soh_within_bounds(self, remediation_result):
        soh = remediation_result.df_clean["soh"]
        assert (soh >= 0.0).all()
        assert (soh <= 1.1).all()

    def test_voltage_within_physics(self, remediation_result):
        df = remediation_result.df_clean
        assert (df["min_voltage_v"] >= 2.0).all()
        assert (df["max_voltage_v"] <= 4.35).all()


# ---------------------------------------------------------------------------
# Test: Synthetic augmentation
# ---------------------------------------------------------------------------

class TestSynthetic:
    def test_synthetic_rows_added(self, remediation_result):
        assert remediation_result.n_synthetic > 0

    def test_total_equals_original_plus_synthetic(self, remediation_result):
        assert remediation_result.n_total == remediation_result.n_original + remediation_result.n_synthetic

    def test_is_synthetic_column_exists(self, remediation_result):
        assert "is_synthetic" in remediation_result.df_final.columns

    def test_synthetic_capacity_within_bounds(self, remediation_result):
        synth = remediation_result.df_final[remediation_result.df_final["is_synthetic"] == True]
        if not synth.empty:
            assert (synth["capacity_ah"] >= 1.0).all()
            assert (synth["capacity_ah"] <= 2.2).all()

    def test_synthetic_temperature_safe(self, remediation_result):
        synth = remediation_result.df_final[remediation_result.df_final["is_synthetic"] == True]
        if not synth.empty:
            assert (synth["mean_temperature_c"] >= -20).all()
            assert (synth["mean_temperature_c"] <= 80).all()


# ---------------------------------------------------------------------------
# Test: File outputs
# ---------------------------------------------------------------------------

class TestOutputFiles:
    def test_csv_written(self, remediation_result):
        assert remediation_result.csv_path is not None
        assert remediation_result.csv_path.exists()

    def test_csv_readable(self, remediation_result):
        df = pd.read_csv(remediation_result.csv_path)
        assert len(df) == remediation_result.n_total

    def test_json_written(self, remediation_result):
        assert remediation_result.json_path is not None
        assert remediation_result.json_path.exists()

    def test_report_written(self, remediation_result):
        assert remediation_result.report_path is not None
        assert remediation_result.report_path.exists()

    def test_report_contains_key_sections(self, remediation_result):
        text = remediation_result.report_path.read_text(encoding="utf-8")
        assert "Anomaly" in text
        assert "Remediation" in text or "Fixes" in text
        assert "Before" in text or "Statistics" in text


# ---------------------------------------------------------------------------
# Test: Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_stats_original_has_keys(self, remediation_result):
        assert "capacity_ah" in remediation_result.stats_original

    def test_stats_final_has_keys(self, remediation_result):
        assert "capacity_ah" in remediation_result.stats_final

    def test_correlation_matrices_are_square(self, remediation_result):
        p = remediation_result.pearson_corr
        s = remediation_result.spearman_corr
        assert p.shape[0] == p.shape[1]
        assert s.shape[0] == s.shape[1]
