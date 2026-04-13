from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dashboard.data_loading import CycleRecord
from twin.physics_models import HybridCapacityFadeModel, InternalResistanceGrowthModel


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DataQualityReport:
    total_cycles: int
    dropped_cycles: int
    capacity_outliers: int
    temperature_range: tuple[float, float]
    capacity_range: tuple[float, float]
    warnings: list[str]

    @property
    def quality_score(self) -> float:
        """0–100 heuristic quality score."""
        n = max(self.total_cycles, 1)
        drop_penalty   = 20.0 * (self.dropped_cycles / n)
        outlier_penalty = 10.0 * (self.capacity_outliers / n)
        return max(0.0, 100.0 - drop_penalty - outlier_penalty)


@dataclass(slots=True)
class TrainingMetrics:
    empirical_fit_sse: float          # residual SSE of the physics curve
    ml_cv_rmse: float                 # cross-validated RMSE of the GBR (NaN if unavailable)
    combined_residual_std: float      # final residual std used for CI
    n_anomalies: int
    quality_report: DataQualityReport


@dataclass(slots=True)
class TwinArtifacts:
    summary_df: pd.DataFrame
    nominal_capacity_ah: float
    fade_model: HybridCapacityFadeModel
    resistance_model: InternalResistanceGrowthModel
    snapshot: dict[str, float]
    training_metrics: TrainingMetrics = field(default_factory=lambda: TrainingMetrics(  # type: ignore[call-arg]
        empirical_fit_sse=float("nan"),
        ml_cv_rmse=float("nan"),
        combined_residual_std=0.03,
        n_anomalies=0,
        quality_report=DataQualityReport(0, 0, 0, (0, 0), (0, 0), []),
    ))


# ---------------------------------------------------------------------------
# Data quality assessment
# ---------------------------------------------------------------------------

def _assess_data_quality(
    cycles: list[CycleRecord],
    capacity_ah: np.ndarray,
) -> DataQualityReport:
    warnings_list: list[str] = []
    n = len(cycles)

    # IQR-based outlier count for capacity
    q25, q75 = float(np.percentile(capacity_ah, 25)), float(np.percentile(capacity_ah, 75))
    iqr = q75 - q25
    lo, hi = q25 - 2.5 * iqr, q75 + 2.5 * iqr
    cap_outliers = int(np.sum((capacity_ah < lo) | (capacity_ah > hi)))

    # Detect runs of identical capacity (sensor saturation)
    diffs = np.abs(np.diff(capacity_ah))
    stale_runs = int(np.sum(diffs < 1e-5))
    if stale_runs > n * 0.3:
        warnings_list.append(f"Possible sensor saturation: {stale_runs} consecutive unchanged capacity values.")

    # Temperature range
    all_temps = np.concatenate([c.temperature_c for c in cycles])
    t_min, t_max = float(np.nanmin(all_temps)), float(np.nanmax(all_temps))
    if t_max - t_min > 40.0:
        warnings_list.append(f"Wide temperature range ({t_min:.1f}–{t_max:.1f} °C) may reduce model accuracy.")

    # Short cycle check
    short = sum(1 for c in cycles if c.time_s.size < 100)
    dropped = short  # these would be filtered below
    if short > 0:
        warnings_list.append(f"{short} cycles have fewer than 100 time-steps and were excluded.")

    if n < 20:
        warnings_list.append(f"Only {n} cycles available; ML component is disabled below 8 cycles and less reliable below 20.")

    return DataQualityReport(
        total_cycles=n,
        dropped_cycles=dropped,
        capacity_outliers=cap_outliers,
        temperature_range=(t_min, t_max),
        capacity_range=(float(capacity_ah.min()), float(capacity_ah.max())),
        warnings=warnings_list,
    )


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_capacity_anomalies(
    deviations_ah: np.ndarray,
    z_threshold: float = 2.8,
    absolute_threshold_ah: float = 0.02,
) -> np.ndarray:
    deviation = np.asarray(deviations_ah, dtype=float)
    median = float(np.median(deviation))
    mad = float(np.median(np.abs(deviation - median)))
    scale = max(mad * 1.4826, 1e-6)
    robust_z = np.abs((deviation - median) / scale)
    return (robust_z > z_threshold) & (np.abs(deviation) > absolute_threshold_ah)


# ---------------------------------------------------------------------------
# Stage 1 – expensive, EOL-independent model fitting
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _FitResult:
    """Intermediate result: fitted models + precomputed arrays, no EOL dependency."""
    cycle_ids:           np.ndarray
    capacity_ah:         np.ndarray
    mean_temp_c:         np.ndarray
    mean_abs_current_a:  np.ndarray
    resistance_model:    InternalResistanceGrowthModel
    fade_model:          HybridCapacityFadeModel
    nominal_capacity_ah: float
    quality_report:      DataQualityReport
    model_r0:            np.ndarray
    twin_capacity:       np.ndarray
    real_soh:            np.ndarray


def fit_twin_models(cycles: list[CycleRecord]) -> _FitResult:
    """
    Expensive step – runs curve_fit + GBR.fit exactly ONCE.
    EOL threshold is intentionally absent; add it via compute_eol_artifacts().

    Complexity
    ----------
    O(N)       data accumulation (update() calls, no refitting)
    O(1)       single curve_fit + single GBR.fit  (called from fit())
    O(H)       vectorised predict_batch over history H = len(valid cycles)
    """
    if not cycles:
        raise ValueError("No cycles provided.")

    valid_cycles = [c for c in cycles if c.time_s.size >= 100 and c.capacity_ah > 0]
    if not valid_cycles:
        raise ValueError("All cycles were filtered out during quality check.")

    cycle_ids          = np.array([c.cycle_number for c in valid_cycles], dtype=float)
    capacity_ah        = np.array([c.capacity_ah  for c in valid_cycles], dtype=float)
    mean_temp_c        = np.array([float(np.median(c.temperature_c))     for c in valid_cycles], dtype=float)
    mean_abs_current_a = np.array([float(np.median(np.abs(c.current_a))) for c in valid_cycles], dtype=float)

    quality_report = _assess_data_quality(valid_cycles, capacity_ah)

    # Robust nominal capacity
    n_nominal = max(3, min(15, int(len(valid_cycles) * 0.1 + 0.5)))
    cands     = capacity_ah[:n_nominal]
    q25, q75  = float(np.percentile(cands, 25)), float(np.percentile(cands, 75))
    iqr       = q75 - q25
    inliers   = cands[(cands >= q25 - 1.5 * iqr) & (cands <= q75 + 1.5 * iqr)]
    nominal_capacity_ah = float(np.median(inliers)) if inliers.size else float(np.median(cands))

    real_soh = capacity_ah / max(nominal_capacity_ah, 1e-6)

    resistance_proxy = 0.012 + 0.025 * np.power(np.clip(1.0 - real_soh, 0.0, 1.0), 1.2)
    resistance_proxy = np.maximum.accumulate(resistance_proxy)

    # Accumulate (O(N), no refit per cycle)
    resistance_model = InternalResistanceGrowthModel()
    fade_model       = HybridCapacityFadeModel(eol_capacity_ah=0.0)   # EOL set later
    for cycle, cap, r0, temp in zip(cycle_ids, capacity_ah, resistance_proxy, mean_temp_c, strict=True):
        resistance_model.update(cycle=float(cycle), r0_ohm=float(r0))
        fade_model.update(cycle=float(cycle), capacity_ah=float(cap),
                          r0_ohm=float(r0), temperature_c=float(temp))

    # Single one-shot fit
    resistance_model.fit()
    fade_model.fit()

    # Vectorised history prediction (one GBR.predict call, no Python loop)
    model_r0      = resistance_model.predict_batch(cycle_ids)
    med_temp      = float(np.median(mean_temp_c))
    twin_capacity = fade_model.predict_capacity_batch(cycle_ids, model_r0, med_temp)

    # Small per-cycle temperature correction (~1 mAh / °C)
    temp_delta    = (mean_temp_c - med_temp) * 0.001
    twin_capacity = np.maximum(np.minimum.accumulate(twin_capacity - temp_delta), 0.0)

    return _FitResult(
        cycle_ids=cycle_ids,
        capacity_ah=capacity_ah,
        mean_temp_c=mean_temp_c,
        mean_abs_current_a=mean_abs_current_a,
        resistance_model=resistance_model,
        fade_model=fade_model,
        nominal_capacity_ah=nominal_capacity_ah,
        quality_report=quality_report,
        model_r0=model_r0,
        twin_capacity=twin_capacity,
        real_soh=real_soh,
    )


# ---------------------------------------------------------------------------
# Stage 2 – fast EOL-dependent post-processing (milliseconds)
# ---------------------------------------------------------------------------

def compute_eol_artifacts(fit: _FitResult, eol_capacity_ah: float) -> TwinArtifacts:
    """
    Apply EOL threshold and compute RUL / snapshot / metrics.
    No model refit, no mutation of the cached fit object —
    eol_capacity_ah is passed directly to predict_rul as an argument.
    Safe to call on every Streamlit render without side-effects.
    """
    eol = float(eol_capacity_ah)
    fade_model       = fit.fade_model
    resistance_model = fit.resistance_model
    # ─── DO NOT mutate fade_model.eol_capacity_ah ───────────────────────────
    # The fit object is shared across all cache entries; mutating it here would
    # corrupt previously cached TwinArtifacts for other EOL values.

    cycle_ids   = fit.cycle_ids
    capacity_ah = fit.capacity_ah
    twin_capacity = fit.twin_capacity
    real_soh      = fit.real_soh
    model_r0      = fit.model_r0
    mean_temp_c   = fit.mean_temp_c
    nominal_capacity_ah = fit.nominal_capacity_ah

    twin_soh   = twin_capacity / max(nominal_capacity_ah, 1e-6)
    deviations = capacity_ah - twin_capacity
    anomalies  = detect_capacity_anomalies(deviations)

    rul = fade_model.predict_rul(
        current_cycle=int(cycle_ids[-1]),
        resistance_model=resistance_model,
        temperature_c=float(mean_temp_c[-1]),
        eol_capacity_ah=eol,        # ← passed explicitly, no mutation
    )
    snapshot = {
        "cycle_number":             float(cycle_ids[-1]),
        "capacity_ah":              float(capacity_ah[-1]),
        "soh":                      float(real_soh[-1]),
        "internal_resistance_ohm":  float(model_r0[-1]),
        "rul_mean_cycles":          float(rul.mean_rul_cycles),
        "rul_ci_lower_90":          float(rul.ci_lower_90),
        "rul_ci_upper_90":          float(rul.ci_upper_90),
        "predicted_eol_cycle_mean": float(rul.eol_cycle_mean),
    }

    if fade_model._empirical_params is not None:
        from twin.physics_models import _capacity_law as _cl
        empirical_preds = _cl(cycle_ids, *fade_model._empirical_params)
    else:
        empirical_preds = np.full_like(cycle_ids, float(np.mean(capacity_ah)))
    empirical_sse = float(np.sum(np.square(capacity_ah - empirical_preds)))

    training_metrics = TrainingMetrics(
        empirical_fit_sse=empirical_sse,
        ml_cv_rmse=float(fade_model._residual_std),
        combined_residual_std=float(fade_model._residual_std),
        n_anomalies=int(np.sum(anomalies)),
        quality_report=fit.quality_report,
    )

    summary_df = pd.DataFrame({
        "cycle":                   cycle_ids.astype(int),
        "real_capacity_ah":        capacity_ah,
        "twin_capacity_ah":        twin_capacity,
        "real_soh":                real_soh,
        "twin_soh":                twin_soh,
        "internal_resistance_ohm": model_r0,
        "mean_temperature_c":      mean_temp_c,
        "mean_abs_current_a":      fit.mean_abs_current_a,
        "capacity_deviation_ah":   deviations,
        "is_anomaly":              anomalies,
    })

    return TwinArtifacts(
        summary_df=summary_df,
        nominal_capacity_ah=nominal_capacity_ah,
        fade_model=fade_model,
        resistance_model=resistance_model,
        snapshot=snapshot,
        training_metrics=training_metrics,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper – backward compatible
# ---------------------------------------------------------------------------

def train_twin_model(cycles: list[CycleRecord], eol_capacity_ah: float = 1.4) -> TwinArtifacts:
    """Full pipeline. For the dashboard, prefer the two-stage API."""
    return compute_eol_artifacts(fit_twin_models(cycles), eol_capacity_ah)
