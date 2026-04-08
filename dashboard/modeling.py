from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dashboard.data_loading import CycleRecord
from twin.physics_models import HybridCapacityFadeModel, InternalResistanceGrowthModel


@dataclass(slots=True)
class TwinArtifacts:
    summary_df: pd.DataFrame
    nominal_capacity_ah: float
    fade_model: HybridCapacityFadeModel
    resistance_model: InternalResistanceGrowthModel
    snapshot: dict[str, float]


def train_twin_model(cycles: list[CycleRecord], eol_capacity_ah: float = 1.4) -> TwinArtifacts:
    if not cycles:
        raise ValueError("No cycles provided.")

    cycle_ids = np.array([c.cycle_number for c in cycles], dtype=float)
    capacity_ah = np.array([c.capacity_ah for c in cycles], dtype=float)
    mean_temp_c = np.array([float(np.median(c.temperature_c)) for c in cycles], dtype=float)
    mean_abs_current_a = np.array([float(np.median(np.abs(c.current_a))) for c in cycles], dtype=float)

    nominal_capacity_ah = float(np.median(capacity_ah[: min(10, capacity_ah.size)]))
    real_soh = capacity_ah / max(nominal_capacity_ah, 1e-6)

    # Fast resistance proxy derived from fade trend; keeps resistance physically monotonic.
    resistance_proxy = 0.012 + 0.025 * np.power(np.clip(1.0 - real_soh, 0.0, 1.0), 1.2)
    resistance_proxy = np.maximum.accumulate(resistance_proxy)

    resistance_model = InternalResistanceGrowthModel()
    fade_model = HybridCapacityFadeModel(eol_capacity_ah=eol_capacity_ah)
    for cycle, cap, r0, temp in zip(cycle_ids, capacity_ah, resistance_proxy, mean_temp_c, strict=True):
        resistance_model.update(cycle=float(cycle), r0_ohm=float(r0))
        fade_model.update(
            cycle=float(cycle),
            capacity_ah=float(cap),
            r0_ohm=float(r0),
            temperature_c=float(temp),
        )

    model_r0 = np.array([float(resistance_model.predict(cycle)) for cycle in cycle_ids], dtype=float)
    twin_capacity = np.array(
        [
            float(fade_model.predict_capacity(cycle=cycle, r0_ohm=r0, temperature_c=temp))
            for cycle, r0, temp in zip(cycle_ids, model_r0, mean_temp_c, strict=True)
        ],
        dtype=float,
    )
    twin_capacity = np.minimum.accumulate(twin_capacity)
    twin_soh = twin_capacity / max(nominal_capacity_ah, 1e-6)

    deviations = capacity_ah - twin_capacity
    anomalies = detect_capacity_anomalies(deviations)

    rul = fade_model.predict_rul(
        current_cycle=int(cycle_ids[-1]),
        resistance_model=resistance_model,
        temperature_c=float(mean_temp_c[-1]),
    )
    snapshot = {
        "cycle_number": float(cycle_ids[-1]),
        "capacity_ah": float(capacity_ah[-1]),
        "soh": float(real_soh[-1]),
        "internal_resistance_ohm": float(model_r0[-1]),
        "rul_mean_cycles": float(rul.mean_rul_cycles),
        "rul_ci_lower_90": float(rul.ci_lower_90),
        "rul_ci_upper_90": float(rul.ci_upper_90),
        "predicted_eol_cycle_mean": float(rul.eol_cycle_mean),
    }

    summary_df = pd.DataFrame(
        {
            "cycle": cycle_ids.astype(int),
            "real_capacity_ah": capacity_ah,
            "twin_capacity_ah": twin_capacity,
            "real_soh": real_soh,
            "twin_soh": twin_soh,
            "internal_resistance_ohm": model_r0,
            "mean_temperature_c": mean_temp_c,
            "mean_abs_current_a": mean_abs_current_a,
            "capacity_deviation_ah": deviations,
            "is_anomaly": anomalies,
        }
    )

    return TwinArtifacts(
        summary_df=summary_df,
        nominal_capacity_ah=nominal_capacity_ah,
        fade_model=fade_model,
        resistance_model=resistance_model,
        snapshot=snapshot,
    )


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

