from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dashboard.modeling import TwinArtifacts


@dataclass(frozen=True, slots=True)
class UsageProfile:
    name: str
    temperature_delta_c: float
    current_multiplier: float
    resistance_multiplier: float


USAGE_PROFILES: dict[str, UsageProfile] = {
    "normal": UsageProfile("normal", temperature_delta_c=0.0, current_multiplier=1.0, resistance_multiplier=1.0),
    "aggressive": UsageProfile("aggressive", temperature_delta_c=8.0, current_multiplier=1.35, resistance_multiplier=1.12),
    "high_temperature": UsageProfile("high_temperature", temperature_delta_c=14.0, current_multiplier=1.0, resistance_multiplier=1.10),
    "mild": UsageProfile("mild", temperature_delta_c=-4.0, current_multiplier=0.85, resistance_multiplier=0.95),
}


def simulate_what_if(
    artifacts: TwinArtifacts,
    horizon_cycles: int,
    profile_name: str,
    temperature_delta_c: float,
    current_multiplier: float,
    eol_capacity_ah: float = 1.4,
) -> pd.DataFrame:
    if horizon_cycles <= 0:
        return pd.DataFrame()
    profile = USAGE_PROFILES[profile_name]

    history = artifacts.summary_df
    last_cycle = int(history["cycle"].iloc[-1])
    base_temp = float(history["mean_temperature_c"].iloc[-1])
    base_current = float(history["mean_abs_current_a"].iloc[-1])
    base_capacity = float(history["real_capacity_ah"].iloc[-1])
    nominal_capacity = float(artifacts.nominal_capacity_ah)

    eff_temp = base_temp + profile.temperature_delta_c + float(temperature_delta_c)
    eff_current = max(1e-3, base_current * profile.current_multiplier * float(current_multiplier))
    current_ratio = max(0.2, eff_current / max(base_current, 1e-6))
    thermal_stress = max(0.0, eff_temp - 25.0) / 30.0
    stress = 1.0 + thermal_stress + 0.65 * max(0.0, current_ratio - 1.0)

    rows = []
    prev_capacity = base_capacity
    for step in range(1, int(horizon_cycles) + 1):
        cycle = last_cycle + step
        r0 = artifacts.resistance_model.predict(cycle) * profile.resistance_multiplier * stress
        cap_nominal = artifacts.fade_model.predict_capacity(cycle=cycle, r0_ohm=r0, temperature_c=eff_temp)
        cap = min(prev_capacity, cap_nominal - 0.0012 * max(0.0, stress - 1.0))
        cap = float(max(0.0, cap))
        prev_capacity = cap
        soh = cap / max(nominal_capacity, 1e-6)

        est_rul = _estimate_scenario_rul(
            artifacts=artifacts,
            start_cycle=cycle,
            start_capacity_ah=cap,
            temp_c=eff_temp,
            stress=stress,
            resistance_multiplier=profile.resistance_multiplier,
            eol_capacity_ah=eol_capacity_ah,
        )
        rows.append(
            {
                "cycle": cycle,
                "predicted_capacity_ah": cap,
                "predicted_soh": soh,
                "predicted_internal_resistance_ohm": r0,
                "temperature_c": eff_temp,
                "current_a": eff_current,
                "predicted_rul_cycles": est_rul,
                "profile": profile.name,
            }
        )
    return pd.DataFrame(rows)


def _estimate_scenario_rul(
    artifacts: TwinArtifacts,
    start_cycle: int,
    start_capacity_ah: float,
    temp_c: float,
    stress: float,
    resistance_multiplier: float,
    eol_capacity_ah: float,
    max_horizon_cycles: int = 5000,
) -> float:
    if start_capacity_ah <= eol_capacity_ah:
        return 0.0

    capacity = float(start_capacity_ah)
    for step in range(1, max_horizon_cycles + 1):
        cycle = start_cycle + step
        r0 = artifacts.resistance_model.predict(cycle) * resistance_multiplier * stress
        cap_nominal = artifacts.fade_model.predict_capacity(cycle=cycle, r0_ohm=r0, temperature_c=temp_c)
        capacity = min(capacity, cap_nominal - 0.0012 * max(0.0, stress - 1.0))
        if capacity <= eol_capacity_ah:
            return float(step)
    return float(max_horizon_cycles)

