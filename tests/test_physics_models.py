from __future__ import annotations

import numpy as np

from twin.physics_models import (
    HybridCapacityFadeModel,
    InternalResistanceGrowthModel,
    VoltageCurveEvolutionModel,
)


def test_hybrid_capacity_model_predicts_rul_to_1p4ah() -> None:
    resistance = InternalResistanceGrowthModel()
    model = HybridCapacityFadeModel(eol_capacity_ah=1.4)

    for cycle in range(1, 31):
        capacity = 2.0 - 0.015 * cycle
        r0 = 0.015 + 0.0002 * cycle
        temp = 25.0
        resistance.update(cycle=cycle, r0_ohm=r0)
        model.update(cycle=cycle, capacity_ah=capacity, r0_ohm=r0, temperature_c=temp)

    pred = model.predict_rul(current_cycle=30, resistance_model=resistance, temperature_c=25.0)
    assert pred.mean_rul_cycles >= 0.0
    assert pred.eol_cycle_mean >= 30.0
    assert pred.ci_upper_90 >= pred.ci_lower_90


def test_voltage_curve_evolution_tracks_drift() -> None:
    model = VoltageCurveEvolutionModel()

    t = np.arange(0.0, 120.0, 1.0)
    i = -np.ones_like(t)
    cmax = 2.0

    for cycle in range(3):
        soc = 1.0 - np.cumsum(np.maximum(-i, 0.0)) / (3600.0 * cmax)
        v = 3.0 + 1.1 * soc - 0.01 * cycle
        model.update(cycle_number=cycle, time=t, current=i, voltage=v, c_max=cmax)

    drift = model.drift_rmse(2)
    pred = model.predict_curve(4)
    assert np.isfinite(drift)
    assert drift > 0.0
    assert np.all(np.isfinite(pred))
