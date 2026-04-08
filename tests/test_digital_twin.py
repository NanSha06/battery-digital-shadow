from __future__ import annotations

import numpy as np

from twin.digital_twin import BatteryDigitalTwin


class _FakeRepository:
    def __init__(self, n_cycles: int = 6) -> None:
        self._battery = "b0005"
        self.n_cycles = n_cycles

    def battery_table_name(self, battery: str) -> str:
        if battery.lower() not in {"b0005", "b0006", "b0007", "b0018"}:
            raise ValueError("unsupported")
        return battery.lower()

    def fetch_cycles(self, battery: str, cycle_type: str | None = "discharge", cycle_number_min: int | None = None):
        _ = cycle_type
        cycles = []
        for k in range(self.n_cycles):
            if cycle_number_min is not None and k < cycle_number_min:
                continue
            t = np.arange(0.0, 60.0, 1.0)
            i = -np.full_like(t, 1.0)
            c_max = 2.0 - 0.01 * k
            soc = 1.0 - np.cumsum(np.maximum(-i, 0.0)) / (3600.0 * c_max)
            v = 3.2 + 0.9 * soc - 0.015 * i
            cycles.append(
                {
                    "cycle_number": k,
                    "cycle_idx": k,
                    "type": "discharge",
                    "time": t,
                    "V": v,
                    "I": i,
                    "T": np.full_like(t, 25.0),
                    "dt": 1.0,
                    "C_max": c_max,
                }
            )
        return cycles


def test_digital_twin_select_and_bootstrap() -> None:
    repo = _FakeRepository()
    twin = BatteryDigitalTwin(repository=repo, config_path="config.yaml")
    twin.select_battery("B0005")
    snapshot = twin.bootstrap(max_fit_cycles=4)

    assert snapshot.battery == "b0005"
    assert snapshot.cycle_number >= 0
    assert 0.0 <= snapshot.soc <= 1.0
    assert snapshot.soh > 0.0
    assert snapshot.capacity_ah > 0.0
    assert snapshot.internal_resistance_ohm > 0.0
    assert snapshot.predicted_capacity_next_cycle_ah > 0.0
    assert snapshot.predicted_eol_cycle_1p4ah >= snapshot.cycle_number
    assert snapshot.rul_1p4ah_mean_cycles >= 0.0
    assert snapshot.rul_1p4ah_ci_upper_90 >= snapshot.rul_1p4ah_ci_lower_90


def test_what_if_profiles_and_realtime_update() -> None:
    repo = _FakeRepository(n_cycles=4)
    twin = BatteryDigitalTwin(repository=repo, config_path="config.yaml")
    twin.select_battery("b0005")
    twin.bootstrap(max_fit_cycles=4)

    normal = twin.run_what_if_scenario(n_cycles=15, profile="normal")
    aggressive = twin.run_what_if_scenario(n_cycles=15, profile="aggressive")
    assert aggressive["trajectory"][-1].predicted_capacity_ah <= normal["trajectory"][-1].predicted_capacity_ah
    assert aggressive["estimated_eol_cycle_1p4ah"] <= normal["estimated_eol_cycle_1p4ah"]

    repo.n_cycles = 6
    update_result = twin.update_from_database()
    assert update_result["processed_cycles"] == 2
    assert update_result["snapshot"].cycle_number == 5
