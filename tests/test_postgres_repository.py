from __future__ import annotations

import numpy as np

from data.postgres import BatterySample, PostgresBatteryRepository


def test_battery_selector_accepts_known_tables() -> None:
    assert PostgresBatteryRepository.battery_table_name("b0005") == "b0005"
    assert PostgresBatteryRepository.battery_table_name("B0018") == "b0018"


def test_battery_selector_rejects_unknown_table() -> None:
    try:
        PostgresBatteryRepository.battery_table_name("b9999")
    except ValueError as exc:
        assert "Unsupported battery" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported table name.")


def test_samples_are_grouped_to_cycles() -> None:
    samples = [
        BatterySample(1, "discharge", 0.0, 4.10, -1.0, 25.0, 1.99, 24.0, "a.mat"),
        BatterySample(1, "discharge", 1.0, 4.00, -1.0, 25.2, 1.99, 24.0, "a.mat"),
        BatterySample(2, "discharge", 0.0, 4.08, -1.0, 25.4, 1.98, 24.0, "a.mat"),
    ]

    cycles = PostgresBatteryRepository.to_cycle_dicts(samples)
    assert len(cycles) == 2

    first = cycles[0]
    assert first["cycle_number"] == 1
    assert first["type"] == "discharge"
    assert np.allclose(first["time"], np.array([0.0, 1.0]))
    assert np.allclose(first["V"], np.array([4.10, 4.00]))
    assert np.allclose(first["I"], np.array([-1.0, -1.0]))
    assert np.isclose(float(first["C_max"]), 1.99)
