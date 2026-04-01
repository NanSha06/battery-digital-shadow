from __future__ import annotations

import numpy as np

from ecm.identifier import OfflineIdentifier, _make_synthetic_cycle
from ecm.model import AkimaOCVModel


def _build_ocv_model() -> AkimaOCVModel:
    ocv_model = AkimaOCVModel("config.yaml")
    soc = np.linspace(0.0, 1.0, 11, dtype=float)
    voltage = 3.0 + 0.85 * soc + 0.25 * np.sqrt(soc)
    ocv_model.fit(soc, voltage)
    return ocv_model


def _build_identifier() -> OfflineIdentifier:
    return OfflineIdentifier("config.yaml", ocv_model=_build_ocv_model(), nominal_capacity_ah=2.0)


def _synthetic_cycles(identifier: OfflineIdentifier) -> list[dict[str, np.ndarray]]:
    return [
        _make_synthetic_cycle(
            identifier,
            start_soc=0.95,
            current_a=1.2,
            temperature_c=20.0,
            params={"R0": 0.014, "R1": 0.011, "C1": 2600.0, "R2": 0.023, "C2": 10000.0},
            steps=40,
        ),
        _make_synthetic_cycle(
            identifier,
            start_soc=0.85,
            current_a=1.5,
            temperature_c=27.0,
            params={"R0": 0.016, "R1": 0.013, "C1": 2400.0, "R2": 0.021, "C2": 11000.0},
            steps=40,
        ),
        _make_synthetic_cycle(
            identifier,
            start_soc=0.75,
            current_a=1.0,
            temperature_c=34.0,
            params={"R0": 0.012, "R1": 0.009, "C1": 3000.0, "R2": 0.018, "C2": 12500.0},
            steps=40,
        ),
    ]


def test_offline_identifier_fit_predict_update_and_persistence(tmp_path) -> None:
    identifier = _build_identifier()
    cycles = _synthetic_cycles(identifier)

    identifier.fit(cycles)

    assert identifier.lut is not None
    assert identifier.lut.shape == (10, 5, 5)

    params = identifier.interpolate_params(soc=0.5, T=27.0)
    assert set(params) == {"R0", "R1", "C1", "R2", "C2"}
    assert all(value > 0.0 for value in params.values())

    predicted = identifier.predict(cycles[0])
    assert predicted.shape == cycles[0]["V"].shape
    assert np.all(np.isfinite(predicted))

    save_path = tmp_path / "identifier_lut.npz"
    identifier.save_lut(save_path)

    restored = OfflineIdentifier("config.yaml", ocv_model=_build_ocv_model()).load_lut(save_path)
    assert restored.lut is not None
    assert np.allclose(restored.lut, identifier.lut)
    assert np.allclose(restored.soc_axis, identifier.soc_axis)
    assert np.allclose(restored.temp_axis, identifier.temp_axis)

    updated_cycle = _make_synthetic_cycle(
        identifier,
        start_soc=0.70,
        current_a=1.3,
        temperature_c=30.0,
        params={"R0": 0.013, "R1": 0.010, "C1": 2800.0, "R2": 0.020, "C2": 11800.0},
        steps=40,
    )
    records_before = len(identifier.cycle_records)
    identifier.update([updated_cycle])

    assert len(identifier.cycle_records) == records_before + 1
    assert identifier.lut.shape == (10, 5, 5)
