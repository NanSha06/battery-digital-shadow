from __future__ import annotations

import numpy as np

from estimation.ekf import OnlineEKF
from ecm.model import AkimaOCVModel


def _build_ocv_model() -> AkimaOCVModel:
    ocv_model = AkimaOCVModel("config.yaml")
    soc = np.linspace(0.0, 1.0, 11, dtype=float)
    voltage = 3.0 + 0.85 * soc + 0.25 * np.sqrt(soc)
    ocv_model.fit(soc, voltage)
    return ocv_model


def _build_ekf() -> OnlineEKF:
    return OnlineEKF(
        ocv_model=_build_ocv_model(),
        ecm_params={"R0": 0.015, "R1": 0.01, "C1": 2500.0, "R2": 0.02, "C2": 12000.0},
        config="config.yaml",
        nominal_capacity_ah=2.0,
    )


def test_predict_only() -> None:
    ekf = _build_ekf()
    ekf.reset(x0=[0.8, 0.1, -0.05])

    x_pred = ekf._state_transition(ekf.x, I=1.5, dt=2.0)
    F = ekf._state_transition_jacobian(dt=2.0)

    expected_soc = 0.8 - (1.5 * 2.0) / (3600.0 * 2.0)
    expected_v1 = 0.1 + 2.0 * ((-0.1 / (0.01 * 2500.0)) + (1.5 / 2500.0))
    expected_v2 = -0.05 + 2.0 * ((0.05 / (0.02 * 12000.0)) + (1.5 / 12000.0))

    assert np.allclose(x_pred, [expected_soc, expected_v1, expected_v2])
    assert np.allclose(
        F,
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0 - 2.0 / (0.01 * 2500.0), 0.0],
            [0.0, 0.0, 1.0 - 2.0 / (0.02 * 12000.0)],
        ],
    )


def test_update_step() -> None:
    ekf = _build_ekf()
    ekf.reset(x0=[0.7, 0.01, 0.02], P0=np.diag([0.05, 0.02, 0.02]))

    prior_trace = float(np.trace(ekf.P))
    predicted_voltage = ekf._measurement_function(ekf._state_transition(ekf.x, I=1.0, dt=1.0), I=1.0)
    _, posterior = ekf.step(V_meas=predicted_voltage - 0.03, I=1.0, T=25.0, dt=1.0)

    assert float(np.trace(posterior)) < prior_trace


def test_soc_clipping() -> None:
    ekf = _build_ekf()
    ekf.reset(x0=[0.99, 0.0, 0.0], P0=np.diag([0.05, 0.001, 0.001]))

    high_voltage = float(ekf.ocv_model.ocv(1.0, current=0.0) + 0.5)
    state, _ = ekf.step(V_meas=high_voltage, I=0.0, T=25.0, dt=1.0)

    assert 0.0 <= state[0] <= 1.0
    assert np.isclose(state[0], 1.0)


def test_full_step() -> None:
    ekf = _build_ekf()
    ekf.reset(x0=[0.95, 0.0, 0.0])

    dt = 1.0
    current_profile = 1.1 + 0.4 * np.sin(np.linspace(0.0, 6.0 * np.pi, 100))
    true_state = np.array([0.95, 0.0, 0.0], dtype=float)

    for current in current_profile:
        true_state = ekf._state_transition(true_state, I=float(current), dt=dt)
        voltage = ekf._measurement_function(true_state, I=float(current))
        state, covariance = ekf.step(V_meas=voltage, I=float(current), T=25.0, dt=dt)
        assert 0.0 <= state[0] <= 1.0
        assert np.all(np.isfinite(covariance))

