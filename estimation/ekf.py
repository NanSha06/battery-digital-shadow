from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecm.model import AkimaOCVModel


def _read_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    path = Path(config)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass(slots=True)
class OnlineEKF:
    ocv_model: AkimaOCVModel
    ecm_params: dict[str, float]
    config: str | Path | dict[str, Any] = "config.yaml"
    nominal_capacity_ah: float | None = None
    config_data: dict[str, Any] = field(init=False, repr=False)
    x: np.ndarray = field(init=False, repr=False)
    P: np.ndarray = field(init=False, repr=False)
    Q: np.ndarray = field(init=False, repr=False)
    R: float = field(init=False)
    P0: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.config_data = _read_config(self.config)
        ekf_config = self.config_data["ekf"]
        self.Q = np.diag(np.asarray(ekf_config["Q_diag"], dtype=float))
        self.R = float(ekf_config["R"])
        self.P0 = np.diag(np.asarray(ekf_config["P0_diag"], dtype=float))
        self.nominal_capacity_ah = float(
            self.nominal_capacity_ah
            if self.nominal_capacity_ah is not None
            else self.config_data.get("ageing", {}).get("C0", 1.0)
        )
        self.ecm_params = {key: float(value) for key, value in self.ecm_params.items()}
        self.reset()

    def reset(
        self,
        x0: np.ndarray | list[float] | None = None,
        P0: np.ndarray | list[list[float]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.x = np.asarray(x0 if x0 is not None else [1.0, 0.0, 0.0], dtype=float).reshape(3)
        self.x[0] = float(np.clip(self.x[0], 0.0, 1.0))
        self.P = np.asarray(P0 if P0 is not None else self.P0, dtype=float).reshape(3, 3)
        return self.x.copy(), self.P.copy()

    def fit(
        self,
        x0: np.ndarray | list[float] | None = None,
        P0: np.ndarray | list[list[float]] | None = None,
    ) -> "OnlineEKF":
        self.reset(x0=x0, P0=P0)
        return self

    def predict(
        self,
        V_meas: float,
        I: float,
        T: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.step(V_meas=V_meas, I=I, T=T, dt=dt)

    def update(
        self,
        V_meas: float,
        I: float,
        T: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.step(V_meas=V_meas, I=I, T=T, dt=dt)

    def step(self, V_meas: float, I: float, T: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
        _ = T
        x_pred = self._state_transition(self.x, I=I, dt=dt)
        F = self._state_transition_jacobian(dt=dt)
        P_pred = F @ self.P @ F.T + self.Q

        h_x = self._measurement_function(x_pred, I=I)
        H = self._measurement_jacobian(x_pred[0])
        S = float(H @ P_pred @ H.T + self.R)
        K = (P_pred @ H.T) / S
        residual = float(V_meas - h_x)

        x_post = x_pred + K * residual
        x_post[0] = float(np.clip(x_post[0], 0.0, 1.0))

        identity = np.eye(3, dtype=float)
        P_post = (identity - np.outer(K, H)) @ P_pred
        P_post = 0.5 * (P_post + P_post.T)

        self.x = x_post
        self.P = P_post
        return self.x.copy(), self.P.copy()

    def _state_transition(self, x: np.ndarray, I: float, dt: float) -> np.ndarray:
        soc, v1, v2 = np.asarray(x, dtype=float)
        r1 = self.ecm_params["R1"]
        c1 = self.ecm_params["C1"]
        r2 = self.ecm_params["R2"]
        c2 = self.ecm_params["C2"]

        dsoc_dt = -float(I) / (3600.0 * self.nominal_capacity_ah)
        dv1_dt = (-v1 / (r1 * c1)) + (float(I) / c1)
        dv2_dt = (-v2 / (r2 * c2)) + (float(I) / c2)

        return np.array(
            [
                soc + float(dt) * dsoc_dt,
                v1 + float(dt) * dv1_dt,
                v2 + float(dt) * dv2_dt,
            ],
            dtype=float,
        )

    def _state_transition_jacobian(self, dt: float) -> np.ndarray:
        r1 = self.ecm_params["R1"]
        c1 = self.ecm_params["C1"]
        r2 = self.ecm_params["R2"]
        c2 = self.ecm_params["C2"]
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0 - float(dt) / (r1 * c1), 0.0],
                [0.0, 0.0, 1.0 - float(dt) / (r2 * c2)],
            ],
            dtype=float,
        )

    def _measurement_function(self, x: np.ndarray, I: float) -> float:
        soc, v1, v2 = np.asarray(x, dtype=float)
        return float(self.ocv_model.ocv(soc, current=I) - float(I) * self.ecm_params["R0"] - v1 - v2)

    def _measurement_jacobian(self, soc: float) -> np.ndarray:
        return np.array([float(self.ocv_model.docv_dsoc(soc)), -1.0, -1.0], dtype=float)


def _build_ocv_model() -> AkimaOCVModel:
    ocv_model = AkimaOCVModel("config.yaml")
    soc = np.linspace(0.0, 1.0, 11, dtype=float)
    voltage = 3.0 + 0.85 * soc + 0.25 * np.sqrt(soc)
    ocv_model.fit(soc, voltage)
    return ocv_model


if __name__ == "__main__":
    ocv_model = _build_ocv_model()
    ekf = OnlineEKF(ocv_model=ocv_model, ecm_params={"R0": 0.015, "R1": 0.01, "C1": 2500.0, "R2": 0.02, "C2": 12000.0})

    dt = 1.0
    current_profile = 1.2 + 0.3 * np.sin(np.linspace(0.0, 4.0 * np.pi, 100))
    true_state = np.array([0.95, 0.0, 0.0], dtype=float)
    final_cov = None
    for current in current_profile:
        true_state = ekf._state_transition(true_state, I=float(current), dt=dt)
        voltage = ekf._measurement_function(true_state, I=float(current))
        state, final_cov = ekf.step(V_meas=voltage, I=float(current), T=25.0, dt=dt)

    print(f"Final SoC: {state[0]:.6f}")
    print("Final P diagonal:", np.diag(final_cov))
