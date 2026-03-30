from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.interpolate import Akima1DInterpolator


def _read_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass(slots=True)
class AkimaOCVModel:
    config_path: str | Path = "config.yaml"
    config: dict[str, Any] = field(init=False)
    h_coeff: float = field(init=False)
    _interpolator: Akima1DInterpolator | None = field(init=False, default=None, repr=False)
    _derivative: Any = field(init=False, default=None, repr=False)
    _soc_min: float | None = field(init=False, default=None, repr=False)
    _soc_max: float | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.config_path = Path(self.config_path)
        self.config = _read_config(self.config_path)
        self.h_coeff = float(self.config["ocv"]["hysteresis_coeff"])

    def fit(self, soc: np.ndarray | list[float], voltage: np.ndarray | list[float]) -> "AkimaOCVModel":
        soc_array = np.asarray(soc, dtype=float).reshape(-1)
        voltage_array = np.asarray(voltage, dtype=float).reshape(-1)

        if soc_array.size != voltage_array.size:
            raise ValueError("SoC and voltage arrays must have the same length.")
        if soc_array.size < 2:
            raise ValueError("At least two OCV samples are required.")

        order = np.argsort(soc_array)
        soc_sorted = soc_array[order]
        voltage_sorted = voltage_array[order]

        unique_soc, unique_idx = np.unique(soc_sorted, return_index=True)
        unique_voltage = voltage_sorted[unique_idx]
        if unique_soc.size < 2:
            raise ValueError("OCV fit requires at least two unique SoC points.")

        self._interpolator = Akima1DInterpolator(unique_soc, unique_voltage, extrapolate=True)
        self._derivative = self._interpolator.derivative()
        self._soc_min = float(unique_soc[0])
        self._soc_max = float(unique_soc[-1])
        return self

    def ocv(self, soc: float | np.ndarray, current: float | np.ndarray = 0.0) -> float | np.ndarray:
        interpolator = self._require_fit()
        soc_eval = self._clip_soc(soc)
        ocv_value = np.asarray(interpolator(soc_eval), dtype=float)
        current_array = np.asarray(current, dtype=float)
        ocv_eff = ocv_value + self.h_coeff * np.sign(current_array)
        return self._maybe_scalar(ocv_eff)

    def docv_dsoc(self, soc: float | np.ndarray) -> float | np.ndarray:
        self._require_fit()
        soc_eval = self._clip_soc(soc)
        derivative_value = np.asarray(self._derivative(soc_eval), dtype=float)
        return self._maybe_scalar(derivative_value)

    def _require_fit(self) -> Akima1DInterpolator:
        if self._interpolator is None:
            raise RuntimeError("OCV model has not been fit yet.")
        return self._interpolator

    def _clip_soc(self, soc: float | np.ndarray) -> np.ndarray:
        if self._soc_min is None or self._soc_max is None:
            raise RuntimeError("OCV model has not been fit yet.")
        soc_array = np.asarray(soc, dtype=float)
        return np.clip(soc_array, self._soc_min, self._soc_max)

    @staticmethod
    def _maybe_scalar(value: np.ndarray) -> float | np.ndarray:
        if value.ndim == 0:
            return float(value)
        return value


@dataclass(slots=True)
class TwoRCTheveninECM:
    config_path: str | Path = "config.yaml"
    ocv_model: AkimaOCVModel | None = None
    config: dict[str, Any] = field(init=False)
    R0: float | None = field(init=False, default=None)
    R1: float | None = field(init=False, default=None)
    C1: float | None = field(init=False, default=None)
    R2: float | None = field(init=False, default=None)
    C2: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.config_path = Path(self.config_path)
        self.config = _read_config(self.config_path)
        self.ocv_model = self.ocv_model or AkimaOCVModel(self.config_path)

    def fit(self) -> "TwoRCTheveninECM":
        params = self.config["ecm"]["parameters"]
        self.R0 = float(params["R0"])
        self.R1 = float(params["R1"])
        self.C1 = float(params["C1"])
        self.R2 = float(params["R2"])
        self.C2 = float(params["C2"])
        return self

    def predict(
        self,
        soc: float | np.ndarray,
        V1: float | np.ndarray,
        V2: float | np.ndarray,
        current: float | np.ndarray,
        dt: float,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        self._require_fit()
        _ = soc

        v1_array = np.asarray(V1, dtype=float)
        v2_array = np.asarray(V2, dtype=float)
        current_array = np.asarray(current, dtype=float)
        dt_value = float(dt)

        dV1_dt = (-v1_array / (self.R1 * self.C1)) + (current_array / self.C1)
        dV2_dt = (-v2_array / (self.R2 * self.C2)) + (current_array / self.C2)

        v1_new = v1_array + dt_value * dV1_dt
        v2_new = v2_array + dt_value * dV2_dt
        return self._maybe_scalar(v1_new), self._maybe_scalar(v2_new)

    def terminal_voltage(
        self,
        soc: float | np.ndarray,
        V1: float | np.ndarray,
        V2: float | np.ndarray,
        current: float | np.ndarray,
    ) -> float | np.ndarray:
        self._require_fit()
        ocv_value = np.asarray(self.ocv_model.ocv(soc, current=current), dtype=float)
        current_array = np.asarray(current, dtype=float)
        v1_array = np.asarray(V1, dtype=float)
        v2_array = np.asarray(V2, dtype=float)
        voltage = ocv_value - (current_array * self.R0) - v1_array - v2_array
        return self._maybe_scalar(voltage)

    def _require_fit(self) -> None:
        required = (self.R0, self.R1, self.C1, self.R2, self.C2)
        if any(value is None for value in required):
            raise RuntimeError("ECM parameters have not been loaded. Call fit() first.")

    @staticmethod
    def _maybe_scalar(value: np.ndarray) -> float | np.ndarray:
        if value.ndim == 0:
            return float(value)
        return value


if __name__ == "__main__":
    ocv_model = AkimaOCVModel("config.yaml")
    soc_dummy = np.linspace(0.0, 1.0, 11)
    voltage_dummy = np.array([3.0, 3.18, 3.29, 3.39, 3.48, 3.58, 3.67, 3.77, 3.89, 4.02, 4.15], dtype=float)
    ocv_model.fit(soc_dummy, voltage_dummy)

    ecm = TwoRCTheveninECM("config.yaml", ocv_model=ocv_model).fit()

    soc = 0.8
    current = 1.5
    dt = 1.0
    v1 = 0.0
    v2 = 0.0

    for step in range(10):
        v1, v2 = ecm.predict(soc=soc, V1=v1, V2=v2, current=current, dt=dt)
        vt = ecm.terminal_voltage(soc=soc, V1=v1, V2=v2, current=current)
        print(f"step={step:02d} terminal_voltage={vt:.6f} V")
