from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import Cycle
from ecm.model import AkimaOCVModel, TwoRCTheveninECM


PARAMETER_ORDER = ("R0", "R1", "C1", "R2", "C2")


def _read_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass(slots=True)
class OfflineIdentifier:
    config_path: str | Path = "config.yaml"
    ocv_model: AkimaOCVModel | None = None
    nominal_capacity_ah: float | None = None
    config: dict[str, Any] = field(init=False)
    ecm_model: TwoRCTheveninECM = field(init=False)
    lut: np.ndarray | None = field(init=False, default=None)
    lut_counts: np.ndarray | None = field(init=False, default=None, repr=False)
    soc_axis: np.ndarray | None = field(init=False, default=None)
    temp_axis: np.ndarray | None = field(init=False, default=None)
    cycle_records: list[dict[str, Any]] = field(init=False, default_factory=list, repr=False)
    bounds: list[tuple[float, float]] = field(init=False)
    de_popsize: int = field(init=False)
    de_maxiter: int = field(init=False)
    de_seed: int | None = field(init=False)

    def __post_init__(self) -> None:
        self.config_path = Path(self.config_path)
        self.config = _read_config(self.config_path)
        self.ocv_model = self.ocv_model or AkimaOCVModel(self.config_path)
        self.ecm_model = TwoRCTheveninECM(self.config_path, ocv_model=self.ocv_model)
        self._load_optimiser_config()

    def fit(self, cycle_list: list[Cycle | dict[str, Any]]) -> "OfflineIdentifier":
        discharge_cycles = [cycle for cycle in cycle_list if self._cycle_type(cycle) == "discharge"]
        if not discharge_cycles:
            raise ValueError("fit() requires at least one discharge cycle.")

        temp_min, temp_max = self._temperature_range(discharge_cycles)
        self.soc_axis = np.linspace(0.0, 1.0, int(self.config["ecm"]["soc_bins"]), dtype=float)
        self.temp_axis = np.linspace(temp_min, temp_max, int(self.config["ecm"]["temp_bins"]), dtype=float)
        self.cycle_records = []

        for cycle in discharge_cycles:
            prepared = self._prepare_cycle(cycle)
            result = differential_evolution(
                func=self._objective,
                bounds=self.bounds,
                args=(prepared,),
                popsize=self.de_popsize,
                maxiter=self.de_maxiter,
                seed=self.de_seed,
                polish=False,
            )
            params = self._vector_to_params(result.x)
            self.cycle_records.append({"cycle": prepared, "params": params, "rmse": float(result.fun)})

        self._rebuild_lut()
        return self

    def predict(self, cycle: Cycle | dict[str, Any]) -> np.ndarray:
        if self.lut is None:
            raise RuntimeError("LUT is not available. Call fit() or load_lut() first.")
        prepared = self._prepare_cycle(cycle)
        return self._simulate_cycle(prepared, static_params=None)

    def update(self, new_cycles: list[Cycle | dict[str, Any]]) -> "OfflineIdentifier":
        if self.soc_axis is None or self.temp_axis is None:
            return self.fit(new_cycles)

        discharge_cycles = [cycle for cycle in new_cycles if self._cycle_type(cycle) == "discharge"]
        if not discharge_cycles:
            return self

        current_temp_min = float(self.temp_axis[0])
        current_temp_max = float(self.temp_axis[-1])
        new_temp_min, new_temp_max = self._temperature_range(discharge_cycles)
        merged_temp_min = min(current_temp_min, new_temp_min)
        merged_temp_max = max(current_temp_max, new_temp_max)
        self.temp_axis = np.linspace(merged_temp_min, merged_temp_max, self.temp_axis.size, dtype=float)

        for cycle in discharge_cycles:
            prepared = self._prepare_cycle(cycle)
            result = differential_evolution(
                func=self._objective,
                bounds=self.bounds,
                args=(prepared,),
                popsize=self.de_popsize,
                maxiter=self.de_maxiter,
                seed=self.de_seed,
                polish=False,
            )
            params = self._vector_to_params(result.x)
            self.cycle_records.append({"cycle": prepared, "params": params, "rmse": float(result.fun)})

        self._rebuild_lut()
        return self

    def interpolate_params(self, soc: float, T: float) -> dict[str, float]:
        if self.lut is None or self.soc_axis is None or self.temp_axis is None:
            raise RuntimeError("LUT is not available. Call fit() or load_lut() first.")

        soc_value = float(np.clip(soc, self.soc_axis[0], self.soc_axis[-1]))
        temp_value = float(np.clip(T, self.temp_axis[0], self.temp_axis[-1]))
        soc_hi = int(np.searchsorted(self.soc_axis, soc_value, side="right"))
        temp_hi = int(np.searchsorted(self.temp_axis, temp_value, side="right"))
        soc_hi = min(max(soc_hi, 1), self.soc_axis.size - 1)
        temp_hi = min(max(temp_hi, 1), self.temp_axis.size - 1)
        soc_lo = soc_hi - 1
        temp_lo = temp_hi - 1

        soc0 = self.soc_axis[soc_lo]
        soc1 = self.soc_axis[soc_hi]
        temp0 = self.temp_axis[temp_lo]
        temp1 = self.temp_axis[temp_hi]

        soc_weight = 0.0 if np.isclose(soc1, soc0) else (soc_value - soc0) / (soc1 - soc0)
        temp_weight = 0.0 if np.isclose(temp1, temp0) else (temp_value - temp0) / (temp1 - temp0)

        q11 = self.lut[soc_lo, temp_lo]
        q21 = self.lut[soc_hi, temp_lo]
        q12 = self.lut[soc_lo, temp_hi]
        q22 = self.lut[soc_hi, temp_hi]
        interpolated = (
            (1.0 - soc_weight) * (1.0 - temp_weight) * q11
            + soc_weight * (1.0 - temp_weight) * q21
            + (1.0 - soc_weight) * temp_weight * q12
            + soc_weight * temp_weight * q22
        )
        return self._vector_to_params(interpolated)

    def save_lut(self, path: str | Path) -> Path:
        if self.lut is None or self.soc_axis is None or self.temp_axis is None:
            raise RuntimeError("Nothing to save. Call fit() first.")

        target = Path(path)
        np.savez_compressed(
            target,
            lut=self.lut,
            counts=self.lut_counts,
            soc_axis=self.soc_axis,
            temp_axis=self.temp_axis,
        )
        return target

    def load_lut(self, path: str | Path) -> "OfflineIdentifier":
        payload = np.load(Path(path))
        self.lut = np.asarray(payload["lut"], dtype=float)
        self.lut_counts = np.asarray(payload["counts"], dtype=float)
        self.soc_axis = np.asarray(payload["soc_axis"], dtype=float)
        self.temp_axis = np.asarray(payload["temp_axis"], dtype=float)
        return self

    def _load_optimiser_config(self) -> None:
        ecm_config = self.config["ecm"]
        self.bounds = [
            tuple(map(float, ecm_config["r0_bounds"])),
            tuple(map(float, ecm_config["r1_bounds"])),
            tuple(map(float, ecm_config["c1_bounds"])),
            tuple(map(float, ecm_config["r2_bounds"])),
            tuple(map(float, ecm_config["c2_bounds"])),
        ]
        self.de_popsize = int(ecm_config["de_popsize"])
        self.de_maxiter = int(ecm_config["de_maxiter"])
        self.de_seed = ecm_config.get("de_seed")

    def _objective(self, theta: np.ndarray, cycle: dict[str, np.ndarray]) -> float:
        predicted = self._simulate_cycle(cycle, static_params=self._vector_to_params(theta))
        error = predicted - cycle["V"]
        return float(np.sqrt(np.mean(np.square(error))))

    def _simulate_cycle(
        self,
        cycle: dict[str, np.ndarray],
        static_params: dict[str, float] | None,
    ) -> np.ndarray:
        current = cycle["I"]
        soc = cycle["soc"]
        temperature = cycle["T"]
        dt = cycle["dt"]
        predicted = np.empty_like(cycle["V"])
        v1 = 0.0
        v2 = 0.0

        for idx in range(current.size):
            if static_params is None:
                params = self.interpolate_params(float(soc[idx]), float(temperature[idx]))
            else:
                params = static_params
            self._apply_params(params)
            v1, v2 = self.ecm_model.predict(
                soc=float(soc[idx]),
                V1=v1,
                V2=v2,
                current=float(current[idx]),
                dt=float(dt[idx]),
            )
            predicted[idx] = float(
                self.ecm_model.terminal_voltage(
                    soc=float(soc[idx]),
                    V1=v1,
                    V2=v2,
                    current=float(current[idx]),
                )
            )
        return predicted

    def _apply_params(self, params: dict[str, float]) -> None:
        self.ecm_model.R0 = float(params["R0"])
        self.ecm_model.R1 = float(params["R1"])
        self.ecm_model.C1 = float(params["C1"])
        self.ecm_model.R2 = float(params["R2"])
        self.ecm_model.C2 = float(params["C2"])

    def _prepare_cycle(self, cycle: Cycle | dict[str, Any]) -> dict[str, np.ndarray]:
        voltage = self._extract_array(cycle, ("V", "voltage", "Voltage_measured"))
        current = self._extract_array(cycle, ("I", "current", "Current_measured"))
        temperature = self._extract_array(cycle, ("T", "temperature", "Temperature_measured"))
        soc = self._extract_soc(cycle, current)
        dt = self._extract_dt(cycle, voltage.size)

        if voltage.size != current.size or voltage.size != temperature.size or voltage.size != soc.size:
            raise ValueError("Cycle arrays V/I/T/soc must have the same length.")
        if voltage.size < 2:
            raise ValueError("Each cycle must contain at least two samples.")

        return {
            "V": voltage,
            "I": current,
            "T": temperature,
            "soc": np.clip(soc, 0.0, 1.0),
            "dt": np.maximum(dt, 1e-6),
        }

    def _extract_soc(self, cycle: Cycle | dict[str, Any], current: np.ndarray) -> np.ndarray:
        for key in ("soc", "SoC", "SOC"):
            value = self._get_field(cycle, key)
            if value is not None:
                return np.asarray(value, dtype=float).reshape(-1)

        timestamps = self._get_field(cycle, "timestamps")
        capacity_ah = self._get_field(cycle, "capacity_ah")
        if capacity_ah is None:
            capacity_ah = self.nominal_capacity_ah
        if capacity_ah is None:
            capacity_ah = self.config.get("ageing", {}).get("C0")

        if timestamps is not None and capacity_ah is not None:
            dt = self._compute_dt_from_timestamps(np.asarray(timestamps))
            discharged_ah = np.cumsum(np.maximum(-current, 0.0) * dt / 3600.0)
            return 1.0 - (discharged_ah / float(capacity_ah))

        return np.linspace(1.0, 0.0, current.size, dtype=float)

    def _extract_dt(self, cycle: Cycle | dict[str, Any], size: int) -> np.ndarray:
        dt_value = self._get_field(cycle, "dt")
        if dt_value is not None:
            dt_array = np.asarray(dt_value, dtype=float).reshape(-1)
            if dt_array.size == 1:
                return np.full(size, float(dt_array[0]), dtype=float)
            if dt_array.size == size:
                return dt_array
            raise ValueError("dt must be scalar or match cycle length.")

        time_value = self._get_field(cycle, "time")
        if time_value is not None:
            time_array = np.asarray(time_value, dtype=float).reshape(-1)
            if time_array.size != size:
                raise ValueError("time must match cycle length.")
            return self._compute_dt_from_time(time_array)

        timestamps = self._get_field(cycle, "timestamps")
        if timestamps is not None:
            return self._compute_dt_from_timestamps(np.asarray(timestamps))

        return np.ones(size, dtype=float)

    def _rebuild_lut(self) -> None:
        if self.soc_axis is None or self.temp_axis is None:
            raise RuntimeError("Axes are not defined.")

        lut_shape = (self.soc_axis.size, self.temp_axis.size, len(PARAMETER_ORDER))
        sums = np.zeros(lut_shape, dtype=float)
        counts = np.zeros((self.soc_axis.size, self.temp_axis.size), dtype=float)

        for record in self.cycle_records:
            cycle = record["cycle"]
            theta = self._params_to_vector(record["params"])
            soc_indices = np.abs(self.soc_axis[:, None] - cycle["soc"][None, :]).argmin(axis=0)
            temp_indices = np.abs(self.temp_axis[:, None] - cycle["T"][None, :]).argmin(axis=0)
            for soc_idx, temp_idx in zip(soc_indices, temp_indices, strict=True):
                sums[soc_idx, temp_idx] += theta
                counts[soc_idx, temp_idx] += 1.0

        defaults = self._params_to_vector(
            {
                key: float(self.config["ecm"]["parameters"][key])
                for key in PARAMETER_ORDER
            }
        )
        lut = np.empty_like(sums)
        filled_mask = counts > 0.0
        lut[filled_mask] = sums[filled_mask] / counts[filled_mask, None]
        lut[~filled_mask] = defaults

        known_locations = np.argwhere(filled_mask)
        if known_locations.size > 0:
            missing_locations = np.argwhere(~filled_mask)
            for soc_idx, temp_idx in missing_locations:
                distances = np.square(known_locations[:, 0] - soc_idx) + np.square(known_locations[:, 1] - temp_idx)
                nearest = known_locations[int(np.argmin(distances))]
                lut[soc_idx, temp_idx] = lut[nearest[0], nearest[1]]

        self.lut = lut
        self.lut_counts = counts

    def _temperature_range(self, cycles: list[Cycle | dict[str, Any]]) -> tuple[float, float]:
        all_temperatures = []
        for cycle in cycles:
            values = self._extract_array(cycle, ("T", "temperature", "Temperature_measured"))
            all_temperatures.append(values)
        stacked = np.concatenate(all_temperatures)
        temp_min = float(np.min(stacked))
        temp_max = float(np.max(stacked))
        if np.isclose(temp_min, temp_max):
            temp_min -= 1.0
            temp_max += 1.0
        return temp_min, temp_max

    @staticmethod
    def _compute_dt_from_time(time: np.ndarray) -> np.ndarray:
        dt = np.diff(time, prepend=time[0])
        if dt.size > 1:
            dt[0] = dt[1]
        else:
            dt[0] = 1.0
        return np.maximum(dt, 1e-6)

    @staticmethod
    def _compute_dt_from_timestamps(timestamps: np.ndarray) -> np.ndarray:
        seconds = timestamps.astype("datetime64[ns]").astype(np.int64) / 1e9
        return OfflineIdentifier._compute_dt_from_time(seconds.astype(float))

    @staticmethod
    def _cycle_type(cycle: Cycle | dict[str, Any]) -> str:
        cycle_type = OfflineIdentifier._get_field(cycle, "type")
        if cycle_type is None:
            return "discharge"
        return str(cycle_type).lower()

    @staticmethod
    def _get_field(cycle: Cycle | dict[str, Any], key: str) -> Any:
        if isinstance(cycle, dict):
            return cycle.get(key)
        return getattr(cycle, key, None)

    @staticmethod
    def _extract_array(cycle: Cycle | dict[str, Any], keys: tuple[str, ...]) -> np.ndarray:
        for key in keys:
            value = OfflineIdentifier._get_field(cycle, key)
            if value is not None:
                return np.asarray(value, dtype=float).reshape(-1)
        joined = ", ".join(keys)
        raise ValueError(f"Cycle is missing required field. Expected one of: {joined}.")

    @staticmethod
    def _vector_to_params(theta: np.ndarray) -> dict[str, float]:
        return {name: float(value) for name, value in zip(PARAMETER_ORDER, np.asarray(theta, dtype=float), strict=True)}

    @staticmethod
    def _params_to_vector(params: dict[str, float]) -> np.ndarray:
        return np.array([float(params[name]) for name in PARAMETER_ORDER], dtype=float)


def _make_synthetic_cycle(
    identifier: OfflineIdentifier,
    start_soc: float,
    current_a: float,
    temperature_c: float,
    params: dict[str, float],
    steps: int = 80,
    dt: float = 1.0,
) -> dict[str, np.ndarray]:
    capacity_ah = float(identifier.config.get("ageing", {}).get("C0", 2.0))
    discharged_ah = np.arange(steps, dtype=float) * current_a * dt / 3600.0
    soc = np.clip(start_soc - (discharged_ah / capacity_ah), 0.0, 1.0)
    current = -np.full(steps, abs(current_a), dtype=float)
    temperature = np.full(steps, temperature_c, dtype=float)
    time = np.arange(steps, dtype=float) * dt

    identifier._apply_params(params)
    v1 = 0.0
    v2 = 0.0
    voltage = np.empty(steps, dtype=float)
    for idx in range(steps):
        v1, v2 = identifier.ecm_model.predict(soc=soc[idx], V1=v1, V2=v2, current=current[idx], dt=dt)
        voltage[idx] = identifier.ecm_model.terminal_voltage(soc=soc[idx], V1=v1, V2=v2, current=current[idx])

    rng = np.random.default_rng(7)
    voltage += rng.normal(0.0, 0.002, size=steps)
    return {
        "type": "discharge",
        "V": voltage,
        "I": current,
        "T": temperature,
        "soc": soc,
        "time": time,
        "capacity_ah": capacity_ah,
    }


if __name__ == "__main__":
    ocv_model = AkimaOCVModel("config.yaml")
    soc_knots = np.linspace(0.0, 1.0, 11, dtype=float)
    ocv_knots = 3.0 + 0.85 * soc_knots + 0.25 * np.sqrt(soc_knots)
    ocv_model.fit(soc_knots, ocv_knots)

    identifier = OfflineIdentifier("config.yaml", ocv_model=ocv_model, nominal_capacity_ah=2.0)
    synthetic_cycles = [
        _make_synthetic_cycle(identifier, start_soc=0.95, current_a=1.2, temperature_c=20.0, params={"R0": 0.014, "R1": 0.011, "C1": 2600.0, "R2": 0.023, "C2": 10000.0}),
        _make_synthetic_cycle(identifier, start_soc=0.85, current_a=1.5, temperature_c=27.0, params={"R0": 0.016, "R1": 0.013, "C1": 2400.0, "R2": 0.021, "C2": 11000.0}),
        _make_synthetic_cycle(identifier, start_soc=0.75, current_a=1.0, temperature_c=34.0, params={"R0": 0.012, "R1": 0.009, "C1": 3000.0, "R2": 0.018, "C2": 12500.0}),
    ]

    identifier.fit(synthetic_cycles)
    print("LUT shape:", identifier.lut.shape)
    print("LUT slice [soc=0, :, :]:")
    print(identifier.lut[0])
