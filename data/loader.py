from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy.io
import yaml


@dataclass(slots=True)
class Cycle:
    index: int
    V: np.ndarray
    I: np.ndarray
    T: np.ndarray
    timestamps: np.ndarray
    type: str
    Z_real: np.ndarray | None = None
    Z_imag: np.ndarray | None = None
    frequencies: np.ndarray | None = None
    cell_id: str | None = None
    source_type: str | None = None
    c_rate: float | None = None
    is_quasi_static: bool = False


@dataclass(slots=True)
class CycleDataset:
    cycles: list[Cycle] = field(default_factory=list)
    nominal_capacity_ah: float | None = None

    def __len__(self) -> int:
        return len(self.cycles)

    def __iter__(self) -> Iterable[Cycle]:
        return iter(self.cycles)

    def __getitem__(self, item: int) -> Cycle:
        return self.cycles[item]

    def by_type(self, cycle_type: str) -> list[Cycle]:
        return [cycle for cycle in self.cycles if cycle.type == cycle_type]

    @property
    def quasi_static_discharges(self) -> list[Cycle]:
        return [cycle for cycle in self.cycles if cycle.type == "discharge" and cycle.is_quasi_static]


class NASALoader:
    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        self.config_path = Path(config_path)
        self.config = self._read_config(self.config_path)
        self.data_path = (self.config_path.parent / self.config["dataset"]["data_path"]).resolve()
        self.cell_ids = list(self.config["dataset"]["cell_ids"])
        self.current_hysteresis = float(self.config["dataset"].get("current_hysteresis", 0.05))
        self.datasets: dict[str, CycleDataset] = {}

    def fit(self, cell_ids: list[str] | None = None) -> "NASALoader":
        targets = cell_ids or self.cell_ids
        self.datasets = {cell_id: self._load_cell(cell_id) for cell_id in targets}
        return self

    def predict(self, cell_id: str | None = None) -> CycleDataset | dict[str, CycleDataset]:
        if not self.datasets:
            self.fit()
        if cell_id is None:
            return self.datasets
        if cell_id not in self.datasets:
            self.datasets[cell_id] = self._load_cell(cell_id)
        return self.datasets[cell_id]

    def update(self, cell_id: str) -> CycleDataset:
        dataset = self._load_cell(cell_id)
        self.datasets[cell_id] = dataset
        return dataset

    @staticmethod
    def _read_config(config_path: Path) -> dict[str, Any]:
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _load_cell(self, cell_id: str) -> CycleDataset:
        mat_path = self.data_path / f"{cell_id}.mat"
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        experiments = np.atleast_1d(mat[cell_id].cycle)

        nominal_capacity = self._estimate_nominal_capacity(experiments)
        cycles: list[Cycle] = []
        cycle_index = 0

        for experiment in experiments:
            experiment_type = str(getattr(experiment, "type", "")).lower()
            if experiment_type == "impedance":
                cycles.append(self._parse_impedance_cycle(experiment, cell_id, cycle_index))
                cycle_index += 1
                continue

            segmented_cycles = self._parse_timeseries_experiment(
                experiment=experiment,
                cell_id=cell_id,
                start_index=cycle_index,
                nominal_capacity_ah=nominal_capacity,
            )
            cycles.extend(segmented_cycles)
            cycle_index += len(segmented_cycles)

        return CycleDataset(cycles=cycles, nominal_capacity_ah=nominal_capacity)

    def _estimate_nominal_capacity(self, experiments: np.ndarray) -> float | None:
        capacities: list[float] = []
        for experiment in experiments:
            if str(getattr(experiment, "type", "")).lower() != "discharge":
                continue
            capacity = getattr(getattr(experiment, "data", None), "Capacity", None)
            if capacity is None:
                continue
            value = float(np.asarray(capacity).reshape(-1)[-1])
            if np.isfinite(value) and value > 0.0:
                capacities.append(value)

        if not capacities:
            return None
        return float(np.median(capacities))

    def _parse_timeseries_experiment(
        self,
        experiment: Any,
        cell_id: str,
        start_index: int,
        nominal_capacity_ah: float | None,
    ) -> list[Cycle]:
        data = experiment.data
        time_raw = self._as_1d_float(data.Time)
        voltage_raw = self._as_1d_float(data.Voltage_measured)
        current_raw = self._as_1d_float(data.Current_measured)
        temperature_raw = self._as_1d_float(data.Temperature_measured)

        if time_raw.size == 0:
            return []

        seconds, voltage, current, temperature = self._resample_to_1hz(
            time_raw,
            voltage_raw,
            current_raw,
            temperature_raw,
        )
        timestamps = self._build_timestamps(experiment.time, seconds)
        labels = self._label_current_state(current, self.current_hysteresis)
        boundaries = self._segment_boundaries(labels)

        cycles: list[Cycle] = []
        experiment_type = str(getattr(experiment, "type", "")).lower()

        next_index = start_index
        for start, end in boundaries:
            segment_label = labels[start]
            if segment_label == 0:
                continue

            cycle_type = "charge" if segment_label > 0 else "discharge"
            if experiment_type in {"charge", "discharge"} and cycle_type != experiment_type:
                continue

            segment_current = current[start:end]
            c_rate = self._estimate_c_rate(segment_current, nominal_capacity_ah)

            cycles.append(
                Cycle(
                    index=next_index,
                    V=voltage[start:end],
                    I=segment_current,
                    T=temperature[start:end],
                    timestamps=timestamps[start:end],
                    type=cycle_type,
                    cell_id=cell_id,
                    source_type=experiment_type,
                    c_rate=c_rate,
                    is_quasi_static=self._is_quasi_static_discharge(cycle_type, c_rate),
                )
            )
            next_index += 1

        return cycles

    def _parse_impedance_cycle(self, experiment: Any, cell_id: str, index: int) -> Cycle:
        data = experiment.data
        impedance = np.asarray(getattr(data, "Rectified_Impedance", getattr(data, "Battery_impedance")))
        impedance = np.atleast_1d(impedance).astype(np.complex128, copy=False)
        frequencies = self._extract_frequencies(data, impedance.size)
        timestamp = self._build_timestamps(experiment.time, np.array([0.0]))

        return Cycle(
            index=index,
            V=np.array([], dtype=float),
            I=np.array([], dtype=float),
            T=np.array([], dtype=float),
            timestamps=timestamp,
            type="EIS",
            Z_real=np.real(impedance),
            Z_imag=np.imag(impedance),
            frequencies=frequencies,
            cell_id=cell_id,
            source_type="impedance",
        )

    @staticmethod
    def _as_1d_float(values: Any) -> np.ndarray:
        return np.asarray(values, dtype=float).reshape(-1)

    @staticmethod
    def _resample_to_1hz(
        time_raw: np.ndarray,
        voltage_raw: np.ndarray,
        current_raw: np.ndarray,
        temperature_raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        unique_time, unique_idx = np.unique(time_raw, return_index=True)
        voltage_unique = voltage_raw[unique_idx]
        current_unique = current_raw[unique_idx]
        temperature_unique = temperature_raw[unique_idx]

        if unique_time.size == 1:
            seconds = np.array([0.0], dtype=float)
            return seconds, voltage_unique.copy(), current_unique.copy(), temperature_unique.copy()

        stop = int(np.floor(unique_time[-1]))
        seconds = np.arange(0, stop + 1, 1.0, dtype=float)
        if seconds.size == 0:
            seconds = np.array([0.0], dtype=float)

        voltage = np.interp(seconds, unique_time, voltage_unique)
        current = np.interp(seconds, unique_time, current_unique)
        temperature = np.interp(seconds, unique_time, temperature_unique)
        return seconds, voltage, current, temperature

    @staticmethod
    def _build_timestamps(time_parts: Any, seconds: np.ndarray) -> np.ndarray:
        year, month, day, hour, minute, second = np.asarray(time_parts, dtype=float).tolist()
        sec_int = int(second)
        millis = int(round((second - sec_int) * 1000.0))
        base = np.datetime64(
            f"{int(year):04d}-{int(month):02d}-{int(day):02d}T{int(hour):02d}:{int(minute):02d}:{sec_int:02d}.{millis:03d}"
        )
        return base + seconds.astype("timedelta64[s]")

    @staticmethod
    def _label_current_state(current: np.ndarray, hysteresis: float) -> np.ndarray:
        labels = np.zeros(current.shape[0], dtype=np.int8)
        state = 0
        for idx, value in enumerate(current):
            if value >= hysteresis:
                state = 1
            elif value <= -hysteresis:
                state = -1
            labels[idx] = state
        return labels

    @staticmethod
    def _segment_boundaries(labels: np.ndarray) -> list[tuple[int, int]]:
        if labels.size == 0:
            return []

        boundaries: list[tuple[int, int]] = []
        start = 0
        current_label = labels[0]
        for idx in range(1, labels.size):
            if labels[idx] == current_label:
                continue
            boundaries.append((start, idx))
            start = idx
            current_label = labels[idx]
        boundaries.append((start, labels.size))
        return boundaries

    @staticmethod
    def _estimate_c_rate(current: np.ndarray, nominal_capacity_ah: float | None) -> float | None:
        if nominal_capacity_ah is None or nominal_capacity_ah <= 0.0 or current.size == 0:
            return None
        active_current = np.abs(current[np.abs(current) > 0.0])
        if active_current.size == 0:
            return 0.0
        return float(np.median(active_current) / nominal_capacity_ah)

    @staticmethod
    def _is_quasi_static_discharge(cycle_type: str, c_rate: float | None) -> bool:
        if cycle_type != "discharge" or c_rate is None:
            return False
        return bool(np.isclose(c_rate, 1.0 / 20.0, rtol=0.35, atol=0.01))

    @staticmethod
    def _extract_frequencies(data: Any, size: int) -> np.ndarray:
        for name in ("Frequency", "Frequencies", "freq", "frequency"):
            if hasattr(data, name):
                return np.asarray(getattr(data, name), dtype=float).reshape(-1)

        # The public NASA PCoE .mat files bundled here do not expose the
        # excitation frequency vector inside the impedance structs.
        return np.full(size, np.nan, dtype=float)


if __name__ == "__main__":
    loader = NASALoader()
    dataset = loader.update("B0005")
    discharge_cycles = dataset.by_type("discharge")

    print(f"Total cycles found for B0005: {len(dataset)}")
    if discharge_cycles:
        first = discharge_cycles[0]
        print(f"First discharge cycle V shape: {first.V.shape}")
    else:
        print("No discharge cycles found.")
