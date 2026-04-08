from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io


@dataclass(frozen=True, slots=True)
class CycleRecord:
    cycle_number: int
    time_s: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    temperature_c: np.ndarray
    dt_s: float
    capacity_ah: float


def list_available_cells(data_dir: str | Path) -> list[str]:
    path = Path(data_dir)
    if not path.exists():
        return []
    return sorted([file.stem.upper() for file in path.glob("B*.mat")])


def load_discharge_cycles(cell_id: str, data_dir: str | Path) -> list[CycleRecord]:
    cell = str(cell_id).upper().strip()
    mat_path = Path(data_dir) / f"{cell}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Cell file not found: {mat_path}")

    raw = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    try:
        cycles_raw = raw[cell]["cycle"]
    except KeyError:
        key = [k for k in raw if not k.startswith("_")][-1]
        cycles_raw = raw[key]["cycle"]

    cycles: list[CycleRecord] = []
    idx = 0
    for cycle in cycles_raw:
        if str(cycle.get("type", "")).strip().lower() != "discharge":
            continue

        data = cycle.get("data", {})
        voltage = np.asarray(data.get("Voltage_measured", []), dtype=float).ravel()
        current = np.asarray(data.get("Current_measured", []), dtype=float).ravel()
        temp = np.asarray(data.get("Temperature_measured", []), dtype=float).ravel()
        time = np.asarray(data.get("Time", []), dtype=float).ravel()
        if voltage.size < 10:
            continue

        n = min(voltage.size, current.size, temp.size)
        voltage = voltage[:n]
        current = current[:n]
        temp = temp[:n]
        if time.size >= n:
            time = time[:n]
        else:
            time = np.arange(n, dtype=float)

        dt = float(np.median(np.diff(time))) if time.size > 1 else 1.0
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0
        capacity = float(np.trapz(np.abs(current), dx=dt) / 3600.0)

        cycles.append(
            CycleRecord(
                cycle_number=idx,
                time_s=time,
                voltage_v=voltage,
                current_a=current,
                temperature_c=temp,
                dt_s=dt,
                capacity_ah=capacity,
            )
        )
        idx += 1
    return cycles


def cycle_to_dict(cycle: CycleRecord) -> dict[str, Any]:
    return {
        "cycle_number": cycle.cycle_number,
        "time_s": cycle.time_s,
        "voltage_v": cycle.voltage_v,
        "current_a": cycle.current_a,
        "temperature_c": cycle.temperature_c,
        "dt_s": cycle.dt_s,
        "capacity_ah": cycle.capacity_ah,
    }

