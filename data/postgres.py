from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


ALLOWED_BATTERIES: tuple[str, ...] = ("b0005", "b0006", "b0007", "b0018")


@dataclass(slots=True)
class BatterySample:
    cycle_number: int
    cycle_type: str
    time: float
    voltage_measured: float
    current_measured: float
    temperature_measured: float
    capacity: float | None
    ambient_temperature: float | None
    source_file: str | None


class PostgresBatteryRepository:
    def __init__(self, database_url: str | None = None, engine: Engine | None = None) -> None:
        if engine is None and database_url is None:
            database_url = os.getenv("DATABASE_URL")
        if engine is None and not database_url:
            raise ValueError("Provide a PostgreSQL URL or set DATABASE_URL.")
        self.engine = engine or create_engine(str(database_url), future=True, pool_pre_ping=True)

    @staticmethod
    def battery_table_name(battery: str) -> str:
        normalized = battery.strip().lower()
        if normalized not in ALLOWED_BATTERIES:
            allowed = ", ".join(ALLOWED_BATTERIES)
            raise ValueError(f"Unsupported battery '{battery}'. Allowed values: {allowed}")
        return normalized

    def list_available_batteries(self) -> tuple[str, ...]:
        return ALLOWED_BATTERIES

    def fetch_samples(
        self,
        battery: str,
        cycle_type: str | None = None,
        cycle_number_min: int | None = None,
    ) -> list[BatterySample]:
        table = self.battery_table_name(battery)
        sql = f"""
        SELECT
            cycle_number,
            cycle_type,
            time,
            voltage_measured,
            current_measured,
            temperature_measured,
            capacity,
            ambient_temperature,
            source_file
        FROM {table}
        WHERE (:cycle_type IS NULL OR cycle_type = :cycle_type)
          AND (:cycle_number_min IS NULL OR cycle_number >= :cycle_number_min)
        ORDER BY cycle_number ASC, time ASC
        """
        stmt = text(sql)
        params = {"cycle_type": cycle_type, "cycle_number_min": cycle_number_min}
        with self.engine.connect() as conn:
            rows = conn.execute(stmt, params).mappings().all()

        return [
            BatterySample(
                cycle_number=int(row["cycle_number"]),
                cycle_type=str(row["cycle_type"]),
                time=float(row["time"]),
                voltage_measured=float(row["voltage_measured"]),
                current_measured=float(row["current_measured"]),
                temperature_measured=float(row["temperature_measured"]),
                capacity=None if row["capacity"] is None else float(row["capacity"]),
                ambient_temperature=None if row["ambient_temperature"] is None else float(row["ambient_temperature"]),
                source_file=None if row["source_file"] is None else str(row["source_file"]),
            )
            for row in rows
        ]

    @staticmethod
    def to_cycle_dicts(samples: Iterable[BatterySample]) -> list[dict[str, object]]:
        grouped: dict[tuple[int, str], list[BatterySample]] = {}
        for sample in samples:
            key = (sample.cycle_number, sample.cycle_type.lower())
            grouped.setdefault(key, []).append(sample)

        cycles: list[dict[str, object]] = []
        for (cycle_number, cycle_type), group in sorted(grouped.items(), key=lambda item: item[0][0]):
            group_sorted = sorted(group, key=lambda x: x.time)
            time = np.asarray([s.time for s in group_sorted], dtype=float)
            voltage = np.asarray([s.voltage_measured for s in group_sorted], dtype=float)
            current = np.asarray([s.current_measured for s in group_sorted], dtype=float)
            temp = np.asarray([s.temperature_measured for s in group_sorted], dtype=float)

            capacities = [s.capacity for s in group_sorted if s.capacity is not None and np.isfinite(s.capacity)]
            capacity = float(capacities[-1]) if capacities else float(np.trapz(np.abs(current), x=time) / 3600.0)

            ambient_values = [s.ambient_temperature for s in group_sorted if s.ambient_temperature is not None]
            ambient_temperature = float(np.median(ambient_values)) if ambient_values else np.nan

            source_files = [s.source_file for s in group_sorted if s.source_file]
            source_file = source_files[0] if source_files else None

            cycles.append(
                {
                    "cycle_number": cycle_number,
                    "cycle_idx": cycle_number,
                    "type": cycle_type,
                    "time": time,
                    "V": voltage,
                    "I": current,
                    "T": temp,
                    "dt": float(np.median(np.diff(time))) if time.size > 1 else 1.0,
                    "C_max": capacity,
                    "capacity": capacity,
                    "ambient_temperature": ambient_temperature,
                    "source_file": source_file,
                }
            )

        return cycles

    def fetch_cycles(
        self,
        battery: str,
        cycle_type: str | None = "discharge",
        cycle_number_min: int | None = None,
    ) -> list[dict[str, object]]:
        samples = self.fetch_samples(
            battery=battery,
            cycle_type=cycle_type,
            cycle_number_min=cycle_number_min,
        )
        return self.to_cycle_dicts(samples)
