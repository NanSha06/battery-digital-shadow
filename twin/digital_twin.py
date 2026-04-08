from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ageing.parametric import AgeingModel
from data.postgres import PostgresBatteryRepository
from ecm.identifier import OfflineIdentifier
from ecm.model import AkimaOCVModel
from estimation.ekf import OnlineEKF
from twin.physics_models import (
    HybridCapacityFadeModel,
    InternalResistanceGrowthModel,
    VoltageCurveEvolutionModel,
)


@dataclass(slots=True)
class TwinSnapshot:
    battery: str
    cycle_number: int
    soc: float
    soh: float
    capacity_ah: float
    internal_resistance_ohm: float
    r0_growth_rate_ohm_per_cycle: float
    voltage_curve_drift_rmse: float
    voltage_curve_at_50soc: float
    predicted_voltage_next_cycle_50soc: float
    predicted_capacity_next_cycle_ah: float
    predicted_eol_cycle_1p4ah: float
    rul_1p4ah_mean_cycles: float
    rul_1p4ah_ci_lower_90: float
    rul_1p4ah_ci_upper_90: float
    rul_mean_cycles: float
    rul_ci_lower_90: float
    rul_ci_upper_90: float
    ecm_params: dict[str, float]
    voltage_rmse: float


@dataclass(frozen=True, slots=True)
class UsageProfile:
    name: str
    temperature_delta_c: float
    current_multiplier: float
    resistance_multiplier: float


@dataclass(slots=True)
class TwinSimulationPoint:
    profile: str
    cycle_number: int
    temperature_c: float
    discharge_current_a: float
    predicted_capacity_ah: float
    predicted_soh: float
    predicted_r0_ohm: float
    predicted_voltage_50soc: float
    predicted_rul_1p4ah_cycles: float


class BatteryDigitalTwin:
    USAGE_PROFILES: dict[str, UsageProfile] = {
        "normal": UsageProfile(name="normal", temperature_delta_c=0.0, current_multiplier=1.0, resistance_multiplier=1.0),
        "aggressive": UsageProfile(name="aggressive", temperature_delta_c=8.0, current_multiplier=1.4, resistance_multiplier=1.15),
        "high_temperature": UsageProfile(name="high_temperature", temperature_delta_c=15.0, current_multiplier=1.0, resistance_multiplier=1.12),
        "mild": UsageProfile(name="mild", temperature_delta_c=-4.0, current_multiplier=0.85, resistance_multiplier=0.95),
    }

    def __init__(
        self,
        repository: PostgresBatteryRepository,
        config_path: str | Path = "config.yaml",
    ) -> None:
        self.repository = repository
        self.config_path = Path(config_path)
        self.selected_battery: str | None = None
        self.nominal_capacity_ah: float | None = None

        self.ocv_model = AkimaOCVModel(config_path=self.config_path)
        self.identifier = OfflineIdentifier(config_path=self.config_path, ocv_model=self.ocv_model)
        self.ageing_model = AgeingModel(config_path=str(self.config_path))
        self.hybrid_fade_model = HybridCapacityFadeModel(eol_capacity_ah=1.4)
        self.resistance_growth_model = InternalResistanceGrowthModel()
        self.voltage_curve_model = VoltageCurveEvolutionModel()
        self.ekf: OnlineEKF | None = None

        self._is_bootstrapped = False
        self._latest_cycle_number: int = -1
        self._latest_snapshot: TwinSnapshot | None = None
        self._latest_temperature_c: float = 25.0
        self._latest_discharge_current_a: float = 1.0

    def select_battery(self, battery: str) -> str:
        self.selected_battery = self.repository.battery_table_name(battery)
        self._is_bootstrapped = False
        self._latest_cycle_number = -1
        self._latest_snapshot = None
        self.hybrid_fade_model = HybridCapacityFadeModel(eol_capacity_ah=1.4)
        self.resistance_growth_model = InternalResistanceGrowthModel()
        self.voltage_curve_model = VoltageCurveEvolutionModel()
        self._latest_temperature_c = 25.0
        self._latest_discharge_current_a = 1.0
        return self.selected_battery

    def bootstrap(self, max_fit_cycles: int = 20) -> TwinSnapshot:
        if self.selected_battery is None:
            raise RuntimeError("Call select_battery() first.")

        discharge_cycles = self.repository.fetch_cycles(self.selected_battery, cycle_type="discharge")
        if not discharge_cycles:
            raise RuntimeError(f"No discharge cycles found for {self.selected_battery}.")

        self.nominal_capacity_ah = self._infer_nominal_capacity(discharge_cycles)
        fit_cycles = discharge_cycles[: max(3, min(max_fit_cycles, len(discharge_cycles)))]

        self._fit_ocv_model_from_cycles(fit_cycles)
        self.identifier.nominal_capacity_ah = self.nominal_capacity_ah
        self.identifier.fit(fit_cycles)

        theta_hist = self._build_theta_history(fit_cycles)
        cycle_indices = [int(c["cycle_number"]) for c in fit_cycles]
        self.ageing_model.fit(cycle_indices=cycle_indices, theta_list=theta_hist)

        ecm_params = theta_hist[-1]
        self.ekf = OnlineEKF(
            ocv_model=self.ocv_model,
            ecm_params=ecm_params,
            config=self.config_path,
            nominal_capacity_ah=self.nominal_capacity_ah,
        )
        self.ekf.fit(x0=[1.0, 0.0, 0.0])

        snapshot: TwinSnapshot | None = None
        for cycle in fit_cycles:
            snapshot = self._ingest_cycle(cycle)

        if snapshot is None:
            raise RuntimeError("Bootstrap failed: no cycles were processed.")

        self._is_bootstrapped = True
        return snapshot

    def sync(self) -> TwinSnapshot:
        result = self.update_from_database()
        return result["snapshot"]

    def update_from_database(self) -> dict[str, Any]:
        if self.selected_battery is None:
            raise RuntimeError("Call select_battery() first.")
        if not self._is_bootstrapped:
            snapshot = self.bootstrap()
            return {"processed_cycles": max(0, snapshot.cycle_number + 1), "snapshot": snapshot}

        new_cycles = self.repository.fetch_cycles(
            self.selected_battery,
            cycle_type="discharge",
            cycle_number_min=self._latest_cycle_number + 1,
        )
        if not new_cycles:
            if self._latest_snapshot is None:
                raise RuntimeError("Twin has not processed any cycles yet.")
            return {"processed_cycles": 0, "snapshot": self._latest_snapshot}

        for cycle in new_cycles:
            self._latest_snapshot = self._ingest_cycle(cycle)
        return {"processed_cycles": len(new_cycles), "snapshot": self._latest_snapshot}

    def simulate_future_cycles(
        self,
        n_cycles: int,
        profile: str = "normal",
        temperature_c: float | None = None,
        discharge_current_a: float | None = None,
        temperature_delta_c: float = 0.0,
        current_multiplier: float = 1.0,
    ) -> list[TwinSimulationPoint]:
        if n_cycles <= 0:
            return []
        if self._latest_snapshot is None or self.nominal_capacity_ah is None:
            raise RuntimeError("Twin has no state. Run bootstrap()/sync() first.")

        scenario = self.USAGE_PROFILES.get(profile.lower())
        if scenario is None:
            allowed = ", ".join(sorted(self.USAGE_PROFILES))
            raise ValueError(f"Unsupported profile '{profile}'. Allowed: {allowed}")

        base_temp = float(self._latest_temperature_c if temperature_c is None else temperature_c)
        base_current = float(self._latest_discharge_current_a if discharge_current_a is None else discharge_current_a)
        eff_temp = base_temp + scenario.temperature_delta_c + float(temperature_delta_c)
        eff_current = max(1e-3, base_current * scenario.current_multiplier * float(current_multiplier))

        points: list[TwinSimulationPoint] = []
        prev_capacity = float(self._latest_snapshot.capacity_ah)
        for step in range(1, int(n_cycles) + 1):
            cycle_number = self._latest_cycle_number + step
            thermal_stress = max(0.0, eff_temp - 25.0) / 30.0
            c_rate_stress = max(0.0, eff_current / max(base_current, 1e-6) - 1.0)
            stress = 1.0 + thermal_stress + 0.7 * c_rate_stress

            r0 = self.resistance_growth_model.predict(cycle_number) * scenario.resistance_multiplier * stress
            cap_model = self.hybrid_fade_model.predict_capacity(cycle=cycle_number, r0_ohm=r0, temperature_c=eff_temp)
            cap = min(prev_capacity, cap_model - 0.001 * (stress - 1.0))
            cap = float(max(0.0, cap))
            prev_capacity = cap

            soh = float(cap / self.nominal_capacity_ah)
            v_curve = self.voltage_curve_model.predict_curve(cycle_number)
            v50 = float(np.interp(0.5, self.voltage_curve_model.soc_grid, v_curve))
            rul = self._estimate_scenario_rul(
                start_cycle=cycle_number,
                start_capacity_ah=cap,
                temp_c=eff_temp,
                current_a=eff_current,
                scenario=scenario,
            )
            points.append(
                TwinSimulationPoint(
                    profile=scenario.name,
                    cycle_number=cycle_number,
                    temperature_c=eff_temp,
                    discharge_current_a=eff_current,
                    predicted_capacity_ah=cap,
                    predicted_soh=soh,
                    predicted_r0_ohm=float(r0),
                    predicted_voltage_50soc=v50,
                    predicted_rul_1p4ah_cycles=rul,
                )
            )
        return points

    def run_what_if_scenario(
        self,
        n_cycles: int = 200,
        profile: str = "normal",
        temperature_c: float | None = None,
        discharge_current_a: float | None = None,
        temperature_delta_c: float = 0.0,
        current_multiplier: float = 1.0,
    ) -> dict[str, Any]:
        trajectory = self.simulate_future_cycles(
            n_cycles=n_cycles,
            profile=profile,
            temperature_c=temperature_c,
            discharge_current_a=discharge_current_a,
            temperature_delta_c=temperature_delta_c,
            current_multiplier=current_multiplier,
        )
        if not trajectory:
            return {"trajectory": [], "final_cycle": self._latest_cycle_number, "estimated_eol_cycle_1p4ah": self._latest_cycle_number}

        final_point = trajectory[-1]
        estimated_eol_cycle = int(final_point.cycle_number + final_point.predicted_rul_1p4ah_cycles)
        return {
            "trajectory": trajectory,
            "final_cycle": final_point.cycle_number,
            "estimated_eol_cycle_1p4ah": estimated_eol_cycle,
            "profile": final_point.profile,
        }

    def get_snapshot(self) -> TwinSnapshot | None:
        return self._latest_snapshot

    def _ingest_cycle(self, cycle: dict[str, Any]) -> TwinSnapshot:
        if self.ekf is None or self.nominal_capacity_ah is None:
            raise RuntimeError("Twin is not bootstrapped.")

        voltage = np.asarray(cycle["V"], dtype=float)
        current = np.asarray(cycle["I"], dtype=float)
        temp = np.asarray(cycle["T"], dtype=float)
        dt = float(cycle.get("dt", 1.0))

        predicted = np.empty_like(voltage)
        x = self.ekf.x.copy()
        for idx, (v_meas, i, t) in enumerate(zip(voltage, current, temp, strict=True)):
            x, _ = self.ekf.step(V_meas=float(v_meas), I=float(i), T=float(t), dt=dt)
            predicted[idx] = float(self.ekf._measurement_function(x, I=float(i)))

        rmse = float(np.sqrt(np.mean(np.square(predicted - voltage))))
        soc_end = float(np.clip(x[0], 0.0, 1.0))

        c_max = float(cycle.get("C_max", self.nominal_capacity_ah))
        soh = float(np.clip(c_max / self.nominal_capacity_ah, 0.0, 2.0))

        r0_est = self._estimate_r0(cycle, soc_end)
        self.resistance_growth_model.update(cycle=int(cycle["cycle_number"]), r0_ohm=r0_est)

        voltage_curve = self.voltage_curve_model.update(
            cycle_number=int(cycle["cycle_number"]),
            time=np.asarray(cycle["time"], dtype=float),
            current=current,
            voltage=voltage,
            c_max=c_max,
        )
        v50 = float(np.interp(0.5, self.voltage_curve_model.soc_grid, voltage_curve))
        next_curve = self.voltage_curve_model.predict_curve(int(cycle["cycle_number"]) + 1)
        next_v50 = float(np.interp(0.5, self.voltage_curve_model.soc_grid, next_curve))
        drift_rmse = self.voltage_curve_model.drift_rmse(int(cycle["cycle_number"]))

        temp_med = float(np.median(temp))
        active_discharge = np.abs(current[current < -1e-4])
        if active_discharge.size > 0:
            self._latest_discharge_current_a = float(np.median(active_discharge))
        self._latest_temperature_c = temp_med
        self.hybrid_fade_model.update(
            cycle=int(cycle["cycle_number"]),
            capacity_ah=c_max,
            r0_ohm=r0_est,
            temperature_c=temp_med,
        )
        next_capacity = self.hybrid_fade_model.predict_capacity(
            cycle=int(cycle["cycle_number"]) + 1,
            r0_ohm=self.resistance_growth_model.predict(int(cycle["cycle_number"]) + 1),
            temperature_c=temp_med,
        )
        rul_1p4 = self.hybrid_fade_model.predict_rul(
            current_cycle=int(cycle["cycle_number"]),
            resistance_model=self.resistance_growth_model,
            temperature_c=temp_med,
        )

        current_params = self.identifier.interpolate_params(soc=soc_end, T=float(np.median(temp)))
        current_params["R0"] = r0_est

        cycle_number = int(cycle["cycle_number"])
        self.ageing_model.update(cycle_number, {"R0": r0_est, "C_max": c_max})
        rul = self.ageing_model.predict_rul(cycle_number)

        self._latest_cycle_number = cycle_number
        snapshot = TwinSnapshot(
            battery=str(self.selected_battery),
            cycle_number=cycle_number,
            soc=soc_end,
            soh=soh,
            capacity_ah=c_max,
            internal_resistance_ohm=r0_est,
            r0_growth_rate_ohm_per_cycle=self.resistance_growth_model.growth_rate(cycle_number),
            voltage_curve_drift_rmse=drift_rmse,
            voltage_curve_at_50soc=v50,
            predicted_voltage_next_cycle_50soc=next_v50,
            predicted_capacity_next_cycle_ah=float(next_capacity),
            predicted_eol_cycle_1p4ah=float(rul_1p4.eol_cycle_mean),
            rul_1p4ah_mean_cycles=float(rul_1p4.mean_rul_cycles),
            rul_1p4ah_ci_lower_90=float(rul_1p4.ci_lower_90),
            rul_1p4ah_ci_upper_90=float(rul_1p4.ci_upper_90),
            rul_mean_cycles=float(rul["mean_rul"]),
            rul_ci_lower_90=float(rul["ci_lower_90"]),
            rul_ci_upper_90=float(rul["ci_upper_90"]),
            ecm_params={k: float(v) for k, v in current_params.items()},
            voltage_rmse=rmse,
        )
        self._latest_snapshot = snapshot
        return snapshot

    def _fit_ocv_model_from_cycles(self, cycles: list[dict[str, Any]]) -> None:
        if self.nominal_capacity_ah is None:
            raise RuntimeError("Nominal capacity is not set.")

        candidate = min(
            cycles,
            key=lambda cycle: float(np.median(np.abs(np.asarray(cycle["I"], dtype=float)))),
        )

        current = np.asarray(candidate["I"], dtype=float)
        time = np.asarray(candidate["time"], dtype=float)
        voltage = np.asarray(candidate["V"], dtype=float)
        dt = np.diff(time, prepend=time[0])
        if dt.size > 1:
            dt[0] = dt[1]

        discharged_ah = np.cumsum(np.maximum(-current, 0.0) * dt / 3600.0)
        soc = 1.0 - discharged_ah / self.nominal_capacity_ah
        soc = np.clip(soc, 0.0, 1.0)
        self.ocv_model.fit(soc=soc, voltage=voltage)

    @staticmethod
    def _infer_nominal_capacity(cycles: list[dict[str, Any]]) -> float:
        capacities = [float(c["C_max"]) for c in cycles if np.isfinite(float(c.get("C_max", np.nan)))]
        if not capacities:
            return 2.0
        return float(np.median(capacities))

    def _build_theta_history(self, cycles: list[dict[str, Any]]) -> list[dict[str, float]]:
        theta_hist: list[dict[str, float]] = []
        for cycle in cycles:
            current = np.asarray(cycle["I"], dtype=float)
            temp = np.asarray(cycle["T"], dtype=float)
            time = np.asarray(cycle["time"], dtype=float)
            if current.size != time.size:
                continue

            c_max = float(cycle.get("C_max", self.nominal_capacity_ah or 2.0))
            dt = np.diff(time, prepend=time[0])
            if dt.size > 1:
                dt[0] = dt[1]
            discharged_ah = np.cumsum(np.maximum(-current, 0.0) * dt / 3600.0)
            soc = np.clip(1.0 - discharged_ah / max(c_max, 1e-6), 0.0, 1.0)

            soc_ref = float(np.median(soc))
            temp_ref = float(np.median(temp))
            params = self.identifier.interpolate_params(soc=soc_ref, T=temp_ref)
            params["C_max"] = c_max
            theta_hist.append(params)

        if not theta_hist:
            raise RuntimeError("No valid cycles available for theta history.")
        return theta_hist

    def _estimate_r0(self, cycle: dict[str, Any], soc_ref: float) -> float:
        voltage = np.asarray(cycle["V"], dtype=float)
        current = np.asarray(cycle["I"], dtype=float)

        active = np.abs(current) > 1e-3
        if not np.any(active):
            return float(self.identifier.interpolate_params(soc_ref, T=float(np.median(cycle["T"])))["R0"])

        ocv = float(self.ocv_model.ocv(soc_ref, current=0.0))
        r0_inst = (ocv - voltage[active]) / current[active]
        r0_inst = r0_inst[np.isfinite(r0_inst)]
        if r0_inst.size == 0:
            return float(self.identifier.interpolate_params(soc_ref, T=float(np.median(cycle["T"])))["R0"])

        r0 = float(np.median(np.abs(r0_inst)))
        return float(np.clip(r0, 1e-4, 0.5))

    def _estimate_scenario_rul(
        self,
        start_cycle: int,
        start_capacity_ah: float,
        temp_c: float,
        current_a: float,
        scenario: UsageProfile,
        max_horizon_cycles: int = 5000,
    ) -> float:
        if start_capacity_ah <= 1.4:
            return 0.0
        base_current = max(self._latest_discharge_current_a, 1e-6)
        current_ratio = max(0.2, current_a / base_current)
        thermal_stress = max(0.0, temp_c - 25.0) / 30.0
        stress = scenario.resistance_multiplier * (1.0 + thermal_stress + 0.7 * max(0.0, current_ratio - 1.0))

        capacity = float(start_capacity_ah)
        for step in range(1, max_horizon_cycles + 1):
            cycle = start_cycle + step
            r0 = self.resistance_growth_model.predict(cycle) * stress
            model_cap = self.hybrid_fade_model.predict_capacity(cycle=cycle, r0_ohm=r0, temperature_c=temp_c)
            decay_penalty = 0.001 * max(0.0, stress - 1.0)
            capacity = min(capacity, model_cap - decay_penalty)
            if capacity <= 1.4:
                return float(step)
        return float(max_horizon_cycles)
