from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from sklearn.ensemble import GradientBoostingRegressor


def _capacity_law(cycle: np.ndarray, c0: float, beta: float, n_exp: float) -> np.ndarray:
    return c0 * (1.0 - beta * np.power(np.maximum(cycle, 0.0), n_exp))


@dataclass(slots=True)
class RULPrediction:
    mean_rul_cycles: float
    ci_lower_90: float
    ci_upper_90: float
    eol_cycle_mean: float
    eol_cycle_low: float
    eol_cycle_high: float


class InternalResistanceGrowthModel:
    def __init__(self) -> None:
        self._cycles: list[float] = []
        self._r0_values: list[float] = []
        self._coef: np.ndarray | None = None

    def update(self, cycle: float, r0_ohm: float) -> None:
        self._cycles.append(float(cycle))
        self._r0_values.append(float(r0_ohm))
        self._fit()

    def predict(self, cycle: float) -> float:
        if self._coef is None:
            if self._r0_values:
                return float(self._r0_values[-1])
            return 0.015
        return float(np.polyval(self._coef, float(cycle)))

    def growth_rate(self, cycle: float | None = None) -> float:
        if self._coef is None:
            return 0.0
        c = float(self._cycles[-1] if cycle is None and self._cycles else cycle or 0.0)
        deriv = np.polyder(self._coef)
        return float(np.polyval(deriv, c))

    def _fit(self) -> None:
        if len(self._cycles) < 2:
            self._coef = None
            return
        degree = 2 if len(self._cycles) >= 4 else 1
        x = np.asarray(self._cycles, dtype=float)
        y = np.asarray(self._r0_values, dtype=float)
        self._coef = np.polyfit(x, y, deg=degree)


class VoltageCurveEvolutionModel:
    def __init__(self, soc_grid: np.ndarray | None = None) -> None:
        self.soc_grid = np.asarray(soc_grid if soc_grid is not None else np.linspace(0.05, 0.95, 19), dtype=float)
        self._curves: dict[int, np.ndarray] = {}
        self._baseline_cycle: int | None = None
        self._trend_coef: np.ndarray | None = None

    def update(self, cycle_number: int, time: np.ndarray, current: np.ndarray, voltage: np.ndarray, c_max: float) -> np.ndarray:
        curve = self._extract_curve(time=time, current=current, voltage=voltage, c_max=c_max)
        if curve is not None:
            self._curves[int(cycle_number)] = curve
            if self._baseline_cycle is None:
                self._baseline_cycle = int(cycle_number)
            self._fit_trend()
            return curve

        if self._curves:
            last_cycle = max(self._curves)
            return self._curves[last_cycle]
        return np.full_like(self.soc_grid, np.nan, dtype=float)

    def predict_curve(self, cycle_number: int) -> np.ndarray:
        if self._trend_coef is not None:
            slope = self._trend_coef[0, :]
            intercept = self._trend_coef[1, :]
            pred = slope * float(cycle_number) + intercept
            return np.asarray(pred, dtype=float)
        if self._curves:
            return self._curves[max(self._curves)]
        return np.full_like(self.soc_grid, np.nan, dtype=float)

    def drift_rmse(self, cycle_number: int) -> float:
        if self._baseline_cycle is None or self._baseline_cycle not in self._curves:
            return 0.0
        baseline = self._curves[self._baseline_cycle]
        current = self._curves.get(int(cycle_number), self.predict_curve(int(cycle_number)))
        valid = np.isfinite(baseline) & np.isfinite(current)
        if not np.any(valid):
            return 0.0
        return float(np.sqrt(np.mean(np.square(current[valid] - baseline[valid]))))

    def _extract_curve(self, time: np.ndarray, current: np.ndarray, voltage: np.ndarray, c_max: float) -> np.ndarray | None:
        time = np.asarray(time, dtype=float)
        current = np.asarray(current, dtype=float)
        voltage = np.asarray(voltage, dtype=float)
        if time.size < 5 or current.size != time.size or voltage.size != time.size:
            return None

        discharge = current < -1e-4
        if np.count_nonzero(discharge) < 5:
            return None

        td = time[discharge]
        idd = current[discharge]
        vd = voltage[discharge]
        dt = np.diff(td, prepend=td[0])
        if dt.size > 1:
            dt[0] = dt[1]
        discharged_ah = np.cumsum(np.maximum(-idd, 0.0) * dt / 3600.0)
        soc = np.clip(1.0 - discharged_ah / max(c_max, 1e-6), 0.0, 1.0)
        order = np.argsort(soc)
        soc_sorted = soc[order]
        v_sorted = vd[order]
        if np.unique(soc_sorted).size < 5:
            return None
        return np.interp(self.soc_grid, soc_sorted, v_sorted)

    def _fit_trend(self) -> None:
        if len(self._curves) < 3:
            self._trend_coef = None
            return
        cycles = np.asarray(sorted(self._curves), dtype=float)
        curves = np.vstack([self._curves[int(c)] for c in cycles])
        coef_rows = []
        for idx in range(curves.shape[1]):
            y = curves[:, idx]
            valid = np.isfinite(y)
            if np.count_nonzero(valid) < 2:
                fill = float(np.nanmean(y[valid])) if np.any(valid) else 0.0
                coef_rows.append([0.0, fill])
                continue
            coef_rows.append(np.polyfit(cycles[valid], y[valid], deg=1))
        self._trend_coef = np.asarray(coef_rows, dtype=float).T


class HybridCapacityFadeModel:
    def __init__(self, eol_capacity_ah: float = 1.4) -> None:
        self.eol_capacity_ah = float(eol_capacity_ah)
        self._cycles: list[float] = []
        self._capacity: list[float] = []
        self._r0: list[float] = []
        self._temp: list[float] = []
        self._empirical_params: np.ndarray | None = None
        self._residual_std = 0.03
        self._ml: GradientBoostingRegressor | None = None

    def update(self, cycle: float, capacity_ah: float, r0_ohm: float, temperature_c: float) -> None:
        self._cycles.append(float(cycle))
        self._capacity.append(float(capacity_ah))
        self._r0.append(float(r0_ohm))
        self._temp.append(float(temperature_c))
        self._fit()

    def predict_capacity(self, cycle: float, r0_ohm: float, temperature_c: float) -> float:
        empirical = self._predict_empirical(float(cycle))
        if self._ml is None:
            return empirical
        x = np.array([[float(cycle), float(r0_ohm), float(temperature_c), empirical]], dtype=float)
        residual = float(self._ml.predict(x)[0])
        return empirical + residual

    def predict_rul(
        self,
        current_cycle: int,
        resistance_model: InternalResistanceGrowthModel,
        temperature_c: float,
        max_horizon_cycles: int = 5000,
    ) -> RULPrediction:
        cycles = np.arange(current_cycle, current_cycle + max_horizon_cycles + 1, dtype=float)
        r0_forecast = np.array([resistance_model.predict(c) for c in cycles], dtype=float)
        cap = np.array(
            [self.predict_capacity(cycle=c, r0_ohm=r0, temperature_c=temperature_c) for c, r0 in zip(cycles, r0_forecast)],
            dtype=float,
        )

        hit = np.where(cap <= self.eol_capacity_ah)[0]
        mean_eol = float(cycles[hit[0]]) if hit.size else float(cycles[-1])

        cap_low = cap - 1.64 * self._residual_std
        cap_high = cap + 1.64 * self._residual_std
        hit_low = np.where(cap_low <= self.eol_capacity_ah)[0]
        hit_high = np.where(cap_high <= self.eol_capacity_ah)[0]
        eol_low = float(cycles[hit_low[0]]) if hit_low.size else float(cycles[-1])
        eol_high = float(cycles[hit_high[0]]) if hit_high.size else float(cycles[-1])

        rul_mean = max(0.0, mean_eol - current_cycle)
        rul_low = max(0.0, eol_low - current_cycle)
        rul_high = max(0.0, eol_high - current_cycle)
        lo = min(rul_low, rul_high)
        hi = max(rul_low, rul_high)

        return RULPrediction(
            mean_rul_cycles=float(rul_mean),
            ci_lower_90=float(lo),
            ci_upper_90=float(hi),
            eol_cycle_mean=float(mean_eol),
            eol_cycle_low=float(min(eol_low, eol_high)),
            eol_cycle_high=float(max(eol_low, eol_high)),
        )

    def _fit(self) -> None:
        x = np.asarray(self._cycles, dtype=float)
        y = np.asarray(self._capacity, dtype=float)
        if x.size < 2:
            self._empirical_params = np.array([max(y[0], 2.0), 0.0, 0.6], dtype=float)
            self._ml = None
            self._residual_std = 0.03
            return

        c0_guess = max(y[0], np.max(y))
        p0 = [c0_guess, 1e-3, 0.6]
        bounds = ([0.5, 0.0, 0.2], [5.0, 0.3, 1.2])
        try:
            self._empirical_params, _ = curve_fit(_capacity_law, x, y, p0=p0, bounds=bounds, maxfev=20_000)
        except RuntimeError:
            self._empirical_params = np.asarray(p0, dtype=float)

        empirical = _capacity_law(x, *self._empirical_params)
        residual = y - empirical
        self._residual_std = float(np.std(residual)) if residual.size > 1 else 0.03

        if x.size >= 6:
            features = np.column_stack(
                [
                    x,
                    np.asarray(self._r0, dtype=float),
                    np.asarray(self._temp, dtype=float),
                    empirical,
                ]
            )
            self._ml = GradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=3,
                n_estimators=200,
                random_state=42,
            )
            self._ml.fit(features, residual)
        else:
            self._ml = None

    def _predict_empirical(self, cycle: float) -> float:
        if self._empirical_params is None:
            if self._capacity:
                return float(self._capacity[-1])
            return 2.0
        return float(_capacity_law(np.array([float(cycle)]), *self._empirical_params)[0])
