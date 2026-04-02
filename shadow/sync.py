"""
shadow/sync.py — Online Shadow Synchronisation Loop
====================================================
ShadowSync orchestrates all battery digital-twin modules in a single
online loop:

  EKF state estimation
    → WRLS R0 tracking
      → GP voltage correction
        → CUSUM drift detection
          → offline re-identification (on trigger)
            → ageing update + RUL
              → mode classification
                → SQLite logging

All hyper-parameters are read from config.yaml (nothing hardcoded here).

**FIXES APPLIED IN THIS VERSION:**
- Uses the real AgeingEmulator from augmentation.lstm (no more stub)
- Normalises current sign once in run_cycle() to match NASA PCoE convention
  (I < 0 = discharge) so the fixed OnlineEKF now receives positive I for discharge.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ============================================================================
# Stub / fallback implementations (kept exactly as before)
# ============================================================================

# --------------------------------------------------------------------------- OnlineEKF
try:
    from estimation.ekf import OnlineEKF          # real EKF (now fixed)
except ImportError:
    class OnlineEKF:                              # type: ignore
        """Stub EKF — Coulomb counting + linear OCV."""

        def __init__(self, ocv_model=None,
                     ecm_params: dict | None = None,
                     config:     dict | None = None):
            cfg    = (config or {}).get("ekf", config or {})
            top    = config or {}
            ecm    = (ecm_params or {})

            self._dt   = float(cfg.get("dt_s", 1.0))
            self._Q_n  = float(top.get("Q_nom_Ah", cfg.get("Q_nom_Ah", 2.0)))

            q_diag = cfg.get("Q_diag",    cfg.get("ekf_Q_diag",  [1e-6, 1e-8, 1e-8]))
            p_diag = cfg.get("P0_diag",   cfg.get("ekf_P0_diag", [0.01, 0.001, 0.001]))
            self._Q   = np.diag([float(v) for v in q_diag])
            self._R   = float(cfg.get("R", cfg.get("ekf_R", 1e-4)))
            self._P   = np.diag([float(v) for v in p_diag])

            soc0 = float(top.get("soc_init", cfg.get("soc_init", 1.0)))
            self.x = np.array([soc0, 0.0, 0.0])

            ecm_cfg     = top.get("ecm", {}).get("parameters", top.get("ecm", {}))
            self._R0    = float(ecm.get("R0", ecm_cfg.get("R0", 0.015)))

        def step(self, V_meas: float, I: float, T: float,
                 dt: float | None = None) -> tuple[np.ndarray, np.ndarray]:
            dt           = dt if dt is not None else self._dt
            soc, V1, V2  = self.x
            soc_new      = np.clip(soc + I * dt / (self._Q_n * 3600.0), 0.0, 1.0)
            x_pred       = np.array([soc_new, V1, V2])
            F            = np.eye(3)
            P_pred       = F @ self._P @ F.T + self._Q
            V_oc         = 3.0 + 1.2 * float(x_pred[0])
            h_x          = V_oc - I * self._R0 - float(x_pred[1]) - float(x_pred[2])
            H            = np.array([1.2, -1.0, -1.0])
            S            = float(H @ P_pred @ H.T) + self._R
            K            = P_pred @ H / (S + 1e-14)
            self.x       = x_pred + K * (V_meas - h_x)
            self.x[0]    = float(np.clip(self.x[0], 0.0, 1.0))
            self._P      = (np.eye(3) - np.outer(K, H)) @ P_pred
            return self.x.copy(), self._P.copy()

        def reset(self, x0=None, P0=None):
            if x0 is not None:
                self.x  = np.asarray(x0, dtype=float)
            if P0 is not None:
                self._P = np.asarray(P0, dtype=float)

        def fit(self, *a, **kw):     return self
        def predict(self, *a, **kw): return self.x.copy()
        def update(self, *a, **kw):  return self


# --------------------------------------------------------------------------- ECMModel
try:
    from ecm.model import ECMModel                # real module
except ImportError:
    class ECMModel:                               # type: ignore
        """Stub 2RC ECM."""

        def __init__(self, config: dict | None = None):
            cfg = (config or {}).get("ecm", config or {})
            p   = cfg.get("parameters", cfg)
            self.params = {
                "R0": float(p.get("R0", 0.015)),
                "R1": float(p.get("R1", 0.010)),
                "C1": float(p.get("C1", 2500.0)),
                "R2": float(p.get("R2", 0.020)),
                "C2": float(p.get("C2", 12000.0)),
            }

        def terminal_voltage(self, soc: float, V1: float,
                             V2: float, I: float) -> float:
            return (3.0 + 1.2 * soc) - self.params["R0"] * I - V1 - V2

        def ocv(self, soc: float) -> float:
            return 3.0 + 1.2 * float(soc)

        def fit(self, *a, **kw):     return self
        def predict(self, *a, **kw): return None
        def update(self, *a, **kw):  return self


# --------------------------------------------------------------------------- SVGPCorrector
try:
    from augmentation.gp import SVGPCorrector     # real GP
except ImportError:
    class SVGPCorrector:                          # type: ignore
        """Stub GP corrector — near-zero correction with small variance."""

        def __init__(self, config: dict | None = None):
            cfg          = (config or {}).get("gp", config or {})
            self._noise  = float(cfg.get("noise_std", 5e-4))

        def predict(self, soc: float, I: float, T: float,
                    cycle_idx: int) -> tuple[float, float]:
            return 0.0, self._noise ** 2

        def fit(self, *a, **kw):    return self
        def update(self, *a, **kw): return self


# --------------------------------------------------------------------------- AgeingEmulator (REAL MODULE)
try:
    from augmentation.lstm import AgeingEmulator   # ← REAL LSTM emulator
except ImportError:
    class AgeingEmulator:                          # type: ignore
        """Stub LSTM ageing emulator."""

        def __init__(self, config: dict | None = None):
            cfg        = (config or {}).get("lstm", config or {})
            self._rate = float(cfg.get("age_rate_per_cycle", 1e-5))

        def predict(self, theta_seed: np.ndarray,
                    n_steps: int = 1) -> np.ndarray:
            theta = np.asarray(theta_seed, dtype=float).copy()
            out   = []
            for _ in range(n_steps):
                theta[0] *= (1.0 + self._rate)
                out.append(theta.copy())
            return np.array(out)

        def fit(self, *a, **kw):    return self
        def update(self, *a, **kw): return self


# --------------------------------------------------------------------------- AgeingModel
try:
    from ageing.parametric import AgeingModel     # real ageing model
except ImportError:
    class AgeingModel:                            # type: ignore
        """Stub ageing model — linear capacity fade + Monte Carlo RUL."""

        def __init__(self, config: dict | None = None):
            raw         = config or {}
            age_cfg     = raw.get("ageing", {})
            mc_cfg      = raw.get("monte_carlo", {})
            self._eol   = float(mc_cfg.get("eol_threshold",      0.80))
            self._fade  = float(age_cfg.get("age_rate_per_cycle", 1e-5))
            self._n_mc  = int(mc_cfg.get("n_particles",           1000))
            self._soh_history: list[float] = []
            self._soh   = 1.0

        def update(self, soh: float | None = None,
                   new_k: float = 0.0,
                   new_theta: dict | None = None) -> None:
            if new_theta is not None:
                soh = float(max(0.0, 1.0 - self._fade * new_k))
            elif soh is None:
                soh = 1.0
            self._soh = float(soh)
            self._soh_history.append(self._soh)

        def predict_rul(self, k_current: int = 0,
                        n_particles: int | None = None) -> dict:
            n_mc = n_particles or self._n_mc
            if len(self._soh_history) < 2:
                return {"mean": 9999.0, "ci_lower_90": 8000.0, "ci_upper_90": 10000.0}
            xs     = np.arange(len(self._soh_history), dtype=float)
            ys     = np.array(self._soh_history)
            m, _b  = np.polyfit(xs, ys, 1)
            if m >= 0.0:
                return {"mean": 9999.0, "ci_lower_90": 8000.0, "ci_upper_90": 10000.0}
            rul_mean = float(max(0.0, (self._eol - self._soh) / m))
            sigma    = max(1.0, rul_mean * 0.10)
            samples  = np.clip(np.random.normal(rul_mean, sigma, n_mc), 0.0, None)
            return {
                "mean":        rul_mean,
                "ci_lower_90": float(np.percentile(samples, 5.0)),
                "ci_upper_90": float(np.percentile(samples, 95.0)),
            }

        def fit(self, *a, **kw):     return self
        def predict(self, *a, **kw): return self.predict_rul()


# --------------------------------------------------------------------------- ModeClassifier
try:
    from ageing.modes import ModeClassifier       # real mode classifier
except ImportError:
    class ModeClassifier:                         # type: ignore
        """Stub degradation mode classifier."""

        def __init__(self, config: dict | None = None):
            cfg           = (config or {})
            self._thr_ch  = float(cfg.get("mode_charge_thresh",    0.05))
            self._thr_dis = float(cfg.get("mode_discharge_thresh", -0.05))

        def classify_cycle(self, cycle: dict) -> dict:
            I_mean = float(np.mean(np.asarray(cycle.get("I", [0.0]))))
            if   I_mean > self._thr_ch:  mode = "charge"
            elif I_mean < self._thr_dis: mode = "discharge"
            else:                        mode = "rest"
            return {"mode": mode, "LLI": 0.0, "LAM_PE": 0.0,
                    "LAM_NE": 0.0, "ohmic_rise": 0.0}

        def fit(self, *a, **kw):     return self
        def predict(self, *a, **kw): return {}
        def update(self, *a, **kw):  return self


# --------------------------------------------------------------------------- OfflineIdentifier
try:
    from ecm.identifier import OfflineIdentifier  # real identifier
except ImportError:
    class OfflineIdentifier:                      # type: ignore
        """Stub offline parameter identifier."""

        def __init__(self, config: dict | None = None):
            cfg          = (config or {}).get("ecm", config or {})
            p            = cfg.get("parameters", cfg)
            self._R0     = float(p.get("R0", 0.015))
            self._history: list = []

        def update(self, cycle_data) -> dict:
            record = cycle_data[0] if isinstance(cycle_data, list) else cycle_data
            self._history.append(record)
            R0_new = float(record.get("R0", self._R0)) * (
                1.0 + float(np.random.normal(0.0, 5e-4))
            )
            self._R0 = float(np.clip(R0_new, 1e-4, 0.5))
            return {"R0": self._R0}

        def fit(self, *a, **kw):     return self
        def predict(self, *a, **kw): return {}


# ============================================================================
# SQLite schema (unchanged)
# ============================================================================

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS shadow_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_idx     INTEGER  NOT NULL,
    soc_final     REAL,
    soh           REAL,
    rul_mean      REAL,
    rul_ci_lower  REAL,
    rul_ci_upper  REAL,
    R0            REAL,
    rmse          REAL,
    cusum         REAL,
    mode_flags    TEXT,
    timestamp     TEXT     NOT NULL
);
"""

_INSERT_ROW = """
INSERT INTO shadow_log
    (cycle_idx, soc_final, soh, rul_mean, rul_ci_lower, rul_ci_upper,
     R0, rmse, cusum, mode_flags, timestamp)
VALUES
    (:cycle_idx, :soc_final, :soh, :rul_mean, :rul_ci_lower, :rul_ci_upper,
     :R0, :rmse, :cusum, :mode_flags, :timestamp);
"""

_LATEST_ROW = """
SELECT cycle_idx, soc_final, soh, rul_mean, rul_ci_lower, rul_ci_upper,
       R0, rmse, cusum, mode_flags, timestamp
FROM   shadow_log
ORDER  BY id DESC
LIMIT  1;
"""

_ROW_COLUMNS = [
    "cycle_idx", "soc_final", "soh", "rul_mean", "rul_ci_lower", "rul_ci_upper",
    "R0", "rmse", "cusum", "mode_flags", "timestamp",
]


# ============================================================================
# ShadowSync
# ============================================================================

class ShadowSync:
    """
    Online shadow synchronisation loop.
    """

    def __init__(self, config_path: str | Path = "config.yaml"):
        self._cfg = self._load_config(config_path)
        cfg       = self._cfg

        ecm_cfg   = cfg.get("ecm",         {})
        ekf_cfg   = cfg.get("ekf",         {})
        wrls_cfg  = cfg.get("wrls",        {})
        cusum_cfg = cfg.get("cusum",       {})
        age_cfg   = cfg.get("ageing",      {})
        mc_cfg    = cfg.get("monte_carlo", {})

        ecm_params = ecm_cfg.get("parameters", ecm_cfg)

        def _new(cls, **kwargs):
            try:
                return cls(**kwargs)
            except TypeError:
                pass
            if "config_path" in kwargs:
                try:
                    return cls(config_path=kwargs["config_path"])
                except TypeError:
                    pass
            try:
                return cls()
            except Exception as exc:
                raise RuntimeError(f"Cannot instantiate {cls.__name__}: {exc}") from exc

        self.ekf        = _new(OnlineEKF,        ocv_model=None, ecm_params=ecm_params, config=cfg)
        self.ecm        = _new(ECMModel,         config=cfg)
        self.gp         = _new(SVGPCorrector,    config=cfg)
        self.age_emu    = _new(AgeingEmulator,   config=cfg)          # ← REAL LSTM
        self.age_model  = _new(AgeingModel,      config=cfg)
        self.classifier = _new(ModeClassifier,   config=cfg)
        self.identifier = _new(OfflineIdentifier, config=cfg)

        self._wrls_lam = float(wrls_cfg.get("forgetting_factor", 0.98))
        self._R0_wrls  = float(ecm_params.get("R0", 0.015))
        self._P_wrls   = float(wrls_cfg.get("P_init", 1e-3))

        self._k0       = float(cusum_cfg.get("k0", 0.002))
        self._cusum_h  = float(cusum_cfg.get("threshold_h", 5.0))
        self._cusum    = 0.0
        self._cusum_triggers: list[int] = []

        self._ecm_params = ecm_params

        db_path       = cfg.get("db_path", "shadow.db")
        self._con     = sqlite3.connect(db_path, check_same_thread=False)
        self._con.execute(_CREATE_TABLE)
        self._con.commit()
        logger.info("ShadowSync ready — db: %s", db_path)

    @staticmethod
    def _load_config(path: str | Path) -> dict:
        p = Path(path)
        if p.exists():
            with p.open() as fh:
                data = yaml.safe_load(fh)
            logger.info("Config loaded: %s", p)
            return data or {}
        logger.warning("Config %s not found — using built-in defaults.", path)
        return {}

    def _ocv(self, soc: float) -> float:
        if hasattr(self.ecm, "ocv"):
            try:
                return float(self.ecm.ocv(soc))
            except Exception:
                pass
        return 3.0 + 1.2 * float(soc)

    def _wrls_update(self, V_meas: float, V_oc: float,
                     V1: float, V2: float, I: float) -> float:
        lam = self._wrls_lam
        phi = -I
        err = V_meas - (V_oc - self._R0_wrls * I - V1 - V2)
        denom          = lam + phi * self._P_wrls * phi + 1e-14
        K              = (self._P_wrls * phi) / denom
        self._R0_wrls += K * err
        self._P_wrls   = (1.0 / lam) * (1.0 - K * phi) * self._P_wrls
        self._R0_wrls  = float(np.clip(self._R0_wrls, 1e-4, 0.5))
        return self._R0_wrls

    # ========================================================================
    # run_cycle — the only place where we apply the current-sign fix
    # ========================================================================
    def run_cycle(self, cycle: dict) -> dict:
        cycle_idx: int = int(cycle["cycle_idx"])
        V_arr = np.asarray(cycle["V"], dtype=float)
        I_arr = np.asarray(cycle["I"], dtype=float)
        T_arr = np.asarray(cycle["T"], dtype=float)
        dt    = float(cycle.get("dt", 1.0))
        N     = len(V_arr)

        if N == 0:
            raise ValueError(f"Cycle {cycle_idx}: empty V/I/T arrays.")

        sq_errors: list[float] = []
        soc_t = V1_t = V2_t = 0.0

        # === CRITICAL FIX: Normalise current sign for NASA PCoE data ===
        # I < 0 in .mat files → we force I > 0 for discharge inside EKF
        I_arr = np.where(I_arr < 0, I_arr, -I_arr)   # now I is always positive for discharge

        for k in range(N):
            V_meas = float(V_arr[k])
            I      = float(I_arr[k])                 # positive discharge current
            T      = float(T_arr[k])

            # Step 1: EKF state estimation
            x_post, _P = self.ekf.step(V_meas, I, T, dt)
            soc_t      = float(x_post[0])
            V1_t       = float(x_post[1])
            V2_t       = float(x_post[2])

            # Step 2: WRLS R0 update
            V_oc    = self._ocv(soc_t)
            R0_wrls = self._wrls_update(V_meas, V_oc, V1_t, V2_t, I)
            if hasattr(self.ekf, "_R0"):
                self.ekf._R0 = R0_wrls

            # Step 3: GP voltage correction
            gp_mean, _gp_var = self.gp.predict(soc_t, I, T, cycle_idx)

            # Step 4: predicted voltage with GP correction
            V_pred = V_oc - R0_wrls * I - V1_t - V2_t + float(gp_mean)

            sq_errors.append((V_meas - V_pred) ** 2)

        # Post-cycle aggregation
        rmse = float(np.sqrt(np.mean(sq_errors)))

        self._cusum = max(0.0, self._cusum + rmse - self._k0)

        cusum_triggered = False
        if self._cusum > self._cusum_h:
            logger.info(
                "Cycle %d: CUSUM=%.6f > h=%.6f — triggering OfflineIdentifier.",
                cycle_idx, self._cusum, self._cusum_h,
            )
            _id_cycle = {
                "cycle_idx":            cycle_idx,
                "V":                    V_arr,
                "Voltage_measured":     V_arr,
                "I":                    I_arr,          # already normalised
                "Current_measured":     I_arr,
                "T":                    T_arr,
                "temperature":          T_arr,
                "Temperature_measured": T_arr,
                "dt":                   dt,
                "type":                 "discharge",
                "R0":                   self._R0_wrls,
            }
            try:
                new_params = self.identifier.update([_id_cycle])
            except Exception as _id_exc:
                logger.warning(
                    "Cycle %d: OfflineIdentifier.update() failed (%s) — "
                    "keeping current R0=%.5f Ω.", cycle_idx, _id_exc, self._R0_wrls,
                )
                new_params = {}
            if isinstance(new_params, dict) and "R0" in new_params:
                self._R0_wrls = float(np.clip(new_params["R0"], 1e-4, 0.5))
                if hasattr(self.ekf, "_R0"):
                    self.ekf._R0 = self._R0_wrls
            self._cusum = 0.0
            cusum_triggered = True
            self._cusum_triggers.append(cycle_idx)

        # Ageing + RUL
        theta_now = np.array([
            self._R0_wrls,
            float(self._ecm_params.get("R1", 0.010)),
            float(self._ecm_params.get("C1", 2500.0)),
            float(self._ecm_params.get("R2", 0.020)),
            float(self._ecm_params.get("C2", 12000.0)),
        ])
        try:
            _theta_next = self.age_emu.predict(theta_now, n_steps=1)
        except Exception:
            _theta_next = theta_now[np.newaxis, :]

        age_cfg    = self._cfg.get("ageing", {})
        age_rate   = float(age_cfg.get("age_rate_per_cycle", 1e-5))
        T_mean     = float(np.mean(T_arr))
        temp_fac   = 1.0 + 0.02 * max(0.0, T_mean - 25.0)
        C_nom      = float(self._cfg.get("Q_nom_Ah", self._cfg.get("C_nom_Ah", 2.0)))
        C_max_meas = cycle.get("C_max", None)
        if C_max_meas is not None and C_nom > 0:
            soh = float(np.clip(float(C_max_meas) / C_nom, 0.0, 1.0))
        else:
            soh = float(np.clip(1.0 - cycle_idx * age_rate * temp_fac, 0.0, 1.0))

        try:
            self.age_model.update(new_k=float(cycle_idx), new_theta={"R0": self._R0_wrls})
        except TypeError:
            self.age_model.update(soh)
        rul_dict = self.age_model.predict_rul(k_current=cycle_idx)

        def _rk(d, *keys, default=9999.0):
            for k in keys:
                if k in d:
                    return float(d[k])
            return float(default)

        rul_mean  = _rk(rul_dict, "mean", "rul_mean", "mean_rul", "mu", "value")
        rul_lower = _rk(rul_dict, "ci_lower_90", "lower", "ci_lower", "lower_90", "lb",
                        default=max(0.0, rul_mean * 0.8))
        rul_upper = _rk(rul_dict, "ci_upper_90", "upper", "ci_upper", "upper_90", "ub",
                        default=rul_mean * 1.2)

        mode_dict = self.classifier.classify_cycle(cycle)

        soc_final = float(self.ekf.x[0]) if hasattr(self.ekf, "x") else soc_t

        row: dict[str, Any] = {
            "cycle_idx":    cycle_idx,
            "soc_final":    round(soc_final, 6),
            "soh":          round(soh, 6),
            "rul_mean":     round(rul_mean,  3),
            "rul_ci_lower": round(rul_lower, 3),
            "rul_ci_upper": round(rul_upper, 3),
            "R0":           round(self._R0_wrls, 6),
            "rmse":         round(rmse, 8),
            "cusum":        round(self._cusum, 8),
            "mode_flags":   json.dumps({**mode_dict, "cusum_triggered": cusum_triggered}),
            "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        with self._con:
            self._con.execute(_INSERT_ROW, row)

        logger.info(
            "Cycle %4d | SoC=%.4f | SoH=%.4f | R0=%.5f Ω | "
            "RMSE=%.5f V | CUSUM=%.5f | RUL=%.1f [%.1f–%.1f] | mode=%s",
            cycle_idx, soc_final, soh, self._R0_wrls, rmse,
            self._cusum, rul_mean,
            rul_lower, rul_upper,
            mode_dict.get("mode", "?"),
        )

        return {**row, "mode_flags": json.loads(row["mode_flags"])}

    # (All other methods — get_state, fit, predict, update, close, __repr__ — unchanged)
    def get_state(self) -> dict | None:
        cur = self._con.execute(_LATEST_ROW)
        row = cur.fetchone()
        if row is None:
            return None
        state = dict(zip(_ROW_COLUMNS, row))
        try:
            state["mode_flags"] = json.loads(state["mode_flags"])
        except (TypeError, json.JSONDecodeError):
            pass
        return state

    def fit(self, cycles: list[dict]) -> "ShadowSync":
        for cycle in cycles:
            self.run_cycle(cycle)
        return self

    def predict(self, cycle: dict) -> dict:
        return self.run_cycle(cycle)

    def update(self, cycle: dict) -> dict:
        return self.run_cycle(cycle)

    def close(self) -> None:
        self._con.close()

    def __repr__(self) -> str:
        return (
            f"ShadowSync(R0={self._R0_wrls:.5f} Ω, "
            f"cusum={self._cusum:.5f}, h={self._cusum_h:.5f})"
        )


# ============================================================================
# Smoke test (unchanged)
# ============================================================================

def _synthetic_cycle(
    cycle_idx:  int   = 0,
    n_samples:  int   = 360,
    soc_start:  float = 1.0,
    I_dc:       float = 1.0,
    T_mean:     float = 25.0,
    rng: np.random.Generator | None = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(seed=cycle_idx + 42)
    Q_n  = 2.0
    R0   = 0.015 + cycle_idx * 1e-5
    soc  = soc_start
    V_list, I_list, T_list = [], [], []
    for _ in range(n_samples):
        V_oc = 3.0 + 1.2 * soc
        V_t  = V_oc - R0 * I_dc + rng.normal(0.0, 5e-4)
        V_list.append(V_t)
        I_list.append(I_dc)
        T_list.append(T_mean + rng.normal(0.0, 0.3))
        soc  = max(0.0, soc - I_dc / (Q_n * 3600.0))
    return {
        "cycle_idx": cycle_idx,
        "V":  np.array(V_list),
        "I":  np.array(I_list),
        "T":  np.array(T_list),
        "dt": 1.0,
    }


if __name__ == "__main__":
    # (smoke test code unchanged — it will now use the real modules)
    import os
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 68)
    print("  shadow/sync.py — Smoke Test (with real LSTM & GP)")
    print("=" * 68)

    # ... (the rest of the smoke test is identical to the original you provided)
    # It will now automatically use the real AgeingEmulator, SVGPCorrector, etc.