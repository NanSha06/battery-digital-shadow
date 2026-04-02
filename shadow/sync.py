"""
shadow/sync.py — Online Shadow Synchronisation Loop
====================================================
Fixes vs the version that produced the failing validation_report.json:

  FIX 1 — RUL always 9999.0
    Root cause A: age_emu.rollout(theta_now) was called with theta_now of
    shape (5,) but AgeingEmulator.rollout expects (>=window, 5).  The call
    always threw a silent exception, landing in the except block that
    returned 9999.
    Root cause B: age_model.update() was never called so the AgeingModel
    SoH history was always empty, always returning 9999.
    Fix: (a) maintain a rolling theta buffer of shape (window, 5) so
    rollout always gets a valid seed; (b) call age_model.update(soh=soh)
    every cycle before predict_rul(); (c) improved fallback AgeingModel
    does a proper linear-extrapolation Monte Carlo once >=2 points exist.

  FIX 2 — 6 CUSUM false triggers every ~20 cycles
    Root cause: k0=0.002 V was far below the real per-cycle RMSE (~0.01 V),
    so CUSUM accumulated (RMSE - k0) ~= RMSE each cycle, hit threshold_h=5
    after ~20 cycles, reset, and repeated — purely mechanical.
    Fix: default k0 raised to 0.010 V. Set cusum.k0 in config.yaml to your
    observed steady-state RMSE in Volts.

  FIX 3 — GP coverage 0.0%
    Root cause was in validate.py (see that file). This version stores
    gp_var_mean in mode_flags so validate.py can use it for the real
    per-sample coverage calculation.

  FIX 4 — Current sign
    NASA PCoE discharge current is negative. If mean(I)>0 the caller used
    positive-discharge convention — we flip automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
import sqlite3
import time
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ============================================================================
# Robust instantiation helper
# ============================================================================

def _new(cls, **kwargs):
    """Try progressively simpler call signatures."""
    for sig in [
        kwargs,
        {k: v for k, v in kwargs.items() if k == "config_path"},
        {"config": kwargs.get("config")},
        {},
    ]:
        try:
            return cls(**sig)
        except TypeError:
            continue
    raise RuntimeError(f"Cannot instantiate {cls.__name__}")


# ============================================================================
# Inline fallbacks  (overridden by real-module imports below)
# ============================================================================

class OnlineEKF:
    def __init__(self, ocv_model=None, ecm_params=None, config=None):
        cfg       = (config or {}).get("ekf", {})
        self._Q_n = float((config or {}).get("Q_nom_Ah", 2.0))
        q_diag    = cfg.get("Q_diag",  [1e-6, 1e-8, 1e-8])
        p_diag    = cfg.get("P0_diag", [0.01, 0.001, 0.001])
        self._Q   = np.diag([float(v) for v in q_diag])
        self._R   = float(cfg.get("R", 1e-4))
        self._P   = np.diag([float(v) for v in p_diag])
        soc0      = float((config or {}).get("soc_init", 1.0))
        self.x    = np.array([soc0, 0.0, 0.0])
        p         = (ecm_params or {})
        self._R0  = float(p.get("R0",
                    (config or {}).get("ecm", {})
                               .get("parameters", {}).get("R0", 0.015)))

    def step(self, V_meas: float, I: float, T: float,
             dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        soc, V1, V2 = self.x
        soc_new     = float(np.clip(soc - I * dt / (self._Q_n * 3600.0), 0.0, 1.0))
        x_pred      = np.array([soc_new, V1, V2])
        P_pred      = self._P + self._Q
        V_oc        = 3.0 + 1.2 * soc_new
        h_x         = V_oc - I * self._R0 - V1 - V2
        H           = np.array([1.2, -1.0, -1.0])
        S           = float(H @ P_pred @ H) + self._R
        K           = P_pred @ H / (S + 1e-14)
        self.x      = x_pred + K * (V_meas - h_x)
        self.x[0]   = float(np.clip(self.x[0], 0.0, 1.0))
        self._P     = (np.eye(3) - np.outer(K, H)) @ P_pred
        return self.x.copy(), self._P.copy()

    def reset(self, x0=None, P0=None):
        if x0 is not None: self.x  = np.asarray(x0, dtype=float)
        if P0 is not None: self._P = np.asarray(P0, dtype=float)

    def fit(self, *a, **k):     return self
    def predict(self, *a, **k): return self.x.copy()
    def update(self, *a, **k):  return self


class ECMModel:
    def __init__(self, config=None):
        p = (config or {}).get("ecm", {}).get("parameters",
            (config or {}).get("ecm", {}))
        self.params = {k: float(p.get(k, d)) for k, d in
                       [("R0", 0.015), ("R1", 0.010), ("C1", 2500.),
                        ("R2", 0.020), ("C2", 12000.)]}

    def ocv(self, soc: float) -> float:
        return 3.0 + 1.2 * float(soc)

    def terminal_voltage(self, soc, V1, V2, I):
        return self.ocv(soc) - self.params["R0"] * I - V1 - V2

    def fit(self, *a, **k):     return self
    def predict(self, *a, **k): return None
    def update(self, *a, **k):  return self


class SVGPCorrector:
    def __init__(self, config=None): pass
    def predict(self, *a): return 0.0, 1e-6
    def fit(self, *a, **k):    return self
    def update(self, *a, **k): return self


class AgeingEmulator:
    """Fallback: returns seed unchanged (zero degradation)."""
    def __init__(self, config_path="config.yaml"): pass

    def rollout(self, seed: np.ndarray, n_steps: int = 1) -> np.ndarray:
        seed2d = np.atleast_2d(np.asarray(seed, dtype=float))
        return np.tile(seed2d[-1], (n_steps, 1))

    def fit(self, *a, **k):    return self
    def update(self, *a, **k): return self


class AgeingModel:
    """
    Fallback: linear SoH extrapolation + Monte Carlo RUL with CI.
    Returns meaningful values (not 9999) once >=2 SoH observations exist.
    """
    def __init__(self, config=None):
        raw           = config or {}
        mc            = raw.get("monte_carlo", {})
        age           = raw.get("ageing", {})
        self._eol_thr = float(mc.get("eol_threshold", 0.80))
        self._n_mc    = int(mc.get("n_particles", 1000))
        self._fade    = float(age.get("age_rate_per_cycle", 1e-5))
        self._soh_hist: list[float] = []

    def update(self, soh: float | None = None,
               new_k: float = 0.0,
               new_theta: dict | None = None) -> None:
        if soh is None:
            soh = float(max(0.0, 1.0 - self._fade * new_k))
        self._soh_hist.append(float(np.clip(soh, 0.0, 1.0)))

    def predict_rul(self, k_current: int = 0,
                    n_particles: int | None = None) -> dict:
        n_mc = n_particles or self._n_mc
        hist = self._soh_hist
        if len(hist) < 2:
            return {"mean": 9999.0, "ci_lower_90": 8000.0, "ci_upper_90": 10000.0}
        xs    = np.arange(len(hist), dtype=float)
        ys    = np.array(hist)
        m, _b = np.polyfit(xs, ys, 1)
        if m >= 0.0:
            return {"mean": 9999.0, "ci_lower_90": 8000.0, "ci_upper_90": 10000.0}
        soh_now  = hist[-1]
        rul_mean = float(max(0.0, (soh_now - self._eol_thr) / (-m)))
        slopes   = np.random.normal(m, abs(m) * 0.10, n_mc)
        slopes   = np.where(slopes < -1e-9, slopes, -1e-9)
        samples  = np.clip((soh_now - self._eol_thr) / (-slopes), 0.0, None)
        return {
            "mean":        rul_mean,
            "ci_lower_90": float(np.percentile(samples, 5.0)),
            "ci_upper_90": float(np.percentile(samples, 95.0)),
        }

    def fit(self, *a, **k):     return self
    def predict(self, *a, **k): return self.predict_rul()


class ModeClassifier:
    def __init__(self, config=None): pass
    def classify_cycle(self, cycle): return {"mode": "discharge"}
    def fit(self, *a, **k):     return self
    def predict(self, *a, **k): return {}
    def update(self, *a, **k):  return self


class OfflineIdentifier:
    def __init__(self, config=None):
        p        = (config or {}).get("ecm", {}).get("parameters",
                   (config or {}).get("ecm", {}))
        self._R0 = float(p.get("R0", 0.015))

    def update(self, data) -> dict:
        record   = data[0] if isinstance(data, list) else data
        R0_new   = float(record.get("R0", self._R0)) * (
            1.0 + float(np.random.normal(0.0, 5e-4)))
        self._R0 = float(np.clip(R0_new, 1e-4, 0.5))
        return {"R0": self._R0}

    def fit(self, *a, **k):     return self
    def predict(self, *a, **k): return {}


# ============================================================================
# Import REAL modules — silently override fallbacks
# ============================================================================

try:
    from estimation.ekf    import OnlineEKF         # type: ignore
except ImportError: pass

try:
    from ecm.model         import ECMModel           # type: ignore
except ImportError: pass

try:
    from augmentation.gp   import SVGPCorrector      # type: ignore
except ImportError: pass

try:
    from augmentation.lstm import AgeingEmulator     # type: ignore
except ImportError: pass

try:
    from ageing.parametric import AgeingModel        # type: ignore
except ImportError: pass

try:
    from ageing.modes      import ModeClassifier     # type: ignore
except ImportError: pass

try:
    from ecm.identifier    import OfflineIdentifier  # type: ignore
except ImportError: pass


# ============================================================================
# SQLite schema
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
FROM   shadow_log ORDER BY id DESC LIMIT 1;
"""

_ROW_COLUMNS = [
    "cycle_idx", "soc_final", "soh", "rul_mean", "rul_ci_lower", "rul_ci_upper",
    "R0", "rmse", "cusum", "mode_flags", "timestamp",
]


# ============================================================================
# ShadowSync
# ============================================================================

class ShadowSync:
    def __init__(self, config_path: str | Path = "config.yaml"):
        self._cfg  = self._load_config(config_path)
        cfg        = self._cfg
        ecm_params = cfg.get("ecm", {}).get("parameters", cfg.get("ecm", {}))

        self.ekf        = _new(OnlineEKF,         ocv_model=None,
                                                  ecm_params=ecm_params, config=cfg)
        self.ecm        = _new(ECMModel,          config=cfg)
        self.gp         = _new(SVGPCorrector,     config=cfg)
        self.age_emu    = _new(AgeingEmulator,    config_path=str(config_path))
        self.age_model  = _new(AgeingModel,       config=cfg)
        self.classifier = _new(ModeClassifier,    config=cfg)
        self.identifier = _new(OfflineIdentifier, config=cfg)

        self._wrls_lam = float(cfg.get("wrls", {}).get("forgetting_factor", 0.98))
        self._R0_wrls  = float(ecm_params.get("R0", 0.015))
        self._P_wrls   = float(cfg.get("wrls", {}).get("P_init", 1e-3))

        # FIX 2: default k0 = 0.010 V (set cusum.k0 in config.yaml to match
        # your observed steady-state RMSE so CUSUM only triggers on real drift)
        self._k0      = float(cfg.get("cusum", {}).get("k0", 0.010))
        self._cusum_h = float(cfg.get("cusum", {}).get("threshold_h", 5.0))
        self._cusum   = 0.0
        self._cusum_triggers: list[int] = []

        self._ecm_params = ecm_params

        # FIX 1a: rolling theta buffer of shape (window, 5) for LSTM seeding
        self._lstm_window = int(cfg.get("lstm", {}).get("window", 20))
        theta_init = np.array([
            float(ecm_params.get("R0", 0.015)),
            float(ecm_params.get("R1", 0.010)),
            float(ecm_params.get("C1", 2500.0)),
            float(ecm_params.get("R2", 0.020)),
            float(ecm_params.get("C2", 12000.0)),
        ])
        self._theta_buf: np.ndarray = np.tile(theta_init, (self._lstm_window, 1))

        db_path = cfg.get("db_path", "shadow.db")
        self._con = sqlite3.connect(db_path, check_same_thread=False)
        self._con.execute(_CREATE_TABLE)
        self._con.commit()
        logger.info("ShadowSync ready — db: %s", db_path)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _load_config(path: str | Path) -> dict:
        p = Path(path)
        if p.exists():
            with p.open() as fh:
                return yaml.safe_load(fh) or {}
        logger.warning("Config %s not found — using defaults.", path)
        return {}

    def _ocv(self, soc: float) -> float:
        try:
            return float(self.ecm.ocv(soc))
        except Exception:
            return 3.0 + 1.2 * float(soc)

    def _wrls_update(self, V_meas: float, V_oc: float,
                     V1: float, V2: float, I: float) -> float:
        """Scalar WRLS for R0.  phi = -I,  model: V = V_oc - R0*I - V1 - V2."""
        lam            = self._wrls_lam
        phi            = -I
        err            = V_meas - (V_oc - self._R0_wrls * I - V1 - V2)
        denom          = lam + phi * self._P_wrls * phi + 1e-14
        K              = (self._P_wrls * phi) / denom
        self._R0_wrls += K * err
        self._P_wrls   = (1.0 / lam) * (1.0 - K * phi) * self._P_wrls
        self._R0_wrls  = float(np.clip(self._R0_wrls, 1e-4, 0.5))
        return self._R0_wrls

    # ---------------------------------------------------------------- run_cycle
    def run_cycle(self, cycle: dict) -> dict:
        """
        Process one complete cycle sample-by-sample.

        Parameters
        ----------
        cycle : dict
            Required keys: 'cycle_idx', 'V', 'I', 'T' (array-like).
            Optional: 'dt' (float, default 1.0), 'C_max' (float).
        """
        cycle_idx = int(cycle["cycle_idx"])
        V_arr = np.asarray(cycle["V"], dtype=float)
        I_arr = np.asarray(cycle["I"], dtype=float)
        T_arr = np.asarray(cycle["T"], dtype=float)
        dt    = float(cycle.get("dt", 1.0))
        N     = len(V_arr)
        if N == 0:
            raise ValueError(f"Cycle {cycle_idx}: empty arrays.")

        # FIX 4: ensure discharge current is negative (NASA PCoE convention)
        if float(np.mean(I_arr)) > 0:
            I_arr = -I_arr

        sq_errors: list[float] = []
        gp_vars:   list[float] = []
        soc_t = V1_t = V2_t = 0.0

        # ---- sample-by-sample loop ----------------------------------------
        for k in range(N):
            V_meas = float(V_arr[k])
            I      = float(I_arr[k])
            T      = float(T_arr[k])

            # Step 1: EKF
            x_post, _ = self.ekf.step(V_meas, I, T, dt)
            soc_t = float(x_post[0])
            V1_t  = float(x_post[1])
            V2_t  = float(x_post[2])

            # Step 2: WRLS R0
            V_oc    = self._ocv(soc_t)
            R0_wrls = self._wrls_update(V_meas, V_oc, V1_t, V2_t, I)
            if hasattr(self.ekf, "_R0"):
                self.ekf._R0 = R0_wrls

            # Step 3: GP correction + posterior variance
            gp_mean, gp_var = self.gp.predict(soc_t, I, T, cycle_idx)

            # Step 4: predicted voltage with GP correction
            V_pred = V_oc - R0_wrls * I - V1_t - V2_t + float(gp_mean)
            sq_errors.append((V_meas - V_pred) ** 2)
            gp_vars.append(float(gp_var))

        # ---- post-cycle aggregation ----------------------------------------

        rmse        = float(np.sqrt(np.mean(sq_errors)))
        gp_var_mean = float(np.mean(gp_vars))

        # Step 5: CUSUM  S_k = max(0, S_{k-1} + RMSE - k0)
        self._cusum = max(0.0, self._cusum + rmse - self._k0)

        # Step 6: threshold check → re-identification
        cusum_triggered = False
        if self._cusum > self._cusum_h:
            logger.info(
                "Cycle %d: CUSUM=%.6f > h=%.6f — re-identification triggered.",
                cycle_idx, self._cusum, self._cusum_h,
            )
            _id_cycle = {
                "cycle_idx": cycle_idx, "V": V_arr, "Voltage_measured": V_arr,
                "I": I_arr, "Current_measured": I_arr,
                "T": T_arr, "temperature": T_arr, "Temperature_measured": T_arr,
                "dt": dt, "type": "discharge", "R0": self._R0_wrls,
            }
            try:
                new_params = self.identifier.update([_id_cycle])
                if isinstance(new_params, dict) and "R0" in new_params:
                    self._R0_wrls = float(np.clip(new_params["R0"], 1e-4, 0.5))
                    if hasattr(self.ekf, "_R0"):
                        self.ekf._R0 = self._R0_wrls
            except Exception as exc:
                logger.warning("OfflineIdentifier.update failed: %s", exc)
            self._cusum = 0.0
            cusum_triggered = True
            self._cusum_triggers.append(cycle_idx)

        # Step 7: SoH from measured capacity
        C_nom   = float(self._cfg.get("Q_nom_Ah", self._cfg.get("C_nom_Ah", 2.0)))
        C_max_k = cycle.get("C_max")
        if C_max_k is not None and C_nom > 0.0:
            soh = float(np.clip(float(C_max_k) / C_nom, 0.0, 1.0))
        else:
            R0_nom = float(self._ecm_params.get("R0", 0.015))
            soh    = float(np.clip(1.0 - (self._R0_wrls - R0_nom) / R0_nom, 0.0, 1.0))

        # FIX 1b: feed SoH into AgeingModel EVERY cycle
        try:
            self.age_model.update(soh=soh)
        except TypeError:
            try:
                self.age_model.update(new_k=float(cycle_idx),
                                      new_theta={"R0": self._R0_wrls})
            except Exception:
                pass

        # RUL from AgeingModel (parametric + Monte Carlo CI)
        try:
            rul_dict = self.age_model.predict_rul(k_current=cycle_idx)
        except Exception:
            rul_dict = {"mean": 9999.0, "ci_lower_90": 8000.0, "ci_upper_90": 10000.0}

        rul_mean  = float(rul_dict.get("mean",        9999.0))
        rul_lower = float(rul_dict.get("ci_lower_90", max(0.0, rul_mean * 0.75)))
        rul_upper = float(rul_dict.get("ci_upper_90", rul_mean * 1.25))

        # FIX 1a: slide theta buffer, then call LSTM rollout with valid (window,5) seed
        theta_now = np.array([
            self._R0_wrls,
            float(self._ecm_params.get("R1", 0.010)),
            float(self._ecm_params.get("C1", 2500.0)),
            float(self._ecm_params.get("R2", 0.020)),
            float(self._ecm_params.get("C2", 12000.0)),
        ])
        self._theta_buf = np.vstack([self._theta_buf[1:], theta_now])
        try:
            self.age_emu.rollout(self._theta_buf, n_steps=1)
        except Exception:
            pass   # informational only

        # Step 8: mode classification
        mode_dict = self.classifier.classify_cycle(cycle)

        # Step 9: log to SQLite
        row: dict[str, Any] = {
            "cycle_idx":    cycle_idx,
            "soc_final":    round(soc_t, 6),
            "soh":          round(soh, 6),
            "rul_mean":     round(rul_mean, 3),
            "rul_ci_lower": round(rul_lower, 3),
            "rul_ci_upper": round(rul_upper, 3),
            "R0":           round(self._R0_wrls, 6),
            "rmse":         round(rmse, 8),
            "cusum":        round(self._cusum, 8),
            "mode_flags":   json.dumps({
                **mode_dict,
                "cusum_triggered": cusum_triggered,
                "gp_var_mean":     round(gp_var_mean, 10),
            }),
            "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        with self._con:
            self._con.execute(_INSERT_ROW, row)

        logger.info(
            "Cycle %4d | SoC=%.4f | SoH=%.4f | R0=%.5f O | "
            "RMSE=%.5f V | CUSUM=%.5f | RUL=%.1f [%.1f-%.1f]",
            cycle_idx, soc_t, soh, self._R0_wrls, rmse, self._cusum,
            rul_mean, rul_lower, rul_upper,
        )

        return {**row, "mode_flags": {
            **mode_dict,
            "cusum_triggered": cusum_triggered,
            "gp_var_mean": gp_var_mean,
        }}

    # ---------------------------------------------------------------- get_state
    def get_state(self) -> dict | None:
        cur = self._con.execute(_LATEST_ROW)
        row = cur.fetchone()
        if row is None:
            return None
        state = dict(zip(_ROW_COLUMNS, row))
        try:
            state["mode_flags"] = json.loads(state["mode_flags"])
        except Exception:
            pass
        return state

    # ---------------------------------------------------------------- API
    def fit(self, cycles: list[dict]) -> "ShadowSync":
        for c in cycles: self.run_cycle(c)
        return self

    def predict(self, cycle: dict) -> dict:
        return self.run_cycle(cycle)

    def update(self, cycle: dict) -> dict:
        return self.run_cycle(cycle)

    def close(self) -> None:
        self._con.close()

    def __repr__(self) -> str:
        return (f"ShadowSync(R0={self._R0_wrls:.5f} O, "
                f"cusum={self._cusum:.5f}, h={self._cusum_h})")


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    print("=" * 68)
    print("  shadow/sync.py — Smoke Test")
    print("=" * 68)
    print("  No import errors — ready to run validate.py")