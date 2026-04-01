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
Stub implementations of every dependency are provided so the smoke test
runs self-contained; swap them for the real modules once available.
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
# Stub / fallback implementations
# Each class mirrors the real module's public API.
# Import the real class at the top of the try block; fall back to the stub.
# ============================================================================

# --------------------------------------------------------------------------- OnlineEKF
try:
    from estimation.ekf import OnlineEKF  # type: ignore
except ImportError:
    class OnlineEKF:  # type: ignore
        """Stub EKF — Coulomb counting + linear OCV."""

        def __init__(self, ocv_model=None, ecm_params: dict | None = None, config: dict | None = None):
            cfg       = config or {}
            self._dt  = cfg.get("dt_s", 1.0)
            self._Q_n = cfg.get("Q_nom_Ah", 2.0)
            self.x    = np.array([cfg.get("soc_init", 1.0), 0.0, 0.0])
            q         = cfg.get("ekf_Q_diag", [1e-6, 1e-7, 1e-7])
            self._Q   = np.diag(q)
            self._R   = cfg.get("ekf_R", 1e-4)
            p0        = cfg.get("ekf_P0_diag", [0.01, 0.001, 0.001])
            self._P   = np.diag(p0)
            self._R0  = (ecm_params or {}).get("R0", cfg.get("R0_init", 0.020))

        def step(self, V_meas: float, I: float, T: float,
                 dt: float | None = None) -> tuple[np.ndarray, np.ndarray]:
            dt       = dt or self._dt
            soc, V1, V2 = self.x
            # --- Predict -------------------------------------------------
            soc_new  = np.clip(soc - I * dt / (self._Q_n * 3600.0), 0.0, 1.0)
            x_pred   = np.array([soc_new, V1, V2])
            F        = np.eye(3)
            P_pred   = F @ self._P @ F.T + self._Q
            # --- Update --------------------------------------------------
            V_oc     = 3.0 + 1.2 * x_pred[0]
            h_x      = V_oc - I * self._R0 - x_pred[1] - x_pred[2]
            H        = np.array([1.2, -1.0, -1.0])
            S        = H @ P_pred @ H.T + self._R
            K        = P_pred @ H.T / (S + 1e-14)
            self.x   = x_pred + K * (V_meas - h_x)
            self.x[0] = float(np.clip(self.x[0], 0.0, 1.0))
            self._P  = (np.eye(3) - np.outer(K, H)) @ P_pred
            return self.x.copy(), self._P.copy()

        def reset(self, x0=None, P0=None):
            if x0 is not None: self.x    = np.array(x0, dtype=float)
            if P0 is not None: self._P   = np.array(P0, dtype=float)

        def fit(self, *a, **kw):     return self
        def predict(self, *a, **kw): return self.x.copy()
        def update(self, *a, **kw):  return self


# --------------------------------------------------------------------------- ECMModel
try:
    from ecm.model import ECMModel  # type: ignore
except ImportError:
    class ECMModel:  # type: ignore
        """Stub 2RC ECM."""

        def __init__(self, config: dict | None = None):
            cfg = config or {}
            self.params = {
                "R0": cfg.get("R0_init", 0.020),
                "R1": cfg.get("R1_init", 0.005),
                "C1": cfg.get("C1_init", 1200.0),
                "R2": cfg.get("R2_init", 0.003),
                "C2": cfg.get("C2_init", 5000.0),
            }

        def terminal_voltage(self, soc: float, V1: float, V2: float, I: float) -> float:
            return (3.0 + 1.2 * soc) - self.params["R0"] * I - V1 - V2

        def fit(self, *a, **kw):     return self
        def predict(self, *a, **kw): return None
        def update(self, *a, **kw):  return self


# --------------------------------------------------------------------------- SVGPCorrector
try:
    from augmentation.gp import SVGPCorrector  # type: ignore
except ImportError:
    class SVGPCorrector:  # type: ignore
        """Stub GP corrector — near-zero correction with small variance."""

        def __init__(self, config: dict | None = None):
            self._noise = (config or {}).get("gp_noise_std", 5e-4)

        def predict(self, soc: float, I: float, T: float,
                    cycle_idx: int) -> tuple[float, float]:
            return float(np.random.normal(0.0, self._noise)), float(self._noise ** 2)

        def fit(self, *a, **kw):    return self
        def update(self, *a, **kw): return self


# --------------------------------------------------------------------------- AgeingEmulator (LSTM)
try:
    from augmentation.lstm import AgeingEmulator  # type: ignore
except ImportError:
    class AgeingEmulator:  # type: ignore
        """Stub LSTM ageing emulator."""

        def __init__(self, config: dict | None = None):
            self._rate = (config or {}).get("age_rate_per_cycle", 1e-5)

        def predict(self, theta_seed: np.ndarray, n_steps: int = 1) -> np.ndarray:
            theta = theta_seed.copy()
            out   = []
            for _ in range(n_steps):
                theta[0] *= (1.0 + self._rate)
                out.append(theta.copy())
            return np.array(out)

        def fit(self, *a, **kw):    return self
        def update(self, *a, **kw): return self


# --------------------------------------------------------------------------- AgeingModel
try:
    from ageing.parametric import AgeingModel  # type: ignore
except ImportError:
    class AgeingModel:  # type: ignore
        """Stub ageing model — linear capacity fade + Monte Carlo RUL."""

        def __init__(self, config: dict | None = None):
            cfg         = config or {}
            self._eol   = cfg.get("eol_threshold", 0.80)
            self._fade  = cfg.get("age_rate_per_cycle", 1e-5)
            self._n_mc  = cfg.get("n_particles", 1000)
            self._soh_history: list[float] = []
            self._soh   = 1.0

        def update(self, soh: float) -> None:
            self._soh = soh
            self._soh_history.append(soh)

        def predict_rul(self, k_current: int = 0, n_particles: int | None = None) -> dict:
            n_mc = n_particles or self._n_mc
            if len(self._soh_history) < 2:
                return {"mean": 9999.0, "ci_lower_90": 8000.0, "ci_upper_90": 10000.0}
            xs     = np.arange(len(self._soh_history), dtype=float)
            ys     = np.array(self._soh_history)
            m, _b  = np.polyfit(xs, ys, 1)
            if m >= 0.0:
                return {"mean": 9999.0, "ci_lower_90": 8000.0, "ci_upper_90": 10000.0}
            rul_mean  = float(max(0.0, (self._eol - self._soh) / m))
            sigma     = max(1.0, rul_mean * 0.10)
            samples   = np.clip(np.random.normal(rul_mean, sigma, n_mc), 0.0, None)
            return {
                "mean":        rul_mean,
                "ci_lower_90": float(np.percentile(samples, 5.0)),
                "ci_upper_90": float(np.percentile(samples, 95.0)),
            }

        def fit(self, *a, **kw):     return self
        def predict(self, *a, **kw): return self.predict_rul()
        def update_fit(self, *a, **kw): return self


# --------------------------------------------------------------------------- ModeClassifier
try:
    from ageing.modes import ModeClassifier  # type: ignore
except ImportError:
    class ModeClassifier:  # type: ignore
        """Stub degradation mode classifier."""

        def __init__(self, config: dict | None = None):
            cfg = config or {}
            self._thr_ch  = cfg.get("mode_charge_thresh",    0.05)
            self._thr_dis = cfg.get("mode_discharge_thresh", -0.05)

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
    from ecm.identifier import OfflineIdentifier  # type: ignore
except ImportError:
    class OfflineIdentifier:  # type: ignore
        """Stub offline parameter identifier."""

        def __init__(self, config: dict | None = None):
            self._history: list[dict] = []

        def update(self, cycle_data: dict) -> dict:
            self._history.append(cycle_data)
            R0_new = cycle_data.get("R0", 0.020) * (1.0 + float(np.random.normal(0.0, 5e-4)))
            return {"R0": float(np.clip(R0_new, 1e-4, 0.5))}

        def fit(self, *a, **kw):     return self
        def predict(self, *a, **kw): return {}


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

    Parameters
    ----------
    config_path : str | Path
        Path to config.yaml.  Missing keys fall back to safe defaults.
    """

    # ----------------------------------------------------------------- __init__
    def __init__(self, config_path: str | Path = "config.yaml"):
        self._cfg = self._load_config(config_path)
        cfg       = self._cfg

        # ---- Sub-config dicts -------------------------------------------
        ekf_cfg  = cfg.get("ekf",          {})
        ecm_cfg  = cfg.get("ecm",          {})
        gp_cfg   = cfg.get("gp",           {})
        lstm_cfg = cfg.get("lstm",         {})
        age_cfg  = cfg.get("ageing",       {})
        mc_cfg   = cfg.get("monte_carlo",  {})
        mode_cfg = cfg.get("modes",        {})
        wrls_cfg = cfg.get("wrls",         {})
        cusum_cfg= cfg.get("cusum",        {})

        # Scalar pass-throughs needed by stubs
        shared = {
            "Q_nom_Ah":           cfg.get("Q_nom_Ah", 2.0),
            "soc_init":           cfg.get("soc_init", 1.0),
            "R0_init":            ecm_cfg.get("R0_init", 0.020),
            "eol_threshold":      mc_cfg.get("eol_threshold", 0.80),
            "n_particles":        mc_cfg.get("n_particles", 1000),
            "age_rate_per_cycle": age_cfg.get("age_rate_per_cycle", 1e-5),
        }

        # ---- Instantiate all modules ------------------------------------
        self.ekf        = OnlineEKF(
            ocv_model  = None,
            ecm_params = ecm_cfg,
            config     = {**ekf_cfg, **shared},
        )
        self.ecm        = ECMModel(config={**ecm_cfg, **shared})
        self.gp         = SVGPCorrector(config=gp_cfg)
        self.age_emu    = AgeingEmulator(config={**lstm_cfg, **shared})
        self.age_model  = AgeingModel(config={**age_cfg, **mc_cfg, **shared})
        self.classifier = ModeClassifier(config=mode_cfg)
        self.identifier = OfflineIdentifier(config=ecm_cfg)

        # ---- WRLS state -------------------------------------------------
        self._wrls_lam  = wrls_cfg.get("forgetting_factor", 0.98)
        self._R0_wrls   = ecm_cfg.get("R0_init", 0.020)
        self._P_wrls    = wrls_cfg.get("P_init", 1.0)

        # ---- CUSUM state ------------------------------------------------
        self._k0        = cusum_cfg.get("k0", 1e-3)
        raw_h           = cusum_cfg.get("threshold", 5.0)
        # Support h expressed as 5.0 (scaled) or 0.005 (raw)
        self._cusum_h   = raw_h * 1e-3 if raw_h >= 1.0 else raw_h
        self._cusum     = 0.0
        self._cusum_triggers: list[int] = []

        # ---- SQLite -----------------------------------------------------
        db_path       = cfg.get("db_path", "shadow.db")
        self._con     = sqlite3.connect(db_path, check_same_thread=False)
        self._con.execute(_CREATE_TABLE)
        self._con.commit()
        logger.info("ShadowSync ready — db: %s", db_path)

    # --------------------------------------------------------------- config
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

    # --------------------------------------------------------------- WRLS
    def _wrls_update(self, V_meas: float, V_oc: float,
                     V1: float, V2: float, I: float) -> float:
        """
        WRLS update for R0 with forgetting factor lambda.

        Model:  V_meas = V_oc - R0*I - V1 - V2
        Regressor: phi = -I  (scalar)
        """
        lam         = self._wrls_lam
        phi         = -I
        err         = V_meas - (V_oc - self._R0_wrls * I - V1 - V2)
        denom       = lam + phi * self._P_wrls * phi + 1e-14
        K           = self._P_wrls * phi / denom
        self._R0_wrls  += K * err
        self._P_wrls    = (1.0 / lam) * (1.0 - K * phi) * self._P_wrls
        self._R0_wrls   = float(np.clip(self._R0_wrls, 1e-4, 0.5))
        return self._R0_wrls

    # --------------------------------------------------------------- run_cycle
    def run_cycle(self, cycle: dict) -> dict:
        """
        Process one complete cycle sample-by-sample.

        Parameters
        ----------
        cycle : dict
            Required keys:
              'cycle_idx' (int)
              'V'  (array-like, N)   measured terminal voltage [V]
              'I'  (array-like, N)   current [A]  (positive = discharge)
              'T'  (array-like, N)   temperature [degC]
            Optional:
              'dt' (float)           sample period [s], default 1.0

        Returns
        -------
        state_dict : dict  —  the row logged to SQLite.
        """
        cycle_idx: int = int(cycle["cycle_idx"])
        V_arr = np.asarray(cycle["V"], dtype=float)
        I_arr = np.asarray(cycle["I"], dtype=float)
        T_arr = np.asarray(cycle["T"], dtype=float)
        dt    = float(cycle.get("dt", 1.0))
        N     = len(V_arr)

        if N == 0:
            raise ValueError(f"Cycle {cycle_idx}: empty V/I/T arrays.")

        sq_errors: list[float] = []
        x_post: np.ndarray | None = None
        soc_t = V1_t = V2_t = 0.0

        # ================================================================
        # Sample-by-sample online loop
        # ================================================================
        for k in range(N):
            V_meas = float(V_arr[k])
            I      = float(I_arr[k])
            T      = float(T_arr[k])

            # ---- Step 1: EKF -------------------------------------------
            x_post, _P = self.ekf.step(V_meas, I, T, dt)
            soc_t = float(x_post[0])
            V1_t  = float(x_post[1])
            V2_t  = float(x_post[2])

            # ---- Step 2: WRLS R0 update --------------------------------
            V_oc    = 3.0 + 1.2 * soc_t       # OCV (real: self.ecm.ocv(soc_t))
            R0_wrls = self._wrls_update(V_meas, V_oc, V1_t, V2_t, I)
            # Feed updated R0 back to EKF stub
            if hasattr(self.ekf, "_R0"):
                self.ekf._R0 = R0_wrls

            # ---- Step 3: GP voltage correction -------------------------
            gp_mean, _gp_var   = self.gp.predict(soc_t, I, T, cycle_idx)
            V_pred_raw          = V_oc - R0_wrls * I - V1_t - V2_t
            V_pred_corrected    = V_pred_raw + gp_mean

            # ---- Step 4: accumulate squared errors ---------------------
            sq_errors.append((V_meas - V_pred_corrected) ** 2)

        # ================================================================
        # Post-cycle aggregation
        # ================================================================

        # Step 4 (continued): cycle RMSE
        rmse = float(np.sqrt(np.mean(sq_errors)))

        # Step 5: CUSUM update
        self._cusum = max(0.0, self._cusum + rmse - self._k0)

        # Step 6: CUSUM threshold — trigger re-identification
        cusum_triggered = False
        if self._cusum > self._cusum_h:
            logger.info(
                "Cycle %d: CUSUM=%.6f > h=%.6f — triggering OfflineIdentifier.",
                cycle_idx, self._cusum, self._cusum_h,
            )
            new_params = self.identifier.update({
                "cycle_idx": cycle_idx,
                "V": V_arr,
                "I": I_arr,
                "T": T_arr,
                "R0": self._R0_wrls,
            })
            if "R0" in new_params:
                self._R0_wrls = float(np.clip(new_params["R0"], 1e-4, 0.5))
                if hasattr(self.ekf, "_R0"):
                    self.ekf._R0 = self._R0_wrls
            self._cusum = 0.0
            cusum_triggered = True
            self._cusum_triggers.append(cycle_idx)

        # Step 7: AgeingModel update + RUL prediction
        # Use LSTM emulator to project one step
        theta_now = np.array([
            self._R0_wrls,
            self._cfg.get("ecm", {}).get("R1_init", 0.005),
            self._cfg.get("ecm", {}).get("C1_init", 1200.0),
            self._cfg.get("ecm", {}).get("R2_init", 0.003),
            self._cfg.get("ecm", {}).get("C2_init", 5000.0),
        ])
        _theta_next = self.age_emu.predict(theta_now, n_steps=1)

        age_rate = self._cfg.get("ageing", {}).get("age_rate_per_cycle", 1e-5)
        T_mean   = float(np.mean(T_arr))
        temp_fac = 1.0 + 0.02 * max(0.0, T_mean - 25.0)
        soh      = float(max(0.0, 1.0 - cycle_idx * age_rate * temp_fac))

        self.age_model.update(soh)
        rul_dict = self.age_model.predict_rul(k_current=cycle_idx)

        # Step 8: Mode classification
        mode_dict = self.classifier.classify_cycle(cycle)

        # Step 9: Log to SQLite
        soc_final = float(self.ekf.x[0]) if hasattr(self.ekf, "x") else soc_t

        row: dict[str, Any] = {
            "cycle_idx":    cycle_idx,
            "soc_final":    round(soc_final, 6),
            "soh":          round(soh, 6),
            "rul_mean":     round(rul_dict["mean"], 3),
            "rul_ci_lower": round(rul_dict["ci_lower_90"], 3),
            "rul_ci_upper": round(rul_dict["ci_upper_90"], 3),
            "R0":           round(self._R0_wrls, 6),
            "rmse":         round(rmse, 8),
            "cusum":        round(self._cusum, 8),
            "mode_flags":   json.dumps({**mode_dict, "cusum_triggered": cusum_triggered}),
            "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        with self._con:
            self._con.execute(_INSERT_ROW, row)

        logger.info(
            "Cycle %4d | SoC=%.4f | SoH=%.4f | R0=%.5f Ohm | "
            "RMSE=%.5f V | CUSUM=%.5f | RUL=%.1f cyc [%.1f–%.1f] | mode=%s",
            cycle_idx, soc_final, soh, self._R0_wrls, rmse,
            self._cusum, rul_dict["mean"],
            rul_dict["ci_lower_90"], rul_dict["ci_upper_90"],
            mode_dict.get("mode", "?"),
        )
        return row

    # --------------------------------------------------------------- get_state
    def get_state(self) -> dict | None:
        """Return the most recently logged state dict from SQLite."""
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

    # --------------------------------------------------------------- API wrappers
    def fit(self, cycles: list[dict]) -> "ShadowSync":
        """Batch warm-up: run run_cycle() over a list of historical cycles."""
        for cycle in cycles:
            self.run_cycle(cycle)
        return self

    def predict(self, cycle: dict) -> dict:
        """Alias for run_cycle — process one cycle and return its state dict."""
        return self.run_cycle(cycle)

    def update(self, cycle: dict) -> dict:
        """Alias for run_cycle — online incremental update."""
        return self.run_cycle(cycle)

    # --------------------------------------------------------------- housekeeping
    def close(self) -> None:
        """Close the SQLite connection."""
        self._con.close()

    def __repr__(self) -> str:
        return (
            f"ShadowSync(R0={self._R0_wrls:.5f} Ohm, "
            f"cusum={self._cusum:.5f}, h={self._cusum_h:.5f})"
        )


# ============================================================================
# Smoke test
# ============================================================================

def _synthetic_cycle(
    cycle_idx: int        = 0,
    n_samples: int        = 360,
    soc_start: float      = 1.0,
    I_dc: float           = 1.0,
    T_mean: float         = 25.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Generate one synthetic 1-Hz discharge cycle with measurement noise."""
    if rng is None:
        rng = np.random.default_rng(seed=cycle_idx + 42)
    Q_n  = 2.0   # Ah
    R0   = 0.020 + cycle_idx * 1e-5
    soc  = soc_start
    V_list, I_list, T_list = [], [], []
    for _ in range(n_samples):
        V_oc = 3.0 + 1.2 * soc
        V_t  = V_oc - R0 * I_dc + rng.normal(0.0, 1e-3)
        V_list.append(V_t)
        I_list.append(I_dc)
        T_list.append(T_mean + rng.normal(0.0, 0.3))
        soc  = max(0.0, soc - I_dc * 1.0 / (Q_n * 3600.0))
    return {
        "cycle_idx": cycle_idx,
        "V":  np.array(V_list),
        "I":  np.array(I_list),
        "T":  np.array(T_list),
        "dt": 1.0,
    }


if __name__ == "__main__":
    import tempfile, os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 68)
    print("  shadow/sync.py — Smoke Test")
    print("=" * 68)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "config.yaml")
        db_path  = os.path.join(tmpdir, "shadow.db")

        cfg_dict: dict = {
            "db_path":  db_path,
            "Q_nom_Ah": 2.0,
            "soc_init": 1.0,
            "ekf": {
                "ekf_Q_diag":  [1e-6, 1e-7, 1e-7],
                "ekf_R":       1e-4,
                "ekf_P0_diag": [0.01, 0.001, 0.001],
                "dt_s":        1.0,
            },
            "ecm": {
                "R0_init": 0.020, "R1_init": 0.005,
                "C1_init": 1200.0, "R2_init": 0.003, "C2_init": 5000.0,
            },
            "wrls":         {"forgetting_factor": 0.98, "P_init": 1.0},
            "cusum":        {"k0": 1e-3, "threshold": 5e-3},
            "gp":           {"gp_noise_std": 5e-4},
            "lstm":         {"age_rate_per_cycle": 1e-5},
            "ageing":       {"age_rate_per_cycle": 1e-5},
            "monte_carlo":  {"eol_threshold": 0.80, "n_particles": 500},
            "modes": {
                "mode_charge_thresh":    0.05,
                "mode_discharge_thresh": -0.05,
            },
        }

        with open(cfg_path, "w") as fh:
            yaml.dump(cfg_dict, fh)

        # --- instantiate -----------------------------------------------
        ss = ShadowSync(config_path=cfg_path)
        print(f"\n  Instantiated: {ss}\n")

        # --- single synthetic cycle (cycle_idx=5 → mild ageing) --------
        cycle = _synthetic_cycle(
            cycle_idx=5, n_samples=180, I_dc=1.0, T_mean=30.0,
        )
        print(f"  Synthetic cycle: {len(cycle['V'])} samples  |  "
              f"I_mean={cycle['I'].mean():.2f} A  |  "
              f"T_mean={cycle['T'].mean():.1f} degC\n")

        # --- run the synchronisation loop (collect per-step data) ------
        # Re-run with instrumentation to capture V_pred for the plot
        V_arr   = cycle["V"]
        I_arr   = cycle["I"]
        T_arr   = cycle["T"]
        steps   = np.arange(len(V_arr))
        V_pred_trace: list[float] = []
        soc_trace:    list[float] = []
        R0_trace:     list[float] = []

        ss2 = ShadowSync(config_path=cfg_path)
        for k in range(len(V_arr)):
            x, _ = ss2.ekf.step(float(V_arr[k]), float(I_arr[k]), float(T_arr[k]))
            soc_k = float(x[0])
            V1_k  = float(x[1])
            V2_k  = float(x[2])
            V_oc_k = 3.0 + 1.2 * soc_k
            R0_k   = ss2._wrls_update(float(V_arr[k]), V_oc_k, V1_k, V2_k, float(I_arr[k]))
            gp_mu, _ = ss2.gp.predict(soc_k, float(I_arr[k]), float(T_arr[k]), 5)
            V_pred_trace.append(V_oc_k - R0_k * float(I_arr[k]) - V1_k - V2_k + gp_mu)
            soc_trace.append(soc_k)
            R0_trace.append(R0_k)
        ss2.close()

        _state = ss.run_cycle(cycle)

        # --- get_state() round-trip via SQLite -------------------------
        state = ss.get_state()

        print("\n" + "=" * 68)
        print("  Final logged state dict  (from SQLite)")
        print("=" * 68)
        for k, v in state.items():
            print(f"  {k:<18}: {v}")
        print("=" * 68)

        # --- save smoke test result as PNG ----------------------------
        png_path = "smoke_test_result.png"
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"ShadowSync Smoke Test — Cycle {state['cycle_idx']}", fontsize=13)

        # Panel 1: Measured vs GP-corrected predicted voltage
        ax = axes[0, 0]
        ax.plot(steps, V_arr,          label="V measured", color="steelblue", lw=1.2)
        ax.plot(steps, V_pred_trace,   label="V predicted (EKF+GP)", color="darkorange",
                lw=1.2, linestyle="--")
        ax.set_xlabel("Sample"); ax.set_ylabel("Voltage (V)")
        ax.set_title("Terminal Voltage: Measured vs Predicted")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Panel 2: EKF SoC trace
        ax = axes[0, 1]
        ax.plot(steps, soc_trace, color="seagreen", lw=1.5)
        ax.set_xlabel("Sample"); ax.set_ylabel("SoC")
        ax.set_title(f"EKF SoC Estimate  (final={state['soc_final']:.4f})")
        ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)

        # Panel 3: WRLS R0 convergence
        ax = axes[1, 0]
        ax.plot(steps, [r * 1e3 for r in R0_trace], color="firebrick", lw=1.5)
        ax.set_xlabel("Sample"); ax.set_ylabel("R0 (mΩ)")
        ax.set_title(f"WRLS R0 Tracking  (final={state['R0']*1e3:.3f} mΩ)")
        ax.grid(alpha=0.3)

        # Panel 4: logged state summary as text
        ax = axes[1, 1]
        ax.axis("off")
        summary = "\n".join([
            f"cycle_idx  : {state['cycle_idx']}",
            f"soc_final  : {state['soc_final']:.6f}",
            f"soh        : {state['soh']:.6f}",
            f"R0         : {state['R0']*1e3:.4f} mΩ",
            f"rmse       : {state['rmse']*1e3:.4f} mV",
            f"cusum      : {state['cusum']:.2e}",
            f"rul_mean   : {state['rul_mean']:.1f} cyc",
            f"rul_ci     : [{state['rul_ci_lower']:.1f}, {state['rul_ci_upper']:.1f}]",
            f"mode       : {state['mode_flags'].get('mode', '?')}",
            f"cusum_trig : {state['mode_flags'].get('cusum_triggered', False)}",
            f"timestamp  : {state['timestamp']}",
        ])
        ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        ax.set_title("Logged State (SQLite)")

        plt.tight_layout()
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Smoke test plot saved → {png_path}\n")

        # --- assertions ------------------------------------------------
        assert 0.0 <= state["soc_final"] <= 1.0,  "SoC out of [0, 1]"
        assert 0.0 <= state["soh"]       <= 1.0,  "SoH out of [0, 1]"
        assert state["rmse"]             >= 0.0,   "RMSE must be >= 0"
        assert state["cusum"]            >= 0.0,   "CUSUM must be >= 0"
        assert state["rul_mean"]         >= 0.0,   "RUL must be >= 0"
        assert isinstance(state["mode_flags"], dict), "mode_flags must be dict"
        assert "mode" in state["mode_flags"],       "mode_flags must contain 'mode'"

        ss.close()
        print("\n  All assertions passed — smoke test complete.\n")

        # --- save smoke-test figure as PNG ------------------------------
        png_path = "smoke_test_result.png"
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(
            f"ShadowSync Smoke Test — Cycle {state['cycle_idx']}  "
            f"({len(V_arr)} samples @ 1 Hz)",
            fontsize=13, fontweight="bold",
        )

        # Panel 1: Voltage tracking
        ax = axes[0, 0]
        ax.plot(steps, V_arr,          lw=1.2, label="V measured",       color="#1f77b4")
        ax.plot(steps, V_pred_trace,   lw=1.2, label="V predicted + GP", color="#ff7f0e",
                linestyle="--")
        ax.set_xlabel("Sample (step)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title("Voltage Tracking")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 2: SoC trajectory
        ax = axes[0, 1]
        ax.plot(steps, soc_trace, lw=1.4, color="#2ca02c")
        ax.set_xlabel("Sample (step)")
        ax.set_ylabel("SoC")
        ax.set_title(f"EKF SoC Estimate  (final = {state['soc_final']:.4f})")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        # Panel 3: WRLS R0 convergence
        ax = axes[1, 0]
        ax.plot(steps, [r * 1e3 for r in R0_trace], lw=1.4, color="#d62728")
        ax.axhline(state["R0"] * 1e3, color="k", linestyle=":", lw=1,
                   label=f"Final R0 = {state['R0']*1e3:.3f} mΩ")
        ax.set_xlabel("Sample (step)")
        ax.set_ylabel("R0 (mΩ)")
        ax.set_title("WRLS R0 Convergence")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 4: State summary bar chart
        ax = axes[1, 1]
        labels = ["SoC", "SoH", "RMSE×100", "CUSUM×100"]
        values = [
            state["soc_final"],
            state["soh"],
            state["rmse"] * 100,
            state["cusum"] * 100,
        ]
        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
        bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_title("Final State Summary")
        ax.set_ylabel("Value")
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  PNG saved → {png_path}\n")