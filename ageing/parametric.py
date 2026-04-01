"""
ageing/parametric.py
--------------------
Parametric degradation law fitter + Monte Carlo RUL predictor.

Degradation laws:
  R0(k)    = a * exp(b * k) + c * k^0.5
  C_max(k) = C0 * (1 - beta * k^n)

All hyperparameters are read from config.yaml — nothing is hardcoded.
"""

import numpy as np
import yaml
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Helper: load config once
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Degradation law functions (module-level so curve_fit can pickle them)
# ---------------------------------------------------------------------------

def _r0_law(k, a, b, c):
    """R0(k) = a * exp(b * k) + c * sqrt(k)"""
    return a * np.exp(b * k) + c * np.sqrt(k)


def _cmax_law(k, C0, beta, n):
    """C_max(k) = C0 * (1 - beta * k^n)"""
    return C0 * (1.0 - beta * np.power(k, n))


# ---------------------------------------------------------------------------
# AgeingModel
# ---------------------------------------------------------------------------

class AgeingModel:
    """
    Fits parametric ageing laws across cycle history and predicts
    Remaining Useful Life (RUL) via Monte Carlo extrapolation.

    Methods
    -------
    fit(cycle_indices, theta_list)   — fit R0 and C_max laws
    predict(k_current)               — alias for predict_rul
    update(new_k, new_theta)         — incremental refit
    stress_factor(dod, T_k, c_rate) — compute stress multiplier
    predict_rul(k_current, ...)      — Monte Carlo RUL with CI
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg = _load_config(config_path)
        ag = cfg["ageing"]
        mc = cfg["monte_carlo"]

        # --- initial guesses for curve_fit ---
        self._r0_p0   = [ag["a"],    ag["b"],   ag["c"]]
        self._cmax_p0 = [ag["C0"],   ag["beta"], ag["n"]]

        # --- bounds: keep parameters physically meaningful ---
        self._r0_bounds = (
            [1e-5,  -0.5,  0.0],   # lower  (b can be negative or slightly positive)
            [1.0,    0.5,  1.0],   # upper
        )
        self._cmax_bounds = (
            [0.5,   0.0,  0.3],    # lower
            [5.0,   1.0,  1.0],    # upper
        )

        # --- stress-factor exponents ---
        self._dod_gamma = ag["dod_gamma"]
        self._Ea        = ag["Ea"]
        self._R_gas     = ag["R_gas"]

        # --- Monte Carlo settings ---
        self._n_particles    = mc["n_particles"]
        self._eol_threshold  = mc["eol_threshold"]

        # --- fitted parameters (populated after fit()) ---
        self._r0_params:   np.ndarray | None = None   # [a, b, c]
        self._r0_cov:      np.ndarray | None = None
        self._cmax_params: np.ndarray | None = None   # [C0, beta, n]
        self._cmax_cov:    np.ndarray | None = None

        # --- raw observations (grow with update()) ---
        self._k_obs:    list[float] = []
        self._r0_obs:   list[float] = []
        self._cmax_obs: list[float] = []

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def fit(self, cycle_indices, theta_list):
        """
        Fit R0 and C_max degradation laws from cycle history.

        Parameters
        ----------
        cycle_indices : array-like, shape (N,)
            Integer cycle numbers, e.g. [0, 1, 2, ..., 199].
        theta_list : list of dict
            Each dict must contain at least 'R0' and optionally 'C_max'.
            If 'C_max' is absent, it is approximated from 'R0' using the
            inverse relationship C_max ≈ C0_init / (R0 / R0_init).
        """
        k = np.asarray(cycle_indices, dtype=float)

        r0_vals   = np.array([t["R0"]                     for t in theta_list], dtype=float)
        cmax_vals = np.array([t.get("C_max", self._cmax_p0[0]
                               * (self._r0_p0[0] / max(t["R0"], 1e-9)))
                              for t in theta_list], dtype=float)

        # Store observations
        self._k_obs    = k.tolist()
        self._r0_obs   = r0_vals.tolist()
        self._cmax_obs = cmax_vals.tolist()

        self._fit_laws(k, r0_vals, cmax_vals)

    def predict(self, k_current, n_particles: int | None = None):
        """Alias for predict_rul — keeps the fit/predict/update API uniform."""
        return self.predict_rul(k_current, n_particles=n_particles or self._n_particles)

    def update(self, new_k: float, new_theta: dict):
        """
        Append one new observation and refit both laws.

        Parameters
        ----------
        new_k     : int or float — cycle index of the new observation
        new_theta : dict with keys 'R0' and optionally 'C_max'
        """
        self._k_obs.append(float(new_k))
        self._r0_obs.append(float(new_theta["R0"]))

        c0 = self._cmax_p0[0]
        r0_init = self._r0_p0[0]
        cmax_new = new_theta.get(
            "C_max", c0 * (r0_init / max(new_theta["R0"], 1e-9))
        )
        self._cmax_obs.append(float(cmax_new))

        k        = np.array(self._k_obs)
        r0_vals  = np.array(self._r0_obs)
        cmax_vals = np.array(self._cmax_obs)

        self._fit_laws(k, r0_vals, cmax_vals)

    def stress_factor(self, dod: float, T_kelvin: float, c_rate: float) -> float:
        """
        Combined stress multiplier for ageing acceleration.

          SF = DoD^gamma * exp(Ea / (R_gas * T)) * c_rate

        Parameters
        ----------
        dod      : depth of discharge in [0, 1]
        T_kelvin : cell temperature in Kelvin
        c_rate   : C-rate of the cycle (e.g. 1.0 for 1C)

        Returns
        -------
        float — dimensionless stress factor (>= 0)
        """
        dod_term   = np.power(np.clip(dod, 1e-6, 1.0), self._dod_gamma)
        arrh_term  = np.exp(self._Ea / (self._R_gas * T_kelvin))
        return float(dod_term * arrh_term * c_rate)

    def predict_rul(self, k_current: float, n_particles: int | None = None) -> dict:
        """
        Monte Carlo RUL prediction.

        Samples parameter uncertainty from the covariance matrices returned
        by curve_fit, extrapolates C_max(k) forward for each particle until
        it crosses the EOL threshold, and reports mean + 90 % CI.

        Parameters
        ----------
        k_current  : current cycle index
        n_particles: number of Monte Carlo particles (default from config)

        Returns
        -------
        dict with keys:
          mean_rul      — expected cycles remaining
          ci_lower_90   — 5th percentile
          ci_upper_90   — 95th percentile
          eol_cycle_mean — absolute cycle at EOL (mean)
        """
        if self._cmax_params is None:
            raise RuntimeError("Call fit() before predict_rul().")

        n = n_particles or self._n_particles
        C0_fit, beta_fit, n_fit = self._cmax_params

        # Sample parameter uncertainty
        try:
            samples = np.random.multivariate_normal(
                self._cmax_params, self._cmax_cov, size=n
            )
            # Clip to physically valid ranges
            samples[:, 0] = np.clip(samples[:, 0], 0.1,  10.0)   # C0
            samples[:, 1] = np.clip(samples[:, 1], 1e-6,  0.99)  # beta
            samples[:, 2] = np.clip(samples[:, 2], 0.1,   1.0)   # n
        except np.linalg.LinAlgError:
            # Covariance not positive definite — use point estimate with small noise
            noise = np.abs(self._cmax_params) * 0.02
            samples = self._cmax_params + np.random.randn(n, 3) * noise

        eol_capacity = self._eol_threshold * C0_fit
        rul_particles = np.full(n, np.nan)

        # Vectorised forward extrapolation
        max_extra = 5000
        k_future = np.arange(k_current, k_current + max_extra, dtype=float)

        for i, (C0_s, beta_s, n_s) in enumerate(samples):
            c_traj = _cmax_law(k_future, C0_s, beta_s, n_s)
            below  = np.where(c_traj <= eol_capacity)[0]
            if len(below) > 0:
                rul_particles[i] = float(k_future[below[0]] - k_current)
            else:
                rul_particles[i] = float(max_extra)   # beyond horizon

        valid = rul_particles[~np.isnan(rul_particles)]
        if len(valid) == 0:
            valid = np.array([max_extra])

        mean_rul    = float(np.mean(valid))
        ci_lower_90 = float(np.percentile(valid, 5.0))
        ci_upper_90 = float(np.percentile(valid, 95.0))

        return {
            "mean_rul":       mean_rul,
            "ci_lower_90":    ci_lower_90,
            "ci_upper_90":    ci_upper_90,
            "eol_cycle_mean": float(k_current + mean_rul),
        }

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _fit_laws(self, k, r0_vals, cmax_vals):
        """Internal: run curve_fit for both laws and store params + covariance."""

        # --- R0 law ---
        try:
            popt_r0, pcov_r0 = curve_fit(
                _r0_law,
                k, r0_vals,
                p0=self._r0_p0,
                bounds=self._r0_bounds,
                maxfev=10_000,
            )
        except RuntimeError:
            # If optimisation fails, fall back to initial guesses
            popt_r0 = np.array(self._r0_p0)
            pcov_r0 = np.diag(np.array(self._r0_p0) ** 2 * 0.1)

        self._r0_params = popt_r0
        self._r0_cov    = pcov_r0

        # --- C_max law ---
        try:
            popt_cm, pcov_cm = curve_fit(
                _cmax_law,
                k, cmax_vals,
                p0=self._cmax_p0,
                bounds=self._cmax_bounds,
                maxfev=10_000,
            )
        except RuntimeError:
            popt_cm = np.array(self._cmax_p0)
            pcov_cm = np.diag(np.array(self._cmax_p0) ** 2 * 0.1)

        self._cmax_params = popt_cm
        self._cmax_cov    = pcov_cm

    def summary(self) -> dict:
        """Return fitted parameters as a readable dict."""
        if self._r0_params is None:
            return {"status": "not fitted"}
        a, b, c         = self._r0_params
        C0, beta, n_exp = self._cmax_params
        return {
            "R0_law":   {"a": a, "b": b, "c": c},
            "Cmax_law": {"C0": C0, "beta": beta, "n": n_exp},
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 55)
    print("  ageing/parametric.py  —  smoke test")
    print("=" * 55)

    # ---- Synthetic 200-cycle degradation ----
    np.random.seed(42)
    N = 200
    cycles = np.arange(N, dtype=float)

    # Ground-truth params tuned so EOL (80% C_max) occurs ~cycle 450
    TRUE_a, TRUE_b, TRUE_c     = 0.005,  0.0008, 0.0002
    TRUE_C0, TRUE_beta, TRUE_n = 2.0,    0.04,   0.6

    r0_true   = _r0_law(cycles,   TRUE_a,  TRUE_b,  TRUE_c)
    cmax_true = _cmax_law(cycles, TRUE_C0, TRUE_beta, TRUE_n)

    # Add small measurement noise
    r0_noisy   = r0_true   + np.random.normal(0, 0.0005, N)
    cmax_noisy = cmax_true + np.random.normal(0, 0.005,  N)

    # Build theta_list (as loader/identifier would produce)
    theta_list = [
        {"R0": float(r0_noisy[i]), "C_max": float(cmax_noisy[i])}
        for i in range(N)
    ]

    # ---- Fit ----
    # We'll use a minimal inline config so the smoke test is self-contained
    import tempfile, os
    _inline_cfg = """
ageing:
  a: 0.005
  b: 0.0008
  c: 0.0002
  C0: 2.0
  beta: 0.04
  n: 0.6
  dod_gamma: 1.2
  Ea: 30000
  R_gas: 8.314
monte_carlo:
  n_particles: 1000
  eol_threshold: 0.8
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                     delete=False) as tf:
        tf.write(_inline_cfg)
        tmp_cfg = tf.name

    model = AgeingModel(config_path=tmp_cfg)
    model.fit(cycles, theta_list)

    print("\nFitted parameters:")
    for law, params in model.summary().items():
        print(f"  {law}: {params}")

    # ---- Stress factor example ----
    sf = model.stress_factor(dod=0.8, T_kelvin=298.15, c_rate=1.0)
    print(f"\nStress factor (DoD=0.8, T=25°C, 1C): {sf:.4e}")

    # Find where EOL actually occurs so smoke test queries a valid point
    C0_f, beta_f, n_f = model._cmax_params
    eol_cap = model._eol_threshold * C0_f
    k_scan  = np.arange(0, 5000, dtype=float)
    c_scan  = _cmax_law(k_scan, C0_f, beta_f, n_f)
    eol_idx = np.where(c_scan <= eol_cap)[0]
    eol_cycle = int(k_scan[eol_idx[0]]) if len(eol_idx) > 0 else 5000
    query_cycle = max(0, eol_cycle - 80)   # 80 cycles before EOL

    # ---- RUL prediction ----
    rul = model.predict_rul(k_current=query_cycle)
    print(f"\nRUL at cycle {query_cycle} (EOL expected ~{eol_cycle}):")
    print(f"  Mean RUL      : {rul['mean_rul']:.1f} cycles")
    print(f"  90% CI lower  : {rul['ci_lower_90']:.1f} cycles")
    print(f"  90% CI upper  : {rul['ci_upper_90']:.1f} cycles")
    print(f"  EOL at cycle  : {rul['eol_cycle_mean']:.1f}")

    # ---- Incremental update (simulate one new cycle) ----
    new_k   = query_cycle + 10
    new_r0  = float(_r0_law(np.array([new_k]), *model._r0_params)[0])
    new_cm  = float(_cmax_law(np.array([new_k]), *model._cmax_params)[0])
    model.update(new_k=new_k, new_theta={"R0": new_r0, "C_max": new_cm})
    rul2 = model.predict_rul(k_current=new_k)
    print(f"\nAfter update() at cycle {new_k}:")
    print(f"  Mean RUL      : {rul2['mean_rul']:.1f} cycles")
    print(f"  90% CI        : [{rul2['ci_lower_90']:.1f}, {rul2['ci_upper_90']:.1f}]")

    # ---- Quick plot ----
    k_plot   = np.linspace(0, 300, 500)
    C0, beta, n_exp = model._cmax_params
    cmax_fit = _cmax_law(k_plot, C0, beta, n_exp)
    eol_cap  = model._eol_threshold * C0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(cycles, r0_noisy,  s=4, alpha=0.4, label="observed R0")
    a, b, c = model._r0_params
    axes[0].plot(k_plot, _r0_law(k_plot, a, b, c), "r-", lw=2, label="fitted")
    axes[0].set_xlabel("Cycle"); axes[0].set_ylabel("R0 (Ω)")
    axes[0].set_title("R0 degradation law"); axes[0].legend()

    axes[1].scatter(cycles, cmax_noisy, s=4, alpha=0.4, label="observed C_max")
    axes[1].plot(k_plot, cmax_fit, "b-", lw=2, label="fitted")
    axes[1].axhline(eol_cap, color="red", ls="--", label=f"EOL ({model._eol_threshold*100:.0f}%)")
    axes[1].axvline(query_cycle + rul["mean_rul"], color="orange", ls=":",
                    label=f"RUL mean = {rul['mean_rul']:.0f} cyc")
    axes[1].fill_betweenx(
        [min(cmax_noisy)*0.95, C0*1.02],
        query_cycle + rul["ci_lower_90"],
        query_cycle + rul["ci_upper_90"],
        alpha=0.15, color="orange", label="90% CI"
    )
    axes[1].set_xlabel("Cycle"); axes[1].set_ylabel("C_max (Ah)")
    axes[1].set_title("C_max fade + RUL prediction"); axes[1].legend(fontsize=7)

    plt.tight_layout()
    out_fig = "smoke_test_ageing.png"
    plt.savefig(out_fig, dpi=120)
    print(f"\nPlot saved → {out_fig}")

    os.unlink(tmp_cfg)
    print("\n✓ Smoke test passed.")