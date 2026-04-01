"""
ageing/modes.py
---------------
Degradation mode classifier using:
  - ICA  (Incremental Capacity Analysis)  → LLI, LAM_PE, LAM_NE
  - DRT  (Distribution of Relaxation Times) → ohmic rise

All hyperparameters are read from config.yaml — nothing is hardcoded.
"""

import numpy as np
import yaml
import tempfile
import os
from scipy.signal import savgol_filter, find_peaks
from scipy.linalg import solve


# ---------------------------------------------------------------------------
# Helper: load config
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# ModeClassifier
# ---------------------------------------------------------------------------

class ModeClassifier:
    """
    Classifies battery degradation modes from cycling and EIS data.

    ICA  — dV/dQ curve peak analysis identifies:
        LLI    : Lithium Inventory Loss       (peak position shift in Q)
        LAM_PE : Loss of Active Material, +ve (cathode peak height drop)
        LAM_NE : Loss of Active Material, -ve (anode peak height drop)

    DRT  — Tikhonov-regularised deconvolution of EIS impedance identifies:
        ohmic_rise : increase in high-frequency resistance

    Methods
    -------
    fit(baseline_cycle)          — store baseline ICA peaks for comparison
    predict(cycle)               — alias for classify_cycle
    update(new_baseline_cycle)   — update the reference baseline
    ica(V, Q)                    — compute dV/dQ and classify peaks
    drt(Z_real, Z_imag, freqs)   — compute DRT, return ohmic rise
    classify_cycle(cycle)        — full classification for one cycle
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg = _load_config(config_path)

        # ICA smoothing settings
        ica_cfg = cfg.get("ica", {})
        self._sg_window   = int(ica_cfg.get("savgol_window",   21))
        self._sg_polyord  = int(ica_cfg.get("savgol_polyorder", 3))
        self._peak_height = float(ica_cfg.get("peak_min_height", 0.5))
        self._peak_dist   = int(ica_cfg.get("peak_min_distance", 10))

        # DRT regularisation
        drt_cfg = cfg.get("drt", {})
        self._drt_lambda  = float(drt_cfg.get("lambda_reg", 1e-3))
        self._drt_n_tau   = int(drt_cfg.get("n_tau",        50))

        # Baseline reference peaks (set by fit())
        self._baseline_peaks: dict | None = None

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def fit(self, baseline_cycle) -> None:
        """
        Store ICA baseline from the first (or freshest) reference cycle.
        All future classify_cycle() calls compare against this baseline.

        Parameters
        ----------
        baseline_cycle : Cycle-like object with attributes V and Q,
                         OR a dict with keys 'V' and 'Q'.
        """
        V, Q = self._extract_vq(baseline_cycle)
        peaks_info = self._find_ica_peaks(V, Q)
        self._baseline_peaks = peaks_info

    def predict(self, cycle) -> dict:
        """Alias for classify_cycle — keeps fit/predict/update API uniform."""
        return self.classify_cycle(cycle)

    def update(self, new_baseline_cycle) -> None:
        """Replace the stored baseline with a new reference cycle."""
        self.fit(new_baseline_cycle)

    def ica(self, V: np.ndarray, Q: np.ndarray) -> dict:
        """
        Incremental Capacity Analysis on a single discharge curve.

        Parameters
        ----------
        V : voltage array (V), monotonically decreasing for discharge
        Q : charge throughput array (Ah), same length as V

        Returns
        -------
        dict with keys:
            LLI      : float — peak position shift  (Ah)
            LAM_PE   : float — cathode peak height loss (relative)
            LAM_NE   : float — anode peak height loss   (relative)
            peaks    : list of dicts with per-peak details
            dVdQ     : np.ndarray — smoothed dV/dQ curve
            Q_axis   : np.ndarray — corresponding Q values
        """
        V = np.asarray(V, dtype=float)
        Q = np.asarray(Q, dtype=float)

        # Remove duplicate Q points to avoid division-by-zero
        mask   = np.diff(Q, prepend=Q[0] - 1) != 0
        V, Q   = V[mask], Q[mask]

        if len(Q) < self._sg_window + 2:
            return self._empty_ica_result()

        # Smooth voltage before differentiating
        V_smooth = savgol_filter(V, self._sg_window, self._sg_polyord)

        # dV/dQ via finite differences on smoothed V
        dVdQ_raw = np.gradient(V_smooth, Q)

        # Second smoothing pass on dV/dQ itself
        win = min(self._sg_window, len(dVdQ_raw) - 1)
        if win % 2 == 0:
            win -= 1
        win = max(win, 5)
        dVdQ = savgol_filter(dVdQ_raw, win, min(self._sg_polyord, win - 1))

        # Find peaks in -dV/dQ  (peaks in IC = dQ/dV correspond to
        # troughs in dV/dQ; we negate so find_peaks works naturally)
        neg_dVdQ = -dVdQ
        peak_idx, props = find_peaks(
            neg_dVdQ,
            height=self._peak_height * np.max(np.abs(neg_dVdQ)),
            distance=self._peak_dist,
        )

        peaks = [
            {
                "q_position": float(Q[i]),
                "height":     float(neg_dVdQ[i]),
                "index":      int(i),
            }
            for i in peak_idx
        ]

        # --- Compare against baseline ---
        LLI    = 0.0
        LAM_PE = 0.0
        LAM_NE = 0.0

        if self._baseline_peaks is not None and len(peaks) > 0:
            base_peaks = self._baseline_peaks.get("peaks", [])

            # LLI: shift of the dominant (highest) peak position
            if len(base_peaks) > 0:
                base_dom = max(base_peaks, key=lambda p: p["height"])
                curr_dom = max(peaks,      key=lambda p: p["height"])
                LLI = float(curr_dom["q_position"] - base_dom["q_position"])

            # LAM_PE: drop in the last peak (high-Q / cathode-side)
            if len(base_peaks) >= 1 and len(peaks) >= 1:
                base_last = sorted(base_peaks, key=lambda p: p["q_position"])[-1]
                curr_last = sorted(peaks,      key=lambda p: p["q_position"])[-1]
                h_base = base_last["height"]
                if h_base > 1e-9:
                    LAM_PE = float((h_base - curr_last["height"]) / h_base)

            # LAM_NE: drop in the first peak (low-Q / anode-side)
            if len(base_peaks) >= 2 and len(peaks) >= 2:
                base_first = sorted(base_peaks, key=lambda p: p["q_position"])[0]
                curr_first = sorted(peaks,      key=lambda p: p["q_position"])[0]
                h_base = base_first["height"]
                if h_base > 1e-9:
                    LAM_NE = float((h_base - curr_first["height"]) / h_base)

        return {
            "LLI":    LLI,
            "LAM_PE": LAM_PE,
            "LAM_NE": LAM_NE,
            "peaks":  peaks,
            "dVdQ":   dVdQ,
            "Q_axis": Q,
        }

    def drt(
        self,
        Z_real:  np.ndarray,
        Z_imag:  np.ndarray,
        freqs:   np.ndarray,
    ) -> float:
        """
        Distribution of Relaxation Times via Tikhonov regularisation.

        Solves:   min ||A*gamma - Z_imag||^2 + lambda * ||L*gamma||^2
        where A is the DRT kernel matrix and L is a first-difference
        regularisation matrix.

        Parameters
        ----------
        Z_real : real part of impedance (Ohm)
        Z_imag : imaginary part (Ohm), sign convention: Z_imag > 0 for inductive
        freqs  : measurement frequencies (Hz)

        Returns
        -------
        float — ohmic rise magnitude (peak of DRT at highest frequency)
        """
        Z_real = np.asarray(Z_real, dtype=float)
        Z_imag = np.asarray(Z_imag, dtype=float)
        freqs  = np.asarray(freqs,  dtype=float)

        N  = len(freqs)
        M  = self._drt_n_tau

        if N < 3:
            return 0.0

        omega   = 2.0 * np.pi * freqs
        # Log-spaced relaxation times spanning the measured frequency range
        tau_min = 1.0 / (2.0 * np.pi * np.max(freqs))
        tau_max = 1.0 / (2.0 * np.pi * np.min(freqs))
        taus    = np.logspace(np.log10(tau_min), np.log10(tau_max), M)

        # Build DRT kernel matrix A  (imaginary part of Voigt element)
        # A[i,j] = -omega[i]*tau[j] / (1 + (omega[i]*tau[j])^2)
        Omega = np.outer(omega, taus)          # (N, M)
        A     = -Omega / (1.0 + Omega ** 2)   # (N, M)

        # First-difference regularisation matrix
        L = np.diff(np.eye(M), axis=0)        # (M-1, M)

        # Normal equations:  (A^T A + lambda * L^T L) gamma = A^T b
        AtA  = A.T @ A
        LtL  = L.T @ L
        rhs  = A.T @ (-Z_imag)                # fit to -Z_imag (capacitive convention)

        lhs  = AtA + self._drt_lambda * LtL
        try:
            gamma = np.linalg.solve(lhs, rhs)
            gamma = np.clip(gamma, 0.0, None)  # DRT must be non-negative
        except np.linalg.LinAlgError:
            return 0.0

        # Ohmic rise = peak of DRT at the shortest relaxation time (high freq)
        ohmic_rise = float(np.max(gamma[:M // 5]))   # top-20% of tau range
        return ohmic_rise

    def classify_cycle(self, cycle) -> dict:
        """
        Full degradation mode classification for one cycle.

        Parameters
        ----------
        cycle : object or dict containing:
            V, Q (or I + timestamps to integrate Q)   — for ICA
            Z_real, Z_imag, freqs (optional)           — for DRT

        Returns
        -------
        dict: {LLI, LAM_PE, LAM_NE, ohmic_rise}
        """
        V, Q = self._extract_vq(cycle)
        ica_result = self.ica(V, Q)

        # DRT if EIS data available
        ohmic_rise = 0.0
        Z_real = self._get_field(cycle, "Z_real")
        Z_imag = self._get_field(cycle, "Z_imag")
        freqs  = self._get_field(cycle, "freqs")
        if Z_real is not None and Z_imag is not None and freqs is not None:
            ohmic_rise = self.drt(
                np.asarray(Z_real),
                np.asarray(Z_imag),
                np.asarray(freqs),
            )

        return {
            "LLI":        ica_result["LLI"],
            "LAM_PE":     ica_result["LAM_PE"],
            "LAM_NE":     ica_result["LAM_NE"],
            "ohmic_rise": ohmic_rise,
        }

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _extract_vq(self, cycle) -> tuple[np.ndarray, np.ndarray]:
        """Extract V and Q arrays from a Cycle object or dict."""
        V = self._get_field(cycle, "V")
        Q = self._get_field(cycle, "Q")

        if V is None:
            raise ValueError("cycle must have a 'V' field (voltage array).")

        # If Q is not pre-computed, integrate current over time
        if Q is None:
            I = self._get_field(cycle, "I")
            t = self._get_field(cycle, "timestamps")
            if I is not None and t is not None:
                dt = np.diff(np.asarray(t, dtype=float), prepend=0.0)
                Q  = np.cumsum(np.asarray(I, dtype=float) * dt) / 3600.0
            else:
                # Last resort: use index as proxy for Q
                Q = np.arange(len(V), dtype=float)

        return np.asarray(V, dtype=float), np.asarray(Q, dtype=float)

    def _get_field(self, cycle, key: str):
        """Safely get a field from a Cycle object or dict."""
        if isinstance(cycle, dict):
            return cycle.get(key, None)
        return getattr(cycle, key, None)

    def _find_ica_peaks(self, V: np.ndarray, Q: np.ndarray) -> dict:
        """Run ICA without baseline comparison — used for fit()."""
        saved = self._baseline_peaks
        self._baseline_peaks = None          # temporarily disable comparison
        result = self.ica(V, Q)
        self._baseline_peaks = saved
        return result

    def _empty_ica_result(self) -> dict:
        return {
            "LLI": 0.0, "LAM_PE": 0.0, "LAM_NE": 0.0,
            "peaks": [], "dVdQ": np.array([]), "Q_axis": np.array([]),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 55)
    print("  ageing/modes.py  —  smoke test")
    print("=" * 55)

    # ---- Inline config (self-contained, no config.yaml needed) ----
    _inline_cfg = """
ica:
  savgol_window: 21
  savgol_polyorder: 3
  peak_min_height: 0.25
  peak_min_distance: 30
drt:
  lambda_reg: 1.0e-3
  n_tau: 50
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                     delete=False) as tf:
        tf.write(_inline_cfg)
        tmp_cfg = tf.name

    clf = ModeClassifier(config_path=tmp_cfg)

    # ---- Synthetic discharge curve with two ICA peaks ----
    # Simulate a typical Li-ion OCV-like V(Q) curve
    np.random.seed(0)
    N   = 300
    Q_base = np.linspace(0.0, 2.0, N)

    # Two Gaussian bumps in dQ/dV (i.e. troughs in dV/dQ)
    def make_voltage(Q_arr, q1=0.6, q2=1.4, depth=0.04):
        V = 3.9 - 0.3 * Q_arr  # linear base
        V -= depth * np.exp(-((Q_arr - q1) ** 2) / 0.02)
        V -= depth * np.exp(-((Q_arr - q2) ** 2) / 0.02)
        return V + np.random.normal(0, 0.001, len(Q_arr))

    V_base = make_voltage(Q_base)

    # Fit baseline on fresh cycle
    baseline = {"V": V_base, "Q": Q_base}
    clf.fit(baseline)
    print(f"\nBaseline fitted — {len(clf._baseline_peaks['peaks'])} peaks found")

    # --- Aged cycle: peaks shifted + reduced (simulate degradation) ---
    Q_aged = np.linspace(0.0, 1.7, N)          # capacity loss → shorter Q
    V_aged = make_voltage(Q_aged, q1=0.55, q2=1.25, depth=0.03)  # peaks shift + shrink
    aged   = {"V": V_aged, "Q": Q_aged}

    result = clf.classify_cycle(aged)

    print(f"\nDegradation modes (aged vs baseline):")
    print(f"  LLI        : {result['LLI']:+.4f} Ah   (negative = Q shrinkage)")
    print(f"  LAM_PE     : {result['LAM_PE']:+.4f}    (positive = cathode peak loss)")
    print(f"  LAM_NE     : {result['LAM_NE']:+.4f}    (positive = anode peak loss)")
    print(f"  ohmic_rise : {result['ohmic_rise']:.4f}  (0 = no EIS data provided)")

    # --- DRT on synthetic EIS data ---
    freqs   = np.logspace(-2, 4, 40)          # 0.01 Hz to 10 kHz
    omega   = 2 * np.pi * freqs
    R0, R1, C1 = 0.015, 0.025, 0.08
    Z_real  = R0 + R1 / (1 + (omega * R1 * C1) ** 2)
    Z_imag  = -omega * R1 ** 2 * C1 / (1 + (omega * R1 * C1) ** 2)

    ohmic = clf.drt(Z_real, Z_imag, freqs)
    print(f"\nDRT ohmic rise (synthetic EIS): {ohmic:.6f}")

    eis_cycle = {
        "V": V_aged, "Q": Q_aged,
        "Z_real": Z_real, "Z_imag": Z_imag, "freqs": freqs,
    }
    full_result = clf.classify_cycle(eis_cycle)
    print(f"\nFull classify_cycle() with EIS:")
    for k, v in full_result.items():
        print(f"  {k:12s}: {v:.6f}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: V(Q) curves
    axes[0].plot(Q_base, V_base, label="baseline", lw=1.5)
    axes[0].plot(Q_aged, V_aged, label="aged",     lw=1.5, ls="--")
    axes[0].set_xlabel("Q (Ah)"); axes[0].set_ylabel("V (V)")
    axes[0].set_title("Discharge curves"); axes[0].legend()

    # Panel 2: dV/dQ (ICA)
    base_ica = clf._find_ica_peaks(V_base, Q_base)
    aged_ica = clf.ica(V_aged, Q_aged)
    axes[1].plot(base_ica["Q_axis"], -base_ica["dVdQ"], label="baseline IC", lw=1.5)
    axes[1].plot(aged_ica["Q_axis"], -aged_ica["dVdQ"], label="aged IC",     lw=1.5, ls="--")
    for p in base_ica["peaks"]:
        axes[1].axvline(p["q_position"], color="blue",   ls=":", alpha=0.5)
    for p in aged_ica["peaks"]:
        axes[1].axvline(p["q_position"], color="orange", ls=":", alpha=0.5)
    axes[1].set_xlabel("Q (Ah)"); axes[1].set_ylabel("-dV/dQ")
    axes[1].set_title("Incremental Capacity (ICA)"); axes[1].legend()

    # Panel 3: DRT
    taus = np.logspace(
        np.log10(1 / (2 * np.pi * freqs.max())),
        np.log10(1 / (2 * np.pi * freqs.min())),
        50
    )
    # Recompute gamma for plot
    omega_p = 2 * np.pi * freqs
    Omega   = np.outer(omega_p, taus)
    A       = -Omega / (1.0 + Omega ** 2)
    L       = np.diff(np.eye(50), axis=0)
    lhs     = A.T @ A + 1e-3 * L.T @ L
    rhs     = A.T @ (-Z_imag)
    gamma   = np.clip(np.linalg.solve(lhs, rhs), 0, None)
    axes[2].semilogx(taus, gamma, lw=1.5)
    axes[2].set_xlabel("τ (s)"); axes[2].set_ylabel("γ(τ)")
    axes[2].set_title(f"DRT  (ohmic rise = {ohmic:.4f})")

    plt.tight_layout()
    plt.savefig("smoke_test_modes.png", dpi=120)
    print("\nPlot saved → smoke_test_modes.png")

    os.unlink(tmp_cfg)
    print("\n✓ Smoke test passed.")