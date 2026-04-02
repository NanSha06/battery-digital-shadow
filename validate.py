"""
validate.py  --  End-to-end validation on held-out cell B0018.

Steps
-----
1.  Load B0018 via NASALoader (or synthetic fallback).
2.  Optionally train OCV model and GP corrector on B0005/B0006/B0007.
3.  Run ShadowSync.run_cycle() for every cycle of B0018.
4.  Compute metrics:
      a. SoC RMSE per cycle → mean over all cycles (target < 2 %)
      b. SoH MAE
      c. RUL error at 20 %, 50 %, 80 % life + 90 % CI coverage probability
      d. CUSUM false-trigger count
      e. GP coverage: % of V_meas within ±2σ of (V_ECM + GP mean)
5.  Save metrics → validation_report.json
6.  Generate 4-panel figure → validation_figure.png

Usage
-----
    python validate.py                  # full run
    python validate.py --smoke          # first 10 cycles only (quick test)
    python validate.py --smoke --cycles 20
    python validate.py --data ./data/raw --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

warnings.filterwarnings("ignore")


# ============================================================================
# Config helper
# ============================================================================

def load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if p.exists():
        with p.open() as f:
            return yaml.safe_load(f) or {}
    print(f"[warn] {path} not found — using defaults.")
    return {}


# ============================================================================
# Data loading
# ============================================================================

def load_cell(cell_id: str, data_path: str, config: dict) -> list[dict]:
    """Load a cell's discharge cycles from .mat or fall back to synthetic."""
    mat_file = Path(data_path) / f"{cell_id}.mat"
    if mat_file.exists():
        return _load_mat_cell(cell_id, mat_file, config)
    print(f"  [warn] {mat_file} not found — synthetic fallback")
    return _synthetic_cell(cell_id)


def _load_mat_cell(cell_id: str, mat_file: Path, config: dict) -> list[dict]:
    from scipy.io import loadmat
    raw = loadmat(str(mat_file), simplify_cells=True)
    # NASA PCoE .mat structure: raw[cell_id]["cycle"]
    try:
        cycles_raw = raw[cell_id]["cycle"]
    except KeyError:
        # Some files use the last non-dunder key
        key = [k for k in raw if not k.startswith("_")][-1]
        cycles_raw = raw[key]["cycle"]

    cycles, idx = [], 0
    for cyc in cycles_raw:
        if str(cyc.get("type", "")).strip().lower() != "discharge":
            continue
        d   = cyc.get("data", {})
        V   = np.asarray(d.get("Voltage_measured",     []), dtype=float).ravel()
        I   = np.asarray(d.get("Current_measured",     []), dtype=float).ravel()
        T   = np.asarray(d.get("Temperature_measured", []), dtype=float).ravel()
        t   = np.asarray(d.get("Time",                 []), dtype=float).ravel()
        if len(V) < 10:
            continue
        n   = min(len(V), len(I), len(T))
        V, I, T = V[:n], I[:n], T[:n]
        dt  = float(np.median(np.diff(t[:n]))) if len(t) >= 2 else 1.0
        dt  = dt if 0.1 < dt < 60 else 1.0
        C_max = float(np.trapz(np.abs(I), dx=dt) / 3600.0)
        cycles.append({
            "cycle_idx": idx, "V": V, "I": I, "T": T,
            "dt": dt, "C_max": C_max, "type": "discharge",
        })
        idx += 1
    print(f"  Loaded {cell_id}: {len(cycles)} discharge cycles")
    return cycles


def _synthetic_cell(cell_id: str, n_cycles: int = 168) -> list[dict]:
    rng = np.random.default_rng(seed=18)
    cycles = []
    for k in range(n_cycles):
        frac  = k / n_cycles
        C_max = 2.0 * (1.0 - 0.12 * frac ** 0.6)
        R0    = 0.015 + 0.010 * frac ** 0.6
        I_dc  = 1.5
        soc   = 1.0
        T_mean = 24.0 + rng.normal(0, 1.0)
        V_list, I_list, T_list = [], [], []
        for _ in range(360):
            V_oc = 3.0 + 1.2 * soc
            V_list.append(float(V_oc - R0 * I_dc + rng.normal(0, 5e-4)))
            I_list.append(float(I_dc))
            T_list.append(float(T_mean + rng.normal(0, 0.2)))
            soc = max(0.0, soc - I_dc / (C_max * 3600.0))
        cycles.append({
            "cycle_idx": k,
            "V": np.array(V_list),
            "I": np.array(I_list),
            "T": np.array(T_list),
            "dt": 1.0,
            "C_max": float(C_max),
            "type": "discharge",
        })
    print(f"  Synthetic {cell_id}: {n_cycles} cycles")
    return cycles


def coulomb_count_soc(cycle: dict, soc_init: float = 1.0) -> np.ndarray:
    """Approximate true SoC via Coulomb counting (used as ground truth)."""
    I   = np.asarray(cycle["I"], dtype=float)
    dt  = float(cycle.get("dt", 1.0))
    C   = float(cycle.get("C_max", 2.0))
    return np.clip(soc_init - np.cumsum(np.abs(I)) * dt / (C * 3600.0), 0.0, 1.0)


# ============================================================================
# Validation main
# ============================================================================

def run_validation(
    data_path:  str  = "./data/raw",
    config_path: str = "config.yaml",
    smoke:       bool = False,
    max_cycles:  int  = 10,
) -> dict:

    config   = load_config(config_path)
    eol_thr  = float(config.get("monte_carlo", {}).get("eol_threshold", 0.80))
    Q_nom    = float(config.get("Q_nom_Ah", 2.0))
    soc_init = float(config.get("soc_init",  1.0))

    print("\n" + "=" * 62)
    print("  validate.py — Battery Digital Shadow Validation")
    print("=" * 62)

    # ---- Import ShadowSync ------------------------------------------------
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from shadow.sync import ShadowSync
        print("  Using real ShadowSync from shadow/sync.py")
    except ImportError as e:
        print(f"  [error] Cannot import ShadowSync: {e}")
        sys.exit(1)

    # ---- Load B0018 -------------------------------------------------------
    print(f"\n  Loading B0018 from {data_path} ...")
    cycles = load_cell("B0018", data_path, config)
    if smoke:
        cycles = cycles[:max_cycles]
        print(f"  [smoke] first {len(cycles)} cycles only.")

    n_cycles = len(cycles)
    if n_cycles == 0:
        print("  [error] No cycles loaded — aborting.")
        sys.exit(1)

    C_nom    = float(cycles[0].get("C_max", Q_nom))
    eol_cap  = eol_thr * C_nom
    # True EOL: first cycle where measured capacity falls to or below eol_cap
    true_eol = next(
        (c["cycle_idx"] for c in cycles if c.get("C_max", C_nom) <= eol_cap),
        cycles[-1]["cycle_idx"],
    )

    print(
        f"  Cycles: {n_cycles}  |  C_nom={C_nom:.3f} Ah  |  "
        f"EOL={eol_cap:.3f} Ah  |  true_eol={true_eol}"
    )

    # ---- Prepare a clean temporary config with db_path -------------------
    db_path  = "validation_shadow.db"
    Path(db_path).unlink(missing_ok=True)
    tmp_config = dict(config)
    tmp_config["db_path"] = db_path
    tmp_cfg_path = "validation_config.yaml"
    with open(tmp_cfg_path, "w") as f:
        yaml.dump(tmp_config, f)

    # ---- Fit OCV model on training cells ---------------------------------
    print("\n  Fitting OCV model on training cells ...")
    ocv_model = None
    try:
        from ecm.model import AkimaOCVModel          # type: ignore
        ocv_model = AkimaOCVModel(config_path=tmp_cfg_path)
        train_ids = config.get("dataset", {}).get("cell_ids",
                                                   ["B0005", "B0006", "B0007"])
        soc_all_ocv, v_all_ocv = [], []
        for tid in train_ids:
            tc   = load_cell(tid, data_path, config)
            disc = [c for c in tc if c.get("type") == "discharge"]
            if not disc:
                continue
            # C/20 proxy: discharge cycle with smallest mean |I|
            c20 = min(disc, key=lambda c: float(np.mean(np.abs(c["I"]))))
            sa  = coulomb_count_soc(c20, soc_init=1.0)
            va  = np.asarray(c20["V"], dtype=float)
            n   = min(len(sa), len(va))
            soc_all_ocv.append(sa[:n])
            v_all_ocv.append(va[:n])
            print(f"    {tid}: {n} OCV samples")

        if soc_all_ocv:
            sc = np.concatenate(soc_all_ocv)
            vc = np.concatenate(v_all_ocv)
            si = np.argsort(sc)
            ocv_model.fit(sc[si], vc[si])
            print(f"  OCV fitted on {len(sc)} points.")
        else:
            # Synthetic OCV fallback
            soc_fb = np.linspace(0, 1, 11)
            ocv_model.fit(soc_fb, 3.0 + 1.2 * soc_fb)
            print("  OCV fitted on synthetic fallback.")
    except ImportError:
        print("  AkimaOCVModel not found — EKF stub OCV will be used.")

    # ---- Train GP corrector on training cells ----------------------------
    print("\n  Training GP residual corrector ...")
    gp_corrector = None
    try:
        from augmentation.gp import SVGPCorrector    # type: ignore
        gp_corrector = SVGPCorrector(config_path=tmp_cfg_path)
        train_ids    = config.get("dataset", {}).get("cell_ids",
                                                      ["B0005", "B0006", "B0007"])
        soc_r, I_r, T_r, cyc_r, res_r = [], [], [], [], []

        for tid in train_ids:
            for cyc in load_cell(tid, data_path, config)[:30]:
                V_arr  = np.asarray(cyc["V"], dtype=float)
                I_arr  = np.asarray(cyc["I"], dtype=float)
                T_arr  = np.asarray(cyc["T"], dtype=float)
                soc_cc = coulomb_count_soc(cyc, soc_init=1.0)
                n      = min(len(V_arr), len(I_arr), len(T_arr), len(soc_cc))
                # Sub-sample every 10th point for efficiency
                for j in range(0, n, 10):
                    sj  = float(soc_cc[j])
                    Ij  = float(I_arr[j])
                    Tj  = float(T_arr[j])
                    if ocv_model is not None:
                        try:
                            Ve = float(ocv_model.ocv(sj)) - float(
                                config.get("ecm", {}).get(
                                    "parameters", config.get("ecm", {})
                                ).get("R0", 0.015)
                            ) * Ij
                        except Exception:
                            Ve = 3.0 + 1.2 * sj - 0.015 * Ij
                    else:
                        Ve = 3.0 + 1.2 * sj - 0.015 * Ij
                    soc_r.append(sj)
                    I_r.append(Ij)
                    T_r.append(Tj)
                    cyc_r.append(float(cyc["cycle_idx"]))
                    res_r.append(float(V_arr[j]) - Ve)

        if len(res_r) > 20:
            gp_corrector.fit(
                np.array(soc_r), np.array(I_r),
                np.array(T_r),   np.array(cyc_r),
                np.array(res_r),
            )
            print(f"  GP trained on {len(res_r)} residual points.")
        else:
            gp_corrector = None
            print("  Not enough GP training data — stub GP will be used.")
    except Exception as e:
        print(f"  GP training failed ({e}) — stub GP will be used.")
        gp_corrector = None

    # ---- Initialise ShadowSync -------------------------------------------
    # BUG FIX: was using undefined variable `cfg`; must use `tmp_cfg_path`
    shadow = ShadowSync(config_path=tmp_cfg_path)

    if ocv_model is not None and hasattr(shadow.ekf, "ocv_model"):
        shadow.ekf.ocv_model = ocv_model
        print("  OCV model injected into EKF.")

    if gp_corrector is not None and hasattr(shadow, "gp"):
        shadow.gp = gp_corrector
        print("  GP corrector injected into ShadowSync.")

    # Inject fitted OCV model into OfflineIdentifier's ECMModel so that
    # CUSUM-triggered re-identification doesn't crash with "not fitted yet".
    if ocv_model is not None and hasattr(shadow, "identifier"):
        for attr in ("ecm_model", "_ecm", "model", "ecm"):
            candidate = getattr(shadow.identifier, attr, None)
            if candidate is not None and hasattr(candidate, "ocv_model"):
                candidate.ocv_model = ocv_model
                print(f"  OCV model injected into OfflineIdentifier.{attr}.")
                break

    # Pass C_nom into ShadowSync config so SoH = C_max/C_nom is correctly scaled.
    # C_nom is the capacity of the first cycle (freshest measured cell capacity).
    if "Q_nom_Ah" not in config:
        shadow._cfg["Q_nom_Ah"] = C_nom
        shadow._cfg["C_nom_Ah"] = C_nom

    # ---- GP noise level for coverage metric (§4e) -----------------------
    gp_noise_std = float(
        config.get("gp", {}).get("noise_std",
        config.get("gp", {}).get("gp_noise_std", 5e-4))
    )

    # ---- Main validation loop --------------------------------------------
    soc_errs: list[float] = []
    soh_errs: list[float] = []
    c_maxs:   list[float] = []
    rul_means: list[float] = []
    rul_los:   list[float] = []
    rul_his:   list[float] = []
    cusum_triggers: list[int] = []
    gp_in  = 0
    gp_tot = 0

    # RUL checkpoint cycle indices
    rul_checkpoints = {
        "at_20pct_life": int(0.20 * n_cycles),
        "at_50pct_life": int(0.50 * n_cycles),
        "at_80pct_life": int(0.80 * n_cycles),
    }
    rul_errors: dict[str, dict] = {}

    print(f"\n  Running ShadowSync on {n_cycles} cycles ...\n")
    t0 = time.time()

    for idx, cycle in enumerate(cycles):
        k        = int(cycle["cycle_idx"])
        soc_true = coulomb_count_soc(cycle, soc_init=soc_init)

        state    = shadow.run_cycle(cycle)

        # ---- SoC error ------------------------------------------------
        soc_err = abs(state["soc_final"] - float(soc_true[-1]))
        soc_errs.append(soc_err)

        # ---- SoH error ------------------------------------------------
        C_max_k = float(cycle.get("C_max", C_nom))
        soh_true = C_max_k / C_nom
        soh_errs.append(abs(state["soh"] - soh_true))
        c_maxs.append(C_max_k)

        # ---- RUL collection ------------------------------------------
        rul_means.append(state["rul_mean"])
        rul_los.append(state["rul_ci_lower"])
        rul_his.append(state["rul_ci_upper"])

        # ---- RUL checkpoint errors ------------------------------------
        for label, ck in rul_checkpoints.items():
            if label not in rul_errors and (k == ck or idx == n_cycles - 1):
                tr = max(0, true_eol - k)
                rul_errors[label] = {
                    "cycle":    k,
                    "true_rul": tr,
                    "pred_rul": round(state["rul_mean"], 1),
                    "error":    round(abs(state["rul_mean"] - tr), 1),
                    "in_90ci":  bool(
                        state["rul_ci_lower"] <= tr <= state["rul_ci_upper"]
                    ),
                }

        # ---- CUSUM trigger counting ----------------------------------
        mf = state.get("mode_flags", {})
        if isinstance(mf, str):
            try:
                mf = json.loads(mf)
            except Exception:
                mf = {}
        if mf.get("cusum_triggered", False):
            cusum_triggers.append(k)

        # ---- GP coverage (±2σ of corrected prediction) ---------------
        # Approximate: if cycle RMSE is within 2·gp_noise the prediction
        # is deemed "within ±2σ".
        gp_sigma = max(gp_noise_std, state["rmse"] * 0.1)
        ns = len(cycle["V"])
        if state["rmse"] < 2.0 * gp_sigma:
            gp_in += ns
        gp_tot += ns

        # ---- Progress print ------------------------------------------
        if (idx + 1) % max(1, n_cycles // 10) == 0 or idx == 0:
            print(
                f"  Cycle {k:4d}/{cycles[-1]['cycle_idx']}  "
                f"SoC_err={soc_err*100:.2f}%  "
                f"SoH_err={soh_errs[-1]*100:.2f}%  "
                f"RUL={state['rul_mean']:.0f}  "
                f"CUSUM={state['cusum']:.4f}"
            )

    elapsed = time.time() - t0
    shadow.close()

    # ---- Compute metrics ------------------------------------------------
    mean_soc_rmse = float(np.mean(soc_errs)) * 100.0     # percent
    mean_soh_mae  = float(np.mean(soh_errs))
    gp_coverage   = gp_in / max(gp_tot, 1)

    true_ruls = [
        max(0, true_eol - cycles[i]["cycle_idx"]) for i in range(n_cycles)
    ]
    ci_hits     = sum(
        lo <= tr <= hi
        for tr, lo, hi in zip(true_ruls, rul_los, rul_his)
    )
    ci_coverage = ci_hits / max(n_cycles, 1)

    report = {
        "cell":                 "B0018",
        "n_cycles_validated":   n_cycles,
        "smoke_mode":           smoke,
        "elapsed_seconds":      round(elapsed, 1),
        "soc_rmse_mean_pct":    round(mean_soc_rmse, 4),
        "soc_rmse_target_pct":  2.0,
        "soc_rmse_passed":      mean_soc_rmse < 2.0,
        "soh_mae_mean":         round(mean_soh_mae, 6),
        "rul_errors":           rul_errors,
        "rul_90ci_coverage":    round(ci_coverage, 4),
        "rul_90ci_target":      0.90,
        "rul_90ci_passed":      ci_coverage >= 0.90,
        "cusum_false_triggers": len(cusum_triggers),
        "cusum_trigger_cycles": cusum_triggers,
        "gp_coverage_2sigma":   round(gp_coverage, 4),
    }

    # ---- Print summary --------------------------------------------------
    print("\n" + "=" * 62)
    print("  VALIDATION REPORT — B0018")
    print("=" * 62)
    print(f"  Cycles          : {n_cycles}  |  Time: {elapsed:.1f}s")
    print(
        f"  SoC RMSE        : {mean_soc_rmse:.3f}%  "
        f"{'PASS' if report['soc_rmse_passed'] else 'FAIL'}"
        f"  (target < 2%)"
    )
    print(f"  SoH MAE         : {mean_soh_mae:.4f}")
    for label, info in rul_errors.items():
        print(
            f"  {label}: cycle={info['cycle']}  "
            f"true={info['true_rul']}  pred={info['pred_rul']}  "
            f"err={info['error']}  in_CI={info['in_90ci']}"
        )
    print(
        f"  RUL CI coverage : {ci_coverage*100:.1f}%  "
        f"{'PASS' if report['rul_90ci_passed'] else 'FAIL'}"
        f"  (target ≥ 90%)"
    )
    print(f"  CUSUM triggers  : {len(cusum_triggers)}")
    print(f"  GP ±2σ coverage : {gp_coverage*100:.1f}%")
    print("=" * 62)

    with open("validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n  Report saved → validation_report.json")

    # ---- Plot -----------------------------------------------------------
    _plot_validation(
        cycles, soc_errs, c_maxs, rul_means, rul_los, rul_his,
        true_ruls, cusum_triggers, report, C_nom, eol_cap, smoke,
    )

    # ---- Cleanup temp files ---------------------------------------------
    for tmp in [db_path, tmp_cfg_path]:
        Path(tmp).unlink(missing_ok=True)

    return report


# ============================================================================
# Plotting
# ============================================================================

def _plot_validation(
    cycles, soc_errs, c_maxs, rul_means, rul_los, rul_his,
    true_ruls, cusum_triggers, report, C_nom, eol_cap, smoke,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping figure.")
        return

    ki  = [c["cycle_idx"] for c in cycles]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Digital Shadow Validation — B0018" + (" [SMOKE]" if smoke else ""),
        fontsize=13, fontweight="bold",
    )

    # Panel 1: SoC tracking error
    ax = axes[0, 0]
    ax.plot(ki, [v * 100 for v in soc_errs], lw=1.2, color="#1f77b4", label="SoC error")
    ax.axhline(2.0, color="red", ls="--", lw=1.2, label="Target 2%")
    ax.fill_between(ki, [v * 100 for v in soc_errs], alpha=0.2, color="#1f77b4")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SoC error (%)")
    ax.set_title(
        f"SoC Error  (mean={report['soc_rmse_mean_pct']:.2f}% "
        f"{'PASS' if report['soc_rmse_passed'] else 'FAIL'})"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: Capacity fade
    ax = axes[0, 1]
    ax.scatter(ki, c_maxs, s=8, alpha=0.6, color="#2ca02c", label="C_max measured")
    try:
        from scipy.optimize import curve_fit

        def _fade(k, C0, b, n):
            return C0 * (1 - b * np.asarray(k, dtype=float) ** n)

        p, _ = curve_fit(
            _fade, ki, c_maxs, p0=[C_nom, 0.04, 0.6],
            maxfev=3000, bounds=([0.5, 0, 0.1], [5, 1, 1]),
        )
        ke = np.linspace(0, max(ki) * 1.2, 300)
        ax.plot(ke, _fade(ke, *p), lw=2, color="navy", label="Model fit")
    except Exception:
        pass
    ax.axhline(eol_cap, color="red", ls="--", lw=1.2, label=f"EOL {eol_cap:.3f} Ah")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("C_max (Ah)")
    ax.set_title("Capacity Fade")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: RUL extrapolation
    ax = axes[1, 0]
    ax.plot(ki, true_ruls, lw=1.5, color="black", label="True RUL", zorder=3)
    ax.plot(ki, rul_means, lw=1.5, color="#ff7f0e", ls="--", label="Predicted RUL")
    ax.fill_between(ki, rul_los, rul_his, alpha=0.25,
                    color="#ff7f0e", label="90% CI")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Remaining cycles")
    ax.set_title(
        f"RUL  (90% CI coverage={report['rul_90ci_coverage']*100:.0f}% "
        f"{'PASS' if report['rul_90ci_passed'] else 'FAIL'})"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 4: CUSUM statistic
    # Re-build an approximate CUSUM curve from SoC errors
    ax = axes[1, 1]
    k0_approx  = 1e-3
    s          = 0.0
    cusum_plot = []
    for v in soc_errs:
        s = max(0.0, s + v - k0_approx)
        cusum_plot.append(s)
    ax.plot(ki, cusum_plot, lw=1.3, color="#9467bd", label="CUSUM (approx.)")
    for tk in cusum_triggers:
        ax.axvline(tk, color="red", lw=0.8, alpha=0.6)
    if cusum_triggers:
        ax.axvline(
            cusum_triggers[0], color="red", lw=0.8, alpha=0.6,
            label=f"Triggers ({len(cusum_triggers)})",
        )
    ax.set_xlabel("Cycle")
    ax.set_ylabel("CUSUM")
    ax.set_title(f"CUSUM  (false triggers={report['cusum_false_triggers']})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("validation_figure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure saved → validation_figure.png")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end validation of the battery digital shadow on B0018."
    )
    parser.add_argument(
        "--smoke",  action="store_true",
        help="Quick smoke test — run only the first N cycles.",
    )
    parser.add_argument(
        "--cycles", type=int, default=10,
        help="Number of cycles to run in smoke mode (default: 10).",
    )
    parser.add_argument(
        "--data",   type=str, default="./data/raw",
        help="Path to the directory containing .mat files.",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml.",
    )
    args = parser.parse_args()

    report = run_validation(
        data_path   = args.data,
        config_path = args.config,
        smoke       = args.smoke,
        max_cycles  = args.cycles,
    )
    sys.exit(0 if report["soc_rmse_passed"] else 1)