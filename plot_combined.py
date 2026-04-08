"""
plot_combined.py — Combined Plot: Voltage, Current & Resistance Degradation Rate
=================================================================================
Plots current and voltage variations across discharge cycles along with the
resistance degradation rate (dR₀/dcycle) — all on the SAME graph using a
triple-y-axis layout.

Usage:
    python plot_combined.py                  # Full run on B0018
    python plot_combined.py --smoke          # Quick smoke test (10 cycles)
    python plot_combined.py --cell B0005     # Use a different cell
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if p.exists():
        with p.open() as f:
            return yaml.safe_load(f) or {}
    print(f"[warn] {path} not found — using defaults.")
    return {}


def load_cell(cell_id: str, data_path: str, config: dict) -> list[dict]:
    mat_file = Path(data_path) / f"{cell_id}.mat"
    if not mat_file.exists():
        print(f"  [error] {mat_file} not found.")
        sys.exit(1)

    from scipy.io import loadmat
    raw = loadmat(str(mat_file), simplify_cells=True)
    try:
        cycles_raw = raw[cell_id]["cycle"]
    except KeyError:
        key = [k for k in raw if not k.startswith("_")][-1]
        cycles_raw = raw[key]["cycle"]

    cycles, idx = [], 0
    for cyc in cycles_raw:
        if str(cyc.get("type", "")).strip().lower() != "discharge":
            continue
        d = cyc.get("data", {})
        V = np.asarray(d.get("Voltage_measured", []), dtype=float).ravel()
        I = np.asarray(d.get("Current_measured", []), dtype=float).ravel()
        T = np.asarray(d.get("Temperature_measured", []), dtype=float).ravel()
        t = np.asarray(d.get("Time", []), dtype=float).ravel()
        if len(V) < 10:
            continue
        n = min(len(V), len(I), len(T))
        V, I, T = V[:n], I[:n], T[:n]
        dt = float(np.median(np.diff(t[:n]))) if len(t) >= 2 else 1.0
        dt = dt if 0.1 < dt < 60 else 1.0
        C_max = float(np.trapz(np.abs(I), dx=dt) / 3600.0)
        cycles.append({
            "cycle_idx": idx, "V": V, "I": I, "T": T,
            "dt": dt, "C_max": C_max, "type": "discharge",
        })
        idx += 1

    print(f"  Loaded {cell_id}: {len(cycles)} discharge cycles")
    return cycles


def _set_axis_limits_from_series(ax, values: np.ndarray, min_pad: float) -> None:
    """Apply a tight y-range so flat traces are not visually distorted."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return

    y_min = float(np.min(arr))
    y_max = float(np.max(arr))
    span = y_max - y_min
    pad = max(span * 0.15, min_pad)
    ax.set_ylim(y_min - pad, y_max + pad)


# ---------------------------------------------------------------------------
# Resistance estimation via physical ageing model
# ---------------------------------------------------------------------------

def compute_resistance_series(cycles: list[dict], config: dict) -> np.ndarray:
    """
    Estimate R₀ for each cycle using the physical ageing law from the
    digital shadow:  R₀(k) = R₀_fresh × (1 + α × k^β)
    """
    ecm_params = config.get("ecm", {}).get("parameters", {})
    r0_fresh = float(ecm_params.get("R0", 0.015))

    # Ageing parameters (matches shadow/sync.py)
    alpha = 0.0003
    beta = 0.5

    r0_arr = np.empty(len(cycles))
    for i, cyc in enumerate(cycles):
        k = float(cyc["cycle_idx"])
        r0_arr[i] = r0_fresh * (1.0 + alpha * k ** beta)

    # Monotonically non-decreasing (physical constraint)
    r0_arr = np.maximum.accumulate(r0_arr)
    return r0_arr


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_combined(
    cell_id: str = "B0018",
    data_path: str = "./data/raw",
    config_path: str = "config.yaml",
    smoke: bool = False,
    max_cycles: int = 10,
) -> None:

    config = load_config(config_path)
    print(f"\n{'='*62}")
    print(f"  Combined Plot: Voltage, Current & Resistance Degradation Rate")
    print(f"{'='*62}")
    print(f"  Cell: {cell_id}  |  Data: {data_path}")

    cycles = load_cell(cell_id, data_path, config)
    if smoke:
        cycles = cycles[:max_cycles]
        print(f"  [smoke] first {len(cycles)} cycles only.")

    n_cycles = len(cycles)
    if n_cycles == 0:
        print("  [error] No cycles loaded — aborting.")
        sys.exit(1)

    # -- Aggregate per-cycle statistics ---
    cycle_indices = np.array([c["cycle_idx"] for c in cycles], dtype=float)
    mean_voltage = np.array([np.mean(c["V"]) for c in cycles])
    mean_current = np.array([np.mean(np.abs(c["I"])) for c in cycles])

    # -- Resistance estimation ---
    r0_arr = compute_resistance_series(cycles, config)               # Ω

    # -- Resistance degradation rate ---
    dr0 = np.gradient(r0_arr, cycle_indices)                         # Ω/cycle
    from scipy.ndimage import gaussian_filter1d
    sigma = max(2.0, n_cycles / 50.0)
    dr0_smooth = gaussian_filter1d(dr0, sigma)

    # ====================================================================
    # PLOT — single graph, triple y-axes
    # ====================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    # Premium colour palette
    COLOR_VOLTAGE    = "#3B82F6"   # vivid blue
    COLOR_CURRENT    = "#10B981"   # emerald green
    COLOR_RESISTANCE = "#EF4444"   # warm red

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # ---- Dark premium background ----
    fig.patch.set_facecolor("#0F172A")
    ax1.set_facecolor("#1E293B")

    # ---- Axis 1: Voltage (left) ----
    ln1 = ax1.plot(cycle_indices, mean_voltage, color=COLOR_VOLTAGE,
                   linewidth=2, label="Mean Voltage (V)", zorder=3)
    ax1.fill_between(cycle_indices, mean_voltage,
                     alpha=0.12, color=COLOR_VOLTAGE)
    ax1.set_xlabel("Discharge Cycle Index", fontsize=12, fontweight="bold",
                   color="white", labelpad=10)
    ax1.set_ylabel("Voltage (V)", fontsize=12, fontweight="bold",
                   color=COLOR_VOLTAGE, labelpad=10)
    ax1.tick_params(axis="y", labelcolor=COLOR_VOLTAGE, colors=COLOR_VOLTAGE)
    ax1.tick_params(axis="x", colors="white")
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())

    # ---- Axis 2: Current (right) ----
    ax2 = ax1.twinx()
    ax2.patch.set_alpha(0.0)
    ax2.set_zorder(ax1.get_zorder() + 1)
    ln2 = ax2.plot(cycle_indices, mean_current, color=COLOR_CURRENT,
                   linewidth=2, linestyle="--", label="Mean |Current| (A)",
                   zorder=3)
    ax2.fill_between(cycle_indices, mean_current,
                     alpha=0.08, color=COLOR_CURRENT)
    ax2.set_ylabel("Current (A)", fontsize=12, fontweight="bold",
                   color=COLOR_CURRENT, labelpad=10)
    ax2.tick_params(axis="y", labelcolor=COLOR_CURRENT, colors=COLOR_CURRENT)
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    _set_axis_limits_from_series(ax2, mean_current, min_pad=0.02)

    # ---- Axis 3: Resistance Degradation Rate (far right) ----
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))  # offset outward
    ax3.patch.set_alpha(0.0)
    ax3.set_zorder(ax1.get_zorder() + 2)
    ln3 = ax3.plot(cycle_indices, dr0_smooth, color=COLOR_RESISTANCE,
                   linewidth=2.5, linestyle="-.", label="dR₀/dcycle (Ω/cycle)",
                   zorder=3)
    ax3.fill_between(cycle_indices, dr0_smooth,
                     alpha=0.10, color=COLOR_RESISTANCE)
    ax3.set_ylabel("dR₀/dcycle  (Ω / cycle)", fontsize=12, fontweight="bold",
                   color=COLOR_RESISTANCE, labelpad=15)
    ax3.tick_params(axis="y", labelcolor=COLOR_RESISTANCE, colors=COLOR_RESISTANCE)
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    _set_axis_limits_from_series(ax1, mean_voltage, min_pad=0.01)

    # ---- Spine colours ----
    for spine in ax1.spines.values():
        spine.set_color("#334155")
    ax2.spines["right"].set_color(COLOR_CURRENT)
    ax3.spines["right"].set_color(COLOR_RESISTANCE)
    ax1.spines["left"].set_color(COLOR_VOLTAGE)

    # ---- Grid on primary axis only ----
    ax1.grid(True, which="major", color="#334155", linewidth=0.6, alpha=0.6)
    ax1.grid(True, which="minor", color="#1E293B", linewidth=0.3, alpha=0.4)

    # ---- Combined legend ----
    lines = ln1 + ln2 + ln3
    labels = [l.get_label() for l in lines]
    legend = ax1.legend(lines, labels, loc="upper center",
                        bbox_to_anchor=(0.5, -0.10), ncol=3,
                        fontsize=11, frameon=True, fancybox=True,
                        shadow=True, facecolor="#1E293B", edgecolor="#475569",
                        labelcolor="white")

    # ---- Title ----
    title_text = (
        f"Voltage, Current & Resistance Degradation Rate  —  {cell_id}"
        + ("  [SMOKE]" if smoke else "")
    )
    ax1.set_title(title_text, fontsize=15, fontweight="bold", color="white",
                  pad=18)

    # ---- Annotations for key milestones ----
    # Mark the point of maximum resistance degradation rate
    if len(dr0_smooth) > 5:
        peak_idx = np.argmax(dr0_smooth)
        ax3.annotate(
            f"Peak dR₀ = {dr0_smooth[peak_idx]:.6f} Ω/cyc",
            xy=(cycle_indices[peak_idx], dr0_smooth[peak_idx]),
            xytext=(cycle_indices[peak_idx] + n_cycles * 0.08,
                    dr0_smooth[peak_idx] * 1.3),
            fontsize=9, color=COLOR_RESISTANCE, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLOR_RESISTANCE, lw=1.5),
        )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15, right=0.85)

    out_path = "combined_plot.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"\n  Plot saved -> {out_path}")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined plot: voltage, current & resistance degradation rate."
    )
    parser.add_argument("--smoke",  action="store_true",
                        help="Quick test with limited cycles")
    parser.add_argument("--cycles", type=int, default=10,
                        help="Max cycles in smoke mode")
    parser.add_argument("--cell",   type=str, default="B0018",
                        help="Cell ID to plot")
    parser.add_argument("--data",   type=str, default="./data/raw",
                        help="Path to raw .mat files")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Configuration file")
    args = parser.parse_args()

    plot_combined(
        cell_id     = args.cell,
        data_path   = args.data,
        config_path = args.config,
        smoke       = args.smoke,
        max_cycles  = args.cycles,
    )
