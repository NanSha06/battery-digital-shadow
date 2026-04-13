"""
data/remediation_pipeline.py
----------------------------
Production-ready data remediation pipeline for the NASA battery dataset.
Provides a clean API callable from the Streamlit dashboard and from scripts.

Public API
----------
run_remediation(raw_dir, cell_ids, out_csv, out_json) -> RemediationResult
load_remediation_result(csv_path) -> RemediationResult | None
"""
from __future__ import annotations

import base64
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io
import scipy.stats
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# Constants & domain bounds (lithium-ion physics limits)
# ---------------------------------------------------------------------------

EOL_AH     = 1.4
NOMINAL_AH = 2.0
IQR_FACTOR = 1.5
Z_THRESH   = 3.0
SYNTH_FRACTION = 0.15   # 15 % synthetic augmentation

DOMAIN_RULES: list[tuple[str, str, Any]] = [
    ("min_voltage_v",     "Voltage below 2.0 V (physics minimum)",    lambda v: v < 2.0),
    ("max_voltage_v",     "Voltage above 4.35 V (physics maximum)",   lambda v: v > 4.35),
    ("mean_abs_current_a","Current ~ 0 (sensor stuck)",               lambda v: v < 0.001),
    ("capacity_ah",       "Capacity below 1.0 Ah (below EOL)",        lambda v: v < 1.0),
    ("capacity_ah",       "Capacity above 3.5 Ah (unphysical)",       lambda v: v > 3.5),
    ("mean_temperature_c","Temperature below -20 C",                  lambda v: v < -20.0),
    ("mean_temperature_c","Temperature above 80 C",                   lambda v: v > 80.0),
    ("duration_s",        "Cycle duration < 100 s (too short)",       lambda v: v < 100.0),
    ("soh",               "SOH > 1.10 (> 110 %, unphysical)",        lambda v: v > 1.10),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AnomalyEntry:
    row_idx:  int
    col:      str
    method:   str
    value:    float
    reason:   str


@dataclass
class RemediationResult:
    """Complete result returned by run_remediation()."""
    df_original:     pd.DataFrame            # raw extracted features (before any fixing)
    df_clean:        pd.DataFrame            # fixed dataset (no synthetics)
    df_final:        pd.DataFrame            # clean + synthetic rows
    anomaly_log:     list[AnomalyEntry]
    fix_log:         list[str]
    stats_original:  dict[str, dict]         # per-column stats before fixing
    stats_final:     dict[str, dict]         # per-column stats after fixing
    pearson_corr:    pd.DataFrame
    spearman_corr:   pd.DataFrame
    n_original:      int
    n_synthetic:     int
    n_total:         int
    csv_path:        Path | None = None
    json_path:       Path | None = None
    b64_path:        Path | None = None
    report_path:     Path | None = None

    # Convenience properties ------------------------------------------------
    @property
    def anomaly_df(self) -> pd.DataFrame:
        """Anomaly log as a readable DataFrame."""
        return pd.DataFrame([
            {"Row": a.row_idx, "Column": a.col, "Method": a.method,
             "Value": round(a.value, 5), "Reason": a.reason}
            for a in self.anomaly_log
        ])

    @property
    def cell_counts(self) -> dict[str, int]:
        return self.df_final.groupby("cell_id").size().to_dict()

    @property
    def cell_counts_original(self) -> dict[str, int]:
        return self.df_original.groupby("cell_id").size().to_dict()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_cell(raw_dir: Path, cell_id: str) -> list[dict]:
    """Extract per-cycle scalar features from a NASA .mat file."""
    mat_path = raw_dir / f"{cell_id}.mat"
    raw = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    try:
        cycles_raw = raw[cell_id]["cycle"]
    except KeyError:
        key = [k for k in raw if not k.startswith("_")][-1]
        cycles_raw = raw[key]["cycle"]

    records: list[dict] = []
    cycle_idx = 0
    for cycle in cycles_raw:
        if str(cycle.get("type", "")).strip().lower() != "discharge":
            continue
        data = cycle.get("data", {})
        voltage  = np.asarray(data.get("Voltage_measured",    []), dtype=float).ravel()
        current  = np.asarray(data.get("Current_measured",    []), dtype=float).ravel()
        temp     = np.asarray(data.get("Temperature_measured",[]), dtype=float).ravel()
        time_arr = np.asarray(data.get("Time",                []), dtype=float).ravel()
        cap_raw  = np.asarray(data.get("Capacity",       [np.nan]), dtype=float).ravel()

        if voltage.size < 10:
            cycle_idx += 1
            continue

        n = min(voltage.size, current.size, temp.size)
        voltage, current, temp = voltage[:n], current[:n], temp[:n]
        time_arr = time_arr[:n] if time_arr.size >= n else np.arange(n, dtype=float)
        dt = float(np.median(np.diff(time_arr))) if time_arr.size > 1 else 1.0
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0

        capacity_ah  = float(np.trapz(np.abs(current), dx=dt) / 3600.0)
        cap_endpoint = float(cap_raw[-1]) if cap_raw.size > 0 and np.isfinite(cap_raw[-1]) else np.nan

        records.append({
            "cell_id":            cell_id,
            "cycle_number":       cycle_idx,
            "capacity_ah":        capacity_ah,
            "cap_endpoint_ah":    cap_endpoint,
            "mean_voltage_v":     float(np.mean(voltage)),
            "min_voltage_v":      float(np.min(voltage)),
            "max_voltage_v":      float(np.max(voltage)),
            "mean_abs_current_a": float(np.mean(np.abs(current))),
            "mean_temperature_c": float(np.median(temp)),
            "duration_s":         float(time_arr[-1] - time_arr[0]),
            "n_samples":          n,
            "dt_s":               dt,
        })
        cycle_idx += 1
    return records


def _col_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    return {
        "count":    int(len(arr)),
        "mean":     float(np.mean(arr)),
        "median":   float(np.median(arr)),
        "std":      float(np.std(arr)),
        "min":      float(np.min(arr)),
        "max":      float(np.max(arr)),
        "q1":       q1,
        "q3":       q3,
        "iqr":      q3 - q1,
        "skewness": float(scipy.stats.skew(arr)),
        "kurtosis": float(scipy.stats.kurtosis(arr)),
    }


NUMERIC_COLS = [
    "capacity_ah", "cap_endpoint_ah", "mean_voltage_v", "min_voltage_v",
    "max_voltage_v", "mean_abs_current_a", "mean_temperature_c",
    "duration_s", "n_samples", "dt_s",
]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_remediation(
    raw_dir:  str | Path = "data/raw",
    cell_ids: list[str] | None = None,
    out_csv:  str | Path | None = "data/remediated_dataset.csv",
    out_json: str | Path | None = "data/remediated_dataset.json",
    nominal_capacity_ah: float = NOMINAL_AH,
    eol_capacity_ah:     float = EOL_AH,
    synth_fraction:      float = SYNTH_FRACTION,
    random_seed:         int   = 2024,
) -> RemediationResult:
    """
    Execute all 5 remediation steps and return a RemediationResult.

    Parameters
    ----------
    raw_dir  : folder containing B*.mat files
    cell_ids : list of cell IDs to process (default: all B*.mat in raw_dir)
    out_csv  : path for output CSV (None = no file written)
    out_json : path for output JSON (None = no file written)
    """
    raw_dir = Path(raw_dir)
    if cell_ids is None:
        cell_ids = sorted(p.stem.upper() for p in raw_dir.glob("B*.mat"))

    # ── STEP 1: Ingest ──────────────────────────────────────────────────────
    all_records: list[dict] = []
    for cid in cell_ids:
        if not (raw_dir / f"{cid}.mat").exists():
            continue
        all_records.extend(_load_cell(raw_dir, cid))

    df = pd.DataFrame(all_records)
    df["soh"] = (df["capacity_ah"] / nominal_capacity_ah).clip(0.0, 1.5)

    stats_original = {col: _col_stats(df[col].values) for col in NUMERIC_COLS + ["soh"] if col in df.columns}
    pcols = [c for c in ["capacity_ah","mean_voltage_v","mean_temperature_c","duration_s","soh"] if c in df.columns]
    pearson_corr  = df[pcols].corr(method="pearson")
    spearman_corr = df[pcols].corr(method="spearman")

    # ── STEP 2: Detect anomalies ────────────────────────────────────────────
    anomaly_log: list[AnomalyEntry] = []

    # Z-score
    for col in NUMERIC_COLS:
        vals = df[col].values.astype(float)
        finite = np.isfinite(vals)
        if finite.sum() < 3:
            continue
        mu, sigma = np.nanmean(vals), np.nanstd(vals)
        if sigma < 1e-10:
            continue
        z = np.abs((vals - mu) / sigma)
        for idx in np.where((z > Z_THRESH) & finite)[0]:
            anomaly_log.append(AnomalyEntry(int(idx), col, "Z-score", float(vals[idx]),
                                            f"Z={z[idx]:.2f} > {Z_THRESH}"))

    # IQR
    for col in NUMERIC_COLS:
        s = stats_original.get(col, {})
        if not s:
            continue
        vals = df[col].values.astype(float)
        lo = s["q1"] - IQR_FACTOR * s["iqr"]
        hi = s["q3"] + IQR_FACTOR * s["iqr"]
        for idx in np.where((vals < lo) | (vals > hi))[0]:
            if not np.isfinite(vals[idx]):
                continue
            exists = any(a.row_idx == int(idx) and a.col == col for a in anomaly_log)
            if not exists:
                anomaly_log.append(AnomalyEntry(int(idx), col, "IQR", float(vals[idx]),
                                                f"Outside [{lo:.4f}, {hi:.4f}]"))

    # Isolation Forest
    if len(df) > 50:
        iso_cols = ["capacity_ah","mean_voltage_v","mean_temperature_c","duration_s"]
        X = df[iso_cols].fillna(df[iso_cols].median()).values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iso_labels = IsolationForest(n_estimators=200, contamination=0.05,
                                         random_state=42).fit_predict(X)
        for idx in np.where(iso_labels == -1)[0]:
            anomaly_log.append(AnomalyEntry(int(idx), "multivariate", "IsolationForest",
                                            float(df["capacity_ah"].iloc[idx]),
                                            "Flagged as multivariate outlier (contamination=5%)"))

    # Domain rules
    for col, reason, rule_fn in DOMAIN_RULES:
        if col not in df.columns:
            continue
        vals = df[col].values.astype(float)
        for idx, v in enumerate(vals):
            if np.isfinite(v) and rule_fn(v):
                anomaly_log.append(AnomalyEntry(int(idx), col, "DomainRule", float(v), reason))

    # ── STEP 3: Fix ─────────────────────────────────────────────────────────
    df_clean = df.copy()
    fix_log: list[str] = []
    total_rows = len(df_clean)
    DROP_THRESHOLD = 0.02

    hard_rows = {a.row_idx for a in anomaly_log if a.method == "DomainRule"}
    if hard_rows and len(hard_rows) / total_rows <= DROP_THRESHOLD:
        df_clean = df_clean.drop(index=list(hard_rows)).reset_index(drop=True)
        fix_log.append(f"Dropped {len(hard_rows)} rows with hard domain violations "
                       f"({len(hard_rows)/total_rows*100:.1f}% < 2% threshold).")
    else:
        if hard_rows:
            fix_log.append(f"Domain violations ({len(hard_rows)} rows) exceed 2% threshold "
                           "— capping at physics limits instead of deletion.")
        df_clean["min_voltage_v"]      = df_clean["min_voltage_v"].clip(lower=2.0)
        df_clean["max_voltage_v"]      = df_clean["max_voltage_v"].clip(upper=4.35)
        df_clean["capacity_ah"]        = df_clean["capacity_ah"].clip(lower=1.0, upper=3.5)
        df_clean["mean_temperature_c"] = df_clean["mean_temperature_c"].clip(lower=-20.0, upper=80.0)

    for col in NUMERIC_COLS:
        s = stats_original.get(col, {})
        if not s:
            continue
        lo = s["q1"] - IQR_FACTOR * s["iqr"]
        hi = s["q3"] + IQR_FACTOR * s["iqr"]
        before = df_clean[col].copy()
        df_clean[col] = df_clean[col].clip(lower=lo, upper=hi)
        changed = int((before != df_clean[col]).sum())
        if changed > 0:
            fix_log.append(f"Capped {changed} values in '{col}' to IQR fence [{lo:.4f}, {hi:.4f}].")

    nan_mask = df_clean["cap_endpoint_ah"].isna()
    if nan_mask.any():
        df_clean.loc[nan_mask, "cap_endpoint_ah"] = df_clean.loc[nan_mask, "capacity_ah"]
        fix_log.append(f"Imputed {nan_mask.sum()} NaN cap_endpoint_ah with capacity_ah.")

    df_clean["soh"] = (df_clean["capacity_ah"] / nominal_capacity_ah).clip(0.0, 1.1)

    # ── STEP 4: Synthesise ───────────────────────────────────────────────────
    rng = np.random.default_rng(seed=random_seed)
    synth_cols = [
        "capacity_ah","mean_voltage_v","min_voltage_v","max_voltage_v",
        "mean_abs_current_a","mean_temperature_c","duration_s",
    ]
    synth_rows: list[dict] = []

    for cell in cell_ids:
        cell_df = df_clean[df_clean["cell_id"] == cell]
        if len(cell_df) < 5:
            continue
        n_synth = max(1, int(len(cell_df) * synth_fraction))
        X   = cell_df[synth_cols].values.astype(float)
        mu  = X.mean(axis=0)
        cov = np.cov(X.T) + np.eye(X.shape[1]) * 1e-6
        samples     = rng.multivariate_normal(mu, cov, size=n_synth)
        col_mins    = X.min(axis=0) * 0.98
        col_maxs    = X.max(axis=0) * 1.02
        samples     = np.clip(samples, col_mins, col_maxs)
        cap_idx     = synth_cols.index("capacity_ah")
        min_v_idx   = synth_cols.index("min_voltage_v")
        max_v_idx   = synth_cols.index("max_voltage_v")
        temp_idx    = synth_cols.index("mean_temperature_c")
        samples[:, cap_idx]   = np.clip(samples[:, cap_idx],   eol_capacity_ah, nominal_capacity_ah * 1.05)
        samples[:, min_v_idx] = np.clip(samples[:, min_v_idx], 2.0, 3.6)
        samples[:, max_v_idx] = np.clip(samples[:, max_v_idx], 3.6, 4.35)
        samples[:, temp_idx]  = np.clip(samples[:, temp_idx],  5.0, 55.0)

        last_cycle = int(cell_df["cycle_number"].max())
        for i, row in enumerate(samples):
            d = {c: float(row[j]) for j, c in enumerate(synth_cols)}
            d.update({
                "cell_id":          cell,
                "cycle_number":     last_cycle + i + 1,
                "cap_endpoint_ah":  d["capacity_ah"],
                "soh":              float(np.clip(d["capacity_ah"] / nominal_capacity_ah, 0.0, 1.1)),
                "n_samples":        int(np.clip(d["duration_s"], 100, 5000)),
                "dt_s":             1.0,
                "is_synthetic":     True,
            })
            synth_rows.append(d)

    df_clean["is_synthetic"] = False
    df_synth = pd.DataFrame(synth_rows)
    df_final = pd.concat([df_clean, df_synth], ignore_index=True)
    df_final = df_final.sort_values(["cell_id","cycle_number"]).reset_index(drop=True)

    # Monotonic capacity smoothing per cell
    for cell in cell_ids:
        mask = df_final["cell_id"] == cell
        cap  = df_final.loc[mask, "capacity_ah"].values.astype(float)
        for i in range(1, len(cap)):
            if cap[i] > cap[i-1] * 1.01:
                cap[i] = cap[i-1] * 1.005
        df_final.loc[mask, "capacity_ah"] = cap
        df_final.loc[mask, "soh"] = (cap / nominal_capacity_ah).clip(0.0, 1.1)

    stats_final = {col: _col_stats(df_final[col].values)
                   for col in NUMERIC_COLS + ["soh"] if col in df_final.columns}

    # ── STEP 5: Write files ──────────────────────────────────────────────────
    csv_path = b64_path = json_path = report_path = None

    if out_csv is not None:
        csv_path = Path(out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(csv_path, index=False, float_format="%.6f")

        b64_str  = base64.b64encode(csv_path.read_bytes()).decode("ascii")
        b64_path = csv_path.with_suffix(".b64.txt")
        b64_path.write_text(b64_str, encoding="ascii")

    if out_json is not None:
        json_path = Path(out_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as fj:
            json.dump(df_final.to_dict(orient="records"), fj, indent=2, default=str)

    if csv_path is not None:
        report_path = csv_path.parent / "remediation_report.md"
        _write_report(report_path, df, df_final, anomaly_log, fix_log, stats_original, stats_final,
                      pearson_corr, spearman_corr, cell_ids)

    return RemediationResult(
        df_original=df,
        df_clean=df_clean,
        df_final=df_final,
        anomaly_log=anomaly_log,
        fix_log=fix_log,
        stats_original=stats_original,
        stats_final=stats_final,
        pearson_corr=pearson_corr,
        spearman_corr=spearman_corr,
        n_original=len(df_clean),
        n_synthetic=len(df_synth),
        n_total=len(df_final),
        csv_path=csv_path,
        json_path=json_path,
        b64_path=b64_path,
        report_path=report_path,
    )


def load_remediation_result(csv_path: str | Path) -> pd.DataFrame | None:
    """Load a previously generated remediated CSV.  Returns None if not found."""
    p = Path(csv_path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Internal: markdown report writer
# ---------------------------------------------------------------------------

def _write_report(
    path: Path,
    df_orig: pd.DataFrame,
    df_final: pd.DataFrame,
    anomaly_log: list[AnomalyEntry],
    fix_log: list[str],
    stats_orig: dict,
    stats_final: dict,
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    cell_ids: list[str],
) -> None:
    lines: list[str] = [
        "# Battery Dataset Remediation Report\n",
        f"*Cells processed: {', '.join(cell_ids)} | "
        f"Original rows: {len(df_orig)} | Final rows: {len(df_final)}*\n",
        "\n## 1. Summary of All Calculated Parameters (Original)\n",
        "| Column | Count | Mean | Median | Std | Min | Max | Skew | Kurt |",
        "|--------|-------|------|--------|-----|-----|-----|------|------|",
    ]
    for col, s in stats_orig.items():
        if not s:
            continue
        lines.append(f"| {col} | {s['count']} | {s['mean']:.4f} | {s['median']:.4f} | "
                     f"{s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} | "
                     f"{s['skewness']:.3f} | {s['kurtosis']:.3f} |")

    lines += [
        "\n\n## 2. Anomalies Detected\n",
        f"**Total flags: {len(anomaly_log)}**\n",
        "| Row | Column | Method | Value | Reason |",
        "|-----|--------|--------|-------|--------|",
    ]
    for a in anomaly_log[:120]:
        lines.append(f"| {a.row_idx} | {a.col} | {a.method} | {a.value:.5f} | {a.reason} |")
    if len(anomaly_log) > 120:
        lines.append(f"| ... | ... | ... | ... | *{len(anomaly_log)-120} more rows omitted* |")

    lines += ["\n\n## 3. Fixes Applied\n"]
    for f in fix_log:
        lines.append(f"- {f}")

    lines += [
        "\n\n## 4. New Dataset Specifications\n",
        "| Column | Mean (new) | Std (new) | Skew (new) | Kurt (new) |",
        "|--------|-----------|-----------|-----------|-----------|",
    ]
    for col, s in stats_final.items():
        if not s:
            continue
        lines.append(f"| {col} | {s['mean']:.4f} | {s['std']:.4f} | {s['skewness']:.3f} | {s['kurtosis']:.3f} |")

    path.write_text("\n".join(lines), encoding="utf-8")
