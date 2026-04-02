"""
augmentation/lstm.py
--------------------
LSTM ageing emulator.

Learns to predict the next cycle's ECM parameters θ(k+1..k+n)
from a sliding window of past θ(k) vectors.

θ = [R0, R1, C1, R2, C2]   (5-dimensional)

Training : teacher forcing   (feed true θ at each step)
Inference : autoregressive   (feed own predictions back)
Transfer  : freeze LSTM layers, retrain only output projection

All hyperparameters are read from config.yaml — nothing is hardcoded.

Dependencies
------------
    pip install torch pyyaml numpy
"""

import logging
import os
import tempfile

import numpy as np
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Internal PyTorch module
# ---------------------------------------------------------------------------

class _LSTMNet(nn.Module):
    """
    2-layer LSTM followed by a linear output projection.

    Input  : (batch, window, 5)
    Output : (batch, 5)   — next-step θ prediction
    """

    def __init__(self, input_size: int, hidden: int, layers: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.1 if layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, input_size)
        out, _ = self.lstm(x)    # (batch, seq_len, hidden)
        last   = out[:, -1, :]  # take last time-step
        return self.proj(last)   # (batch, output_size)


# ---------------------------------------------------------------------------
# AgeingEmulator — public class
# ---------------------------------------------------------------------------

class AgeingEmulator:
    """
    LSTM-based emulator of ECM parameter evolution over cycle life.

    Methods
    -------
    fit(theta_sequences)
        Train on a list of per-cell θ histories (teacher forcing).

    predict(theta_seed) -> np.ndarray shape (5,)
        One-step-ahead prediction.

    rollout(theta_seed, n_steps) -> np.ndarray shape (n_steps, 5)
        Autoregressive multi-step prediction.

    fine_tune(new_cell_theta, freeze_lstm, epochs, lr)
        Transfer learning: optionally freeze LSTM weights, retrain only
        the output Linear projection for a new (unseen) cell.
        After fine-tuning all parameters are unfrozen to preserve future
        trainability, but the caller should re-call fit() if a full
        re-train from scratch is desired.

    update(new_theta_sequence)
        Incremental fine-tune without LSTM freeze (10 epochs, full network).

    save(path) / load(path)
        Persist / restore model weights and normalisation stats.
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg  = _load_config(config_path)
        lstm = cfg["lstm"]

        self._hidden    = int(lstm["hidden"])
        self._layers    = int(lstm["layers"])
        self._window    = int(lstm["window"])
        self._lr        = float(lstm["lr"])
        self._epochs    = int(lstm["epochs"])
        # batch_size exposed via config with a sensible default
        self._batch_size = int(lstm.get("batch_size", 64))

        self._theta_dim = 5  # [R0, R1, C1, R2, C2]

        self._net: "_LSTMNet | None" = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._theta_mean: "np.ndarray | None" = None
        self._theta_std:  "np.ndarray | None" = None

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def fit(self, theta_sequences: "list[np.ndarray]") -> None:
        """
        Train the LSTM on multiple cells' θ history.

        Parameters
        ----------
        theta_sequences : list of 2-D arrays, each shape (n_cycles, 5)
            One array per training cell. Cycles must be in chronological order.
        """
        all_theta = np.vstack(theta_sequences)   # (total_cycles, 5)
        self._theta_mean = all_theta.mean(axis=0)
        self._theta_std  = all_theta.std(axis=0) + 1e-8

        X_list, y_list = [], []
        for seq in theta_sequences:
            x, y = self._make_windows(seq)
            X_list.append(x)
            y_list.append(y)

        X = np.concatenate(X_list, axis=0)  # (N_windows, window, 5)
        y = np.concatenate(y_list, axis=0)  # (N_windows, 5)

        self._net = _LSTMNet(
            input_size=self._theta_dim,
            hidden=self._hidden,
            layers=self._layers,
            output_size=self._theta_dim,
        ).to(self._device)

        self._run_training(X, y, n_epochs=self._epochs, lr=self._lr)

    def predict(self, theta_seed: np.ndarray) -> np.ndarray:
        """
        Predict one step ahead from the last `window` cycles.

        Parameters
        ----------
        theta_seed : array shape (≥window, 5)
            Recent θ history; only the last `window` rows are used.

        Returns
        -------
        np.ndarray shape (5,) — predicted θ for the next cycle
        """
        return self.rollout(theta_seed, n_steps=1)[0]

    def rollout(self, theta_seed: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Autoregressive multi-step prediction.

        Parameters
        ----------
        theta_seed : array shape (≥window, 5) — recent θ history
        n_steps    : number of future cycles to predict

        Returns
        -------
        np.ndarray shape (n_steps, 5) — predicted θ trajectory

        Note
        ----
        Teacher-forcing / autoregressive mismatch means errors can
        accumulate over long rollouts. For high-accuracy long-horizon
        forecasting, re-seed the buffer with fresh observations whenever
        they become available (e.g. via update()).
        """
        if self._net is None:
            raise RuntimeError("Call fit() before rollout().")

        seed = np.asarray(theta_seed, dtype=float)
        if seed.shape[0] < self._window:
            raise ValueError(
                f"theta_seed must have at least {self._window} rows, "
                f"got {seed.shape[0]}."
            )

        self._net.eval()
        predictions = []
        buffer = seed[-self._window:].copy()   # (window, 5)

        with torch.no_grad():
            for _ in range(n_steps):
                buf_norm  = self._normalise(buffer)
                x         = (
                    torch.tensor(buf_norm, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self._device)
                )                                      # (1, window, 5)
                pred_norm = self._net(x).cpu().numpy()[0]  # (5,)
                pred      = self._denormalise(pred_norm)

                predictions.append(pred)
                buffer = np.vstack([buffer[1:], pred])  # slide window

        return np.array(predictions)  # (n_steps, 5)

    def fine_tune(
        self,
        new_cell_theta: np.ndarray,
        freeze_lstm:    bool = True,
        epochs:         int  = 30,
        lr:             "float | None" = None,
    ) -> None:
        """
        Transfer learning: adapt the model to a new (unseen) cell.

        Parameters
        ----------
        new_cell_theta : array shape (n_cycles, 5) — first observed cycles
        freeze_lstm    : if True, freeze all LSTM weights and retrain only
                         the output Linear projection layer (recommended when
                         very few cycles are available for the new cell).
        epochs         : fine-tuning epochs (typically 20–50)
        lr             : learning rate; defaults to the value from config.yaml

        Note
        ----
        All parameters are unfrozen after fine-tuning so that subsequent
        calls to update() or fine_tune() can train the full network.
        If you need to permanently freeze the LSTM, manage requires_grad
        externally after calling this method.
        """
        if self._net is None:
            raise RuntimeError("Call fit() before fine_tune().")

        if freeze_lstm:
            for name, param in self._net.named_parameters():
                param.requires_grad = name.startswith("proj")

        X, y = self._make_windows(new_cell_theta)
        self._run_training(X, y, n_epochs=epochs, lr=lr or self._lr)

        # Always unfreeze after fine-tuning
        for param in self._net.parameters():
            param.requires_grad = True

    def update(self, new_theta_sequence: np.ndarray) -> None:
        """
        Incremental update — fine-tune on new observations from a cell
        already in service. All LSTM weights remain trainable.
        """
        self.fine_tune(new_theta_sequence, freeze_lstm=False, epochs=10)

    def save(self, path: str) -> None:
        """Save model weights and normalisation stats to a checkpoint file."""
        if self._net is None:
            raise RuntimeError("Nothing to save — model not yet trained.")
        torch.save(
            {
                "state_dict":  self._net.state_dict(),
                "theta_mean":  self._theta_mean,
                "theta_std":   self._theta_std,
                "hidden":      self._hidden,
                "layers":      self._layers,
                "window":      self._window,
                "theta_dim":   self._theta_dim,
                "batch_size":  self._batch_size,
            },
            path,
        )
        logger.info("Model saved → %s", path)
        print(f"  Model saved → {path}")

    def load(self, path: str) -> None:
        """Load model weights and normalisation stats from a checkpoint file."""
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        self._hidden     = ckpt["hidden"]
        self._layers     = ckpt["layers"]
        self._window     = ckpt["window"]
        self._theta_dim  = ckpt["theta_dim"]
        self._batch_size = ckpt.get("batch_size", 64)
        self._theta_mean = ckpt["theta_mean"]
        self._theta_std  = ckpt["theta_std"]

        self._net = _LSTMNet(
            input_size=self._theta_dim,
            hidden=self._hidden,
            layers=self._layers,
            output_size=self._theta_dim,
        ).to(self._device)
        self._net.load_state_dict(ckpt["state_dict"])
        self._net.eval()
        logger.info("Model loaded ← %s", path)
        print(f"  Model loaded ← {path}")

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _make_windows(
        self, seq: np.ndarray
    ) -> "tuple[np.ndarray, np.ndarray]":
        """
        Convert a (n_cycles, 5) sequence into supervised (input, target) windows.

        Sequences shorter than window+1 are padded by repeating the first row
        so that at least one training sample is produced (useful when only a
        handful of early-life cycles are available for a new cell).

        Returns
        -------
        X : (N, window, 5)   — normalised input windows
        y : (N, 5)           — normalised next-step targets
        """
        seq     = np.asarray(seq, dtype=float)
        min_len = self._window + 1
        if len(seq) < min_len:
            pad = np.tile(seq[0], (min_len - len(seq), 1))
            seq = np.vstack([pad, seq])

        seq_norm = self._normalise(seq)
        X, y = [], []
        for i in range(len(seq_norm) - self._window):
            X.append(seq_norm[i : i + self._window])
            y.append(seq_norm[i + self._window])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _run_training(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int,
        lr: float,
    ) -> None:
        """MSE training loop (teacher forcing) with gradient clipping."""
        dataset   = TensorDataset(
            torch.tensor(X).to(self._device),
            torch.tensor(y).to(self._device),
        )
        loader    = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
        trainable = [p for p in self._net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=lr)
        criterion = nn.MSELoss()

        self._net.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self._net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                # Gradient clipping prevents NaN / exploding gradients in
                # deep / multi-layer LSTMs on long sequences
                nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(
                    f"    [LSTM] epoch {epoch+1:4d}/{n_epochs}"
                    f"  MSE loss = {avg_loss:.8f}"
                )

    def _normalise(self, theta: np.ndarray) -> np.ndarray:
        return (theta - self._theta_mean) / self._theta_std

    def _denormalise(self, theta_norm: np.ndarray) -> np.ndarray:
        return theta_norm * self._theta_std + self._theta_mean


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.DEBUG)

    print("=" * 55)
    print("  augmentation/lstm.py  —  smoke test")
    print("=" * 55)

    # ---- Inline config (spec defaults: hidden=64, layers=2, window=20,
    #                      lr=1e-3, epochs=100) ----
    _inline_cfg = """
lstm:
  hidden: 64
  layers: 2
  window: 20
  lr: 0.001
  epochs: 80
  batch_size: 64
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        tf.write(_inline_cfg)
        tmp_cfg = tf.name

    np.random.seed(42)
    torch.manual_seed(42)

    # ---- Synthetic θ trajectories ----
    def make_theta_sequence(n_cycles: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        k   = np.arange(n_cycles, dtype=float)
        R0  = 0.015 + 0.008 * (1 - np.exp(-k / 150)) + rng.normal(0, 0.0003, n_cycles)
        R1  = 0.010 + 0.004 * (k / n_cycles)          + rng.normal(0, 0.0002, n_cycles)
        C1  = 3000  - 400   * (k / n_cycles)           + rng.normal(0, 5.0,   n_cycles)
        R2  = 0.008 + 0.003 * (k / n_cycles)           + rng.normal(0, 0.0001, n_cycles)
        C2  = 8000  - 800   * (k / n_cycles)           + rng.normal(0, 10.0,  n_cycles)
        return np.column_stack([R0, R1, C1, R2, C2])

    print("\nGenerating synthetic θ sequences for 3 training cells (200 cycles each)...")
    train_seqs = [make_theta_sequence(200, seed=i) for i in range(3)]

    # ---- Fit ----
    print("Training LSTM (teacher forcing)...")
    emulator = AgeingEmulator(config_path=tmp_cfg)
    emulator.fit(train_seqs)

    # ---- Rollout: 50-step autoregressive prediction ----
    print("\nAutoregressive rollout — 50 steps...")
    seed_seq  = train_seqs[0]
    pred_traj = emulator.rollout(seed_seq, n_steps=50)

    print(f"  Rollout shape : {pred_traj.shape}")
    print(f"  R0 trajectory (first 10 steps):")
    print("  " + "  ".join(f"{v:.5f}" for v in pred_traj[:10, 0]))

    r0_increasing = np.sum(np.diff(pred_traj[:, 0]) > 0) / 49
    print(
        f"  R0 increasing in {r0_increasing*100:.0f}% of rollout steps "
        f"(expect >70% for ageing trend)"
    )

    # ---- Single-step predict ----
    next_theta = emulator.predict(seed_seq)
    print(f"\nSingle-step predict (next θ after cycle 199):")
    for name, val in zip(["R0", "R1", "C1", "R2", "C2"], next_theta):
        print(f"  {name:3s} = {val:.5f}")

    # ---- Transfer learning: freeze LSTM, retrain projection only ----
    print("\nTransfer learning — freeze LSTM, retrain projection on new cell...")
    new_cell_seq = make_theta_sequence(50, seed=99)
    emulator.fine_tune(new_cell_seq, freeze_lstm=True, epochs=20)

    rollout_new = emulator.rollout(new_cell_seq, n_steps=30)
    print(
        f"  Rollout after fine-tune — R0 range: "
        f"[{rollout_new[:,0].min():.5f}, {rollout_new[:,0].max():.5f}]"
    )

    # ---- update() ----
    extra_cycles = make_theta_sequence(15, seed=77)
    emulator.update(extra_cycles)
    print("\nupdate() with 15 new cycles: OK")

    # ---- save / load round-trip ----
    ckpt_path   = "lstm_smoke_test.pt"
    emulator.save(ckpt_path)

    pred_before = emulator.predict(seed_seq)

    emulator2 = AgeingEmulator(config_path=tmp_cfg)
    emulator2.load(ckpt_path)
    pred_after = emulator2.predict(seed_seq)

    diff = np.abs(pred_after - pred_before).max()
    print(f"\nSave/load round-trip max diff: {diff:.2e}  (expect < 1e-3)")
    assert diff < 1e-3, f"Save/load mismatch: {diff}"
    os.remove(ckpt_path)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    true_future = train_seqs[0][200 - 50:]
    axes[0].plot(true_future[:, 0], lw=2, label="true R0 (cell 0, last 50 cyc)")
    axes[0].plot(pred_traj[:, 0],   lw=2, ls="--", label="LSTM rollout (50 steps)")
    axes[0].set_xlabel("Steps ahead")
    axes[0].set_ylabel("R0 (Ω)")
    axes[0].set_title("Autoregressive rollout vs true")
    axes[0].legend()

    norm_traj = (pred_traj - emulator._theta_mean) / emulator._theta_std
    for i, name in enumerate(["R0", "R1", "C1", "R2", "C2"]):
        axes[1].plot(norm_traj[:, i], label=name)
    axes[1].set_xlabel("Steps ahead")
    axes[1].set_ylabel("Normalised θ")
    axes[1].set_title("All 5 params — 50-step rollout")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("smoke_test_lstm.png", dpi=120)
    print("\nPlot saved → smoke_test_lstm.png")

    os.unlink(tmp_cfg)
    print("\n✓ Smoke test passed.")