"""
augmentation/gp.py
------------------
Sparse Variational GP (SVGP) residual corrector.

Learns the systematic residual:  residual = V_measured - V_ECM

Inputs  : [SoC, I, T, cycle_index]   (4-dimensional)
Output  : posterior mean correction + variance

All hyperparameters are read from config.yaml — nothing is hardcoded.

Dependencies
------------
    pip install torch gpytorch pyyaml numpy
"""

import numpy as np
import yaml
import tempfile
import os

import torch
import gpytorch
from gpytorch.models        import ApproximateGP
from gpytorch.variational   import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods   import GaussianLikelihood
from gpytorch.mlls          import VariationalELBO
from gpytorch.kernels       import ScaleKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal


# ---------------------------------------------------------------------------
# Helper: load config
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# SVGP model definition
# ---------------------------------------------------------------------------

class _SVGPModel(ApproximateGP):
    """
    Sparse Variational GP with Matern-5/2 kernel.
    Inducing points live in the 4-D input space [SoC, I, T, cycle_idx].
    """

    def __init__(self, inducing_points: torch.Tensor):
        variational_dist     = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_dist,
            learn_inducing_locations=True,   # inducing points are optimised
        )
        super().__init__(variational_strategy)

        self.mean_module  = gpytorch.means.ZeroMean()
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=inducing_points.size(1))
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean  = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


# ---------------------------------------------------------------------------
# SVGPCorrector — public class
# ---------------------------------------------------------------------------

class SVGPCorrector:
    """
    Trains a Sparse Variational GP on voltage residuals and predicts
    corrections + uncertainty at new operating points.

    Methods
    -------
    fit(soc, I, T, cycle_idx, residuals)  — train on collected residuals
    predict(soc, I, T, cycle_idx)         — return (mean, variance)
    update(new_inputs, new_residuals)     — fine-tune on new data
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg = _load_config(config_path)
        gp  = cfg["gp"]

        self._num_inducing = int(gp["num_inducing"])
        self._lr           = float(gp["lr"])
        self._epochs       = int(gp["epochs"])
        self._input_dim    = 4          # [SoC, I, T, cycle_index]

        self._model      = None
        self._likelihood = None
        self._X_mean     = None        # normalisation stats
        self._X_std      = None
        self._y_mean     = None
        self._y_std      = None

        # Use GPU if available, else CPU
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def fit(
        self,
        soc:       np.ndarray,
        I:         np.ndarray,
        T:         np.ndarray,
        cycle_idx: np.ndarray,
        residuals: np.ndarray,
    ) -> None:
        """
        Train the SVGP on collected residual data.

        Parameters
        ----------
        soc       : State of Charge values in [0, 1],  shape (N,)
        I         : Current values (A),                shape (N,)
        T         : Temperature values (°C or K),      shape (N,)
        cycle_idx : Cycle index (integer),             shape (N,)
        residuals : V_measured - V_ECM  (V),           shape (N,)
        """
        X, y = self._prepare_data(soc, I, T, cycle_idx, residuals)
        self._build_model(X)
        self._train(X, y, n_epochs=self._epochs)

    def predict(
        self,
        soc:       float | np.ndarray,
        I:         float | np.ndarray,
        T:         float | np.ndarray,
        cycle_idx: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict voltage correction and uncertainty.

        Returns
        -------
        mean     : np.ndarray — posterior mean correction (V)
        variance : np.ndarray — posterior variance (V²)
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")

        soc       = np.atleast_1d(np.asarray(soc,       dtype=float))
        I         = np.atleast_1d(np.asarray(I,         dtype=float))
        T         = np.atleast_1d(np.asarray(T,         dtype=float))
        cycle_idx = np.atleast_1d(np.asarray(cycle_idx, dtype=float))

        X_raw = np.column_stack([soc, I, T, cycle_idx])
        X_norm = torch.tensor(
            (X_raw - self._X_mean) / (self._X_std + 1e-8),
            dtype=torch.float32,
        ).to(self._device)

        self._model.eval()
        self._likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._likelihood(self._model(X_norm))
            mean_norm = pred.mean.cpu().numpy()
            var_norm  = pred.variance.cpu().numpy()

        # Denormalise
        mean = mean_norm * (self._y_std + 1e-8) + self._y_mean
        var  = var_norm  * (self._y_std + 1e-8) ** 2

        return mean, var

    def update(
        self,
        new_soc:       np.ndarray,
        new_I:         np.ndarray,
        new_T:         np.ndarray,
        new_cycle_idx: np.ndarray,
        new_residuals: np.ndarray,
        fine_tune_epochs: int = 20,
    ) -> None:
        """
        Fine-tune on new residual observations (partial retraining).

        Parameters
        ----------
        new_* : new observations to learn from
        fine_tune_epochs : number of additional training epochs
        """
        if self._model is None:
            # No model yet — do a full fit
            self.fit(new_soc, new_I, new_T, new_cycle_idx, new_residuals)
            return

        X, y = self._prepare_data(
            new_soc, new_I, new_T, new_cycle_idx, new_residuals,
            recompute_norm=False,   # keep existing normalisation
        )
        self._train(X, y, n_epochs=fine_tune_epochs)

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _prepare_data(
        self,
        soc, I, T, cycle_idx, residuals,
        recompute_norm: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack inputs, normalise, convert to tensors."""
        X_raw = np.column_stack([
            np.asarray(soc,       dtype=float),
            np.asarray(I,         dtype=float),
            np.asarray(T,         dtype=float),
            np.asarray(cycle_idx, dtype=float),
        ])
        y_raw = np.asarray(residuals, dtype=float)

        if recompute_norm:
            self._X_mean = X_raw.mean(axis=0)
            self._X_std  = X_raw.std(axis=0) + 1e-8
            self._y_mean = y_raw.mean()
            self._y_std  = y_raw.std() + 1e-8

        X_norm = (X_raw - self._X_mean) / self._X_std
        y_norm = (y_raw - self._y_mean) / self._y_std

        X = torch.tensor(X_norm, dtype=torch.float32).to(self._device)
        y = torch.tensor(y_norm, dtype=torch.float32).to(self._device)
        return X, y

    def _build_model(self, X: torch.Tensor) -> None:
        """Initialise SVGP model and likelihood."""
        # Pick inducing points by k-means++ style subsampling
        n_ind = min(self._num_inducing, X.shape[0])
        idx   = torch.randperm(X.shape[0])[:n_ind]
        inducing_points = X[idx].clone().detach()

        self._model      = _SVGPModel(inducing_points).to(self._device)
        self._likelihood = GaussianLikelihood().to(self._device)

    def _train(self, X: torch.Tensor, y: torch.Tensor, n_epochs: int) -> None:
        """Run the ELBO training loop."""
        self._model.train()
        self._likelihood.train()

        optimizer = torch.optim.Adam(
            list(self._model.parameters()) + list(self._likelihood.parameters()),
            lr=self._lr,
        )
        mll = VariationalELBO(self._likelihood, self._model, num_data=y.size(0))

        prev_loss = float("inf")
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = self._model(X)
            loss   = -mll(output, y)
            loss.backward()
            optimizer.step()

            # Simple early stopping: stop if loss barely moves for 10 epochs
            if epoch > 10 and abs(prev_loss - loss.item()) < 1e-6:
                break
            prev_loss = loss.item()

            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"    [GP] epoch {epoch+1:4d}/{n_epochs}  ELBO loss = {loss.item():.6f}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 55)
    print("  augmentation/gp.py  —  smoke test")
    print("=" * 55)

    # ---- Inline config ----
    _inline_cfg = """
gp:
  num_inducing: 50
  lr: 0.05
  epochs: 100
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                     delete=False) as tf:
        tf.write(_inline_cfg)
        tmp_cfg = tf.name

    np.random.seed(42)
    torch.manual_seed(42)

    # ---- Synthetic residual data ----
    # True residual: a smooth function of SoC and I
    # residual(soc, I) = 0.005 * sin(4*pi*soc) - 0.002 * I
    N = 300
    soc_train       = np.random.uniform(0.1, 0.9, N)
    I_train         = np.random.uniform(-2.0, 2.0, N)
    T_train         = np.random.uniform(20.0, 40.0, N)
    cycle_train     = np.random.randint(0, 100, N).astype(float)

    true_residual = (
        0.005 * np.sin(4 * np.pi * soc_train)
        - 0.002 * I_train
    )
    noisy_residual = true_residual + np.random.normal(0, 0.001, N)

    # ---- Fit ----
    print(f"\nTraining SVGP on {N} residual points...")
    gp = SVGPCorrector(config_path=tmp_cfg)
    gp.fit(soc_train, I_train, T_train, cycle_train, noisy_residual)

    # ---- Predict at training points ----
    mean_train, var_train = gp.predict(soc_train, I_train, T_train, cycle_train)

    train_rmse = np.sqrt(np.mean((mean_train - noisy_residual) ** 2))
    print(f"\nPrediction at training points:")
    print(f"  RMSE        : {train_rmse:.6f} V")
    print(f"  Mean var    : {np.mean(var_train):.6f} V²")

    # ---- Predict at unseen points ----
    soc_test    = np.array([0.3, 0.5, 0.7, 0.5])
    I_test      = np.array([1.0, 0.0, -1.0, 1.5])
    T_test      = np.array([25., 25., 25., 30.])
    cyc_test    = np.array([50., 50., 50., 50.])

    mean_test, var_test = gp.predict(soc_test, I_test, T_test, cyc_test)

    print(f"\nPrediction at 4 unseen points:")
    for i in range(len(soc_test)):
        print(f"  SoC={soc_test[i]:.1f} I={I_test[i]:+.1f}A  →  "
              f"correction={mean_test[i]:+.5f} V   "
              f"std={np.sqrt(var_test[i]):.5f} V")

    # ---- Key check: variance near training data < variance far away ----
    # Near: soc=0.5, I=0.0 (well covered in training set)
    # Far:  soc=0.95, I=3.0 (outside training range)
    mean_near, var_near = gp.predict(
        np.array([0.5]), np.array([0.0]),
        np.array([25.0]), np.array([50.0])
    )
    mean_far, var_far = gp.predict(
        np.array([0.95]), np.array([3.0]),
        np.array([25.0]), np.array([50.0])
    )
    print(f"\nUncertainty check:")
    print(f"  Near training data  std = {np.sqrt(var_near[0]):.5f} V")
    print(f"  Far from training   std = {np.sqrt(var_far[0]):.5f} V")
    assert var_far[0] >= var_near[0], \
        "FAIL: variance should be higher far from training data"
    print("  ✓ Variance is correctly higher away from training data")

    # ---- update() test ----
    new_soc  = np.random.uniform(0.1, 0.9, 30)
    new_I    = np.random.uniform(-2.0, 2.0, 30)
    new_T    = np.full(30, 35.0)
    new_cyc  = np.full(30, 110.0)
    new_res  = 0.005 * np.sin(4 * np.pi * new_soc) - 0.002 * new_I + np.random.normal(0, 0.001, 30)

    gp.update(new_soc, new_I, new_T, new_cyc, new_res)
    print("\nupdate() with 30 new points: OK")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Panel 1: predicted vs true residual
    sort_idx = np.argsort(soc_train)
    axes[0].scatter(soc_train, noisy_residual,  s=6, alpha=0.3, label="observed")
    axes[0].scatter(soc_train, mean_train,       s=6, alpha=0.3, label="GP mean")
    axes[0].set_xlabel("SoC"); axes[0].set_ylabel("Residual (V)")
    axes[0].set_title("SVGP fit on training data"); axes[0].legend()

    # Panel 2: posterior std along SoC axis (I=0, T=25, cyc=50)
    soc_line  = np.linspace(0.0, 1.0, 100)
    I_line    = np.zeros(100)
    T_line    = np.full(100, 25.0)
    cyc_line  = np.full(100, 50.0)
    m_line, v_line = gp.predict(soc_line, I_line, T_line, cyc_line)
    std_line = np.sqrt(v_line)

    axes[1].plot(soc_line, m_line,              lw=2,   label="GP mean")
    axes[1].fill_between(soc_line,
                         m_line - 2*std_line,
                         m_line + 2*std_line,
                         alpha=0.25,            label="±2σ")
    axes[1].set_xlabel("SoC"); axes[1].set_ylabel("Correction (V)")
    axes[1].set_title("Posterior along SoC (I=0, T=25°C)"); axes[1].legend()

    plt.tight_layout()
    plt.savefig("smoke_test_gp.png", dpi=120)
    print("\nPlot saved → smoke_test_gp.png")

    os.unlink(tmp_cfg)
    print("\n✓ Smoke test passed.")