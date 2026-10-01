"""Surrogate wrappers with a common interface.

  fit(X, y)                   : (re)fit on the full data (hyper-parameter optimisation for GP)
  condition(X_new, y_new)     : add pseudo-observations WITHOUT re-optimising (Kriging Believer step)
  mean_std(Xc)                : marginal mean / std on candidates
  sample(Xc, n)               : posterior samples, shape (n, len(Xc)).  GP -> joint, TabPFN -> marginal
"""
import os
import numpy as np
from scipy.stats import norm
from simulator import N_SUB

QUANTILES = np.linspace(0.01, 0.99, 41)


def one_hot(X):
    X = np.asarray(X, float)
    cat = X[:, N_SUB].astype(int)
    oh = np.eye(3)[cat]
    return np.column_stack([X[:, :N_SUB], oh])


class GPSurrogate:
    """sklearn GP: Matern-5/2 ARD + noise, standardised y (torch-free baseline)."""

    def __init__(self, seed=0):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
        k = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=np.ones(N_SUB + 3), length_scale_bounds=(1e-2, 1e2), nu=2.5) \
            + WhiteKernel(1e-2, (1e-5, 1e0))
        self.gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=3, random_state=seed)
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        self.X, self.y = one_hot(X), np.asarray(y, float)
        self.gp.fit(self.X, self.y)
        return self

    def condition(self, X_new, y_new):
        from sklearn.gaussian_process import GaussianProcessRegressor
        self.X = np.vstack([self.X, one_hot(X_new)]); self.y = np.concatenate([self.y, y_new])
        fixed = GaussianProcessRegressor(kernel=self.gp.kernel_, optimizer=None, normalize_y=True)
        fixed.fit(self.X, self.y)
        self.gp = fixed
        return self

    def reset_to(self, X, y):
        """Re-condition on real data only, reusing the already-optimised kernel.
        Used to undo Kriging-Believer pseudo-observations before the calibration read."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        self.X, self.y = one_hot(X), np.asarray(y, float)
        fixed = GaussianProcessRegressor(kernel=self.gp.kernel_, optimizer=None, normalize_y=True)
        fixed.fit(self.X, self.y)
        self.gp = fixed
        return self

    def mean_std(self, Xc):
        m, s = self.gp.predict(one_hot(Xc), return_std=True)
        return m, s

    def interval(self, Xc, level=0.8):
        """Predictive interval for a NEW noisy observation (WhiteKernel is part of
        the kernel, so sklearn's predictive std already includes the noise term)."""
        m, s = self.mean_std(Xc)
        z = norm.ppf(0.5 + level / 2.0)
        return m - z * s, m + z * s

    def sample(self, Xc, n=1):
        return self.gp.sample_y(one_hot(Xc), n_samples=n, random_state=int(self.rng.integers(1 << 31))).T


class TabPFNSurrogate:
    """TabPFN v2 regressor (pip install tabpfn; weights auto-download on first use).
    Categorical column passed natively; posterior samples via inverse-CDF of predicted quantiles."""

    # The installed `tabpfn` package defaults to its newest generation; the brief
    # specifies TabPFN v2, so the checkpoint is pinned explicitly.  PBO_TABPFN_MODEL
    # switches generations for the Phase-D comparison.
    MODEL = os.environ.get("PBO_TABPFN_MODEL", "tabpfn-v2-regressor.ckpt")

    def __init__(self, seed=0, device="cpu", n_estimators=4, model_path=None):
        from tabpfn import TabPFNRegressor
        self.reg = TabPFNRegressor(device=device, n_estimators=n_estimators, random_state=seed,
                                   categorical_features_indices=[N_SUB],
                                   model_path=model_path or self.MODEL)
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        self.X, self.y = np.asarray(X, float), np.asarray(y, float)
        self.reg.fit(self.X, self.y)          # no training: just sets the context
        return self

    def condition(self, X_new, y_new):     # identical cost to fit -> KB is cheap
        return self.fit(np.vstack([self.X, X_new]), np.concatenate([self.y, y_new]))

    def reset_to(self, X, y):              # drop pseudo-observations from the context
        return self.fit(X, y)

    def _quantiles(self, Xc):
        q = self.reg.predict(np.asarray(Xc, float), output_type="quantiles", quantiles=list(QUANTILES))
        return np.asarray(q).T             # (n_cand, n_q)

    def mean_std(self, Xc):
        Q = self._quantiles(Xc)
        m = Q.mean(1); s = (Q[:, -1] - Q[:, 0]) / 4.65   # 1%-99% range ~ 4.65 sigma
        return m, s

    def interval(self, Xc, level=0.8):
        """Predictive interval straight from TabPFN's own quantile head."""
        lo, hi = 0.5 - level / 2.0, 0.5 + level / 2.0
        q = np.asarray(self.reg.predict(np.asarray(Xc, float), output_type="quantiles",
                                        quantiles=[lo, hi]))
        return q[0], q[1]

    def sample(self, Xc, n=1):
        Q = self._quantiles(Xc)
        u = self.rng.random((n, len(Q)))
        return np.stack([np.interp(u[:, i], QUANTILES, Q[i]) for i in range(len(Q))], axis=1)


def make_surrogate(name, seed, device="cpu"):
    if name == "tabpfn":
        return TabPFNSurrogate(seed=seed, device=device)
    if name == "gp_bo":                      # BoTorch SingleTaskGP (dim-scaled prior)
        from surrogate_botorch import make_botorch_surrogate
        return make_botorch_surrogate(seed=seed, device=device)
    return {"gp": GPSurrogate}[name](seed=seed)
