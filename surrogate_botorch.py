"""BoTorch `SingleTaskGP` behind the same surrogate API as `GPSurrogate`.

Why this exists (paper/追加実験設計.md §2.1): the main comparison needs a GP that
is strong in high dimensions, and the strongest cheap candidate is the
dimension-scaled LogNormal lengthscale prior of [Hvarfner et al., ICML 2024],
l_i ~ LN(sqrt(2) + log(D)/2, sqrt(3)).  BoTorch has used exactly that prior as the
`SingleTaskGP` default since 0.12, so the faithful implementation is to wrap
BoTorch rather than to re-derive it.

Trying the same thing in sklearn does not work and was abandoned: sklearn
maximises the marginal likelihood with no prior term, so a prior can only be
imitated by moving the lengthscale initialisation and bounds.  Measured at d=40,
that collapses to lengthscale ~0.16 with noise 0.74 — an "explain everything as
noise" fit whose log marginal likelihood (-136.2) is *worse* than the plain GP's
(-111.7), and identical across scenarios.  The prior has to enter the objective,
which is what BoTorch's MAP fit does.

Exposing the same `fit / condition / mean_std / sample / reset_to / interval`
API means `strat_kb_ei`, `strat_lp_ei` and `strat_ts` run against it unchanged,
so a paired contrast against `tabpfn_*` differs in the surrogate alone.
"""
import numpy as np

from simulator import N_SUB
from surrogates import one_hot


class BoTorchGPSurrogate:
    """SingleTaskGP (dimension-scaled LogNormal lengthscale prior, MAP fit).

    `fit` optimises the hyperparameters; `condition` and `reset_to` re-use them
    and only change the conditioning set, mirroring `GPSurrogate` so that
    Kriging-Believer costs the same kind of work in both arms.
    """

    def __init__(self, seed=0, device="cpu"):
        import torch
        self.torch = torch
        self.seed = int(seed)
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.dt = torch.double
        self.rng = np.random.default_rng(seed)
        self.model = None
        self._kernel_repr = ""

    # ---- internals -------------------------------------------------------
    def _t(self, a):
        return self.torch.as_tensor(np.asarray(a, float), dtype=self.dt, device=self.device)

    def _build(self, X, y, optimise):
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        self.torch.manual_seed(self.seed)
        tx, ty = self._t(X), self._t(y).unsqueeze(-1)
        model = SingleTaskGP(tx, ty, outcome_transform=Standardize(m=1))
        if optimise:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            self._state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            self._kernel_repr = (
                f"SingleTaskGP/{type(model.covar_module).__name__} "
                f"ls_median={float(model.covar_module.lengthscale.median()):.4g} "
                f"noise={float(model.likelihood.noise.mean()):.4g}")
        elif getattr(self, "_state", None) is not None:
            model.load_state_dict(self._state)       # keep hyperparameters, swap the data
        model.eval()
        self.model = model

    def _posterior(self, Xc):
        with self.torch.no_grad():
            return self.model.posterior(self._t(one_hot(Xc)))

    # ---- surrogate API ---------------------------------------------------
    def fit(self, X, y):
        self.X, self.y = one_hot(X), np.asarray(y, float)
        self._build(self.X, self.y, optimise=True)
        return self

    def condition(self, X_new, y_new):
        self.X = np.vstack([self.X, one_hot(X_new)])
        self.y = np.concatenate([self.y, np.asarray(y_new, float)])
        self._build(self.X, self.y, optimise=False)
        return self

    def reset_to(self, X, y):
        self.X, self.y = one_hot(X), np.asarray(y, float)
        self._build(self.X, self.y, optimise=False)
        return self

    def mean_std(self, Xc):
        p = self._posterior(Xc)
        m = p.mean.squeeze(-1).cpu().numpy()
        s = p.variance.clamp_min(1e-12).sqrt().squeeze(-1).cpu().numpy()
        return m, s

    def sample(self, Xc, n=1):
        """Joint posterior draws — BoTorch returns the full covariance, so these
        carry cross-candidate correlation exactly as `GPSurrogate.sample` does."""
        p = self._posterior(Xc)
        self.torch.manual_seed(self.seed + int(self.rng.integers(1 << 30)))
        with self.torch.no_grad():
            s = p.rsample(self.torch.Size([int(n)]))
        return s.squeeze(-1).cpu().numpy()

    def interval(self, Xc, level=0.8):
        from scipy.stats import norm
        m, s = self.mean_std(Xc)
        z = norm.ppf(0.5 + level / 2.0)
        return m - z * s, m + z * s

    @property
    def kernel_repr(self):
        return self._kernel_repr


def make_botorch_surrogate(seed=0, device="cpu"):
    return BoTorchGPSurrogate(seed=seed, device=device)
