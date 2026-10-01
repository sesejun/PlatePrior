"""High-dimensional BO baselines: TuRBO, SAASBO, GIT-BO (paper/追加実験設計.md §2.1).

All three were designed for *sequential* BO.  This plate benchmark asks for ~32
unique conditions per round, so each needs a documented batch adapter.  The
adapters below are stated explicitly and are **not** reproductions of the original
implementations — the design doc requires that distinction, since a weak adapter
would understate a baseline and flatter our own result.

Every arm draws from the SAME candidate pool and goes through the SAME well
allocation as every other arm, so the comparison isolates the model and the
selection rule rather than the search space.
"""
import os

import numpy as np

from simulator import N_SUB
from strategies import STRATEGIES, candidate_pool, allocate_wells, n_unique_picks

NUTS_WARMUP = int(os.environ.get("PBO_SAAS_WARMUP", "256"))
NUTS_SAMPLES = int(os.environ.get("PBO_SAAS_SAMPLES", "128"))
NUTS_THINNING = int(os.environ.get("PBO_SAAS_THINNING", "16"))
TURBO_INIT = float(os.environ.get("PBO_TURBO_L0", "0.8"))
GITBO_DIM = int(os.environ.get("PBO_GITBO_SUBDIM", "8"))


def _one_hot(X):
    X = np.asarray(X, float)
    return np.column_stack([X[:, :N_SUB], np.eye(3)[X[:, N_SUB].astype(int)]])


def _device(device):
    import torch
    return torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")


# --------------------------------------------------------------------------
# TuRBO (Eriksson et al., NeurIPS 2019)
# --------------------------------------------------------------------------
def strat_turbo(rng, X_obs, y_obs, seed=0, device="cpu", **_):
    """Trust-region BO, restricted to the shared candidate pool.

    Adapter: the original maintains a hyper-rectangle around the incumbent whose
    side lengths are rescaled by the fitted ARD lengthscales, and fills a batch by
    Thompson sampling inside it.  Here the trust region filters the shared pool and
    the batch is taken by Thompson sampling over the survivors, so TuRBO differs
    from the other arms in *where* it looks, not in which pool it may look at.
    The success/failure counters that expand and shrink the region need state
    across rounds; this benchmark re-fits per round, so the region is sized from
    the lengthscales alone (no adaptive resizing) and that is a simplification.
    """
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    dev, dt = _device(device), torch.double
    torch.manual_seed(int(seed))
    Xo, yo = _one_hot(X_obs), np.asarray(y_obs, float)
    tx = torch.as_tensor(Xo, dtype=dt, device=dev)
    ty = torch.as_tensor(yo, dtype=dt, device=dev).unsqueeze(-1)

    model = SingleTaskGP(tx, ty, outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()

    ls = model.covar_module.lengthscale.detach().flatten().cpu().numpy()
    ls = ls / max(ls.mean(), 1e-9)                    # normalise so the mean side is L0
    centre = Xo[int(np.argmax(yo))]

    pool = candidate_pool(rng, X_obs, y_obs)
    P = _one_hot(pool)
    half = 0.5 * TURBO_INIT * ls
    inside = np.all(np.abs(P - centre) <= half, axis=1)
    q = n_unique_picks()
    if inside.sum() < q:                              # region too tight: widen once
        d = np.abs((P - centre) / np.maximum(half, 1e-9)).max(1)
        inside = np.zeros(len(P), bool)
        inside[np.argsort(d)[: max(q * 4, 1)]] = True

    sub = pool[inside]
    with torch.no_grad():
        post = model.posterior(torch.as_tensor(_one_hot(sub), dtype=dt, device=dev))
        draws = post.rsample(torch.Size([q])).squeeze(-1).cpu().numpy()

    chosen, used = [], set()
    for row in draws:                                 # Thompson: one draw per pick
        for idx in np.argsort(-row):
            if idx not in used:
                used.add(int(idx)); chosen.append(sub[idx]); break
    return allocate_wells(np.array(chosen[:q]), rng)


# --------------------------------------------------------------------------
# SAASBO (Eriksson & Jankowiak, UAI 2021)
# --------------------------------------------------------------------------
def strat_saasbo(rng, X_obs, y_obs, seed=0, device="cpu", **_):
    """Sparse axis-aligned subspace GP, fully Bayesian via NUTS.

    Adapter: the original is sequential.  Here the batch is filled by
    sequential-greedy qLogNEI over the shared pool, which is the same batch rule
    the `gp_qlognei` arm uses — so `saasbo - gp_qlognei` isolates the SAAS prior
    and the fully Bayesian treatment rather than the batching.

    This is the expensive arm by design: NUTS replaces the MLE point estimate with
    MCMC over the kernel hyperparameters (reported at 40-68x a MAP fit), which is
    why the design doc restricts it to the highest dimension.
    """
    import torch
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.fit import fit_fully_bayesian_model_nuts
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
    from botorch.optim.optimize import optimize_acqf_discrete
    from botorch.sampling.normal import SobolQMCNormalSampler

    dev, dt = _device(device), torch.double
    torch.manual_seed(int(seed))
    tx = torch.as_tensor(_one_hot(X_obs), dtype=dt, device=dev)
    ty = torch.as_tensor(np.asarray(y_obs, float), dtype=dt, device=dev).unsqueeze(-1)

    model = SaasFullyBayesianSingleTaskGP(train_X=tx, train_Y=ty)
    fit_fully_bayesian_model_nuts(
        model, warmup_steps=NUTS_WARMUP, num_samples=NUTS_SAMPLES,
        thinning=NUTS_THINNING, disable_progbar=True, seed=int(seed))

    pool = candidate_pool(rng, X_obs, y_obs)
    choices = torch.as_tensor(_one_hot(pool), dtype=dt, device=dev)
    acq = qLogNoisyExpectedImprovement(
        model=model, X_baseline=tx, prune_baseline=True,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64])))
    cand, _ = optimize_acqf_discrete(acq, q=n_unique_picks(), choices=choices,
                                     unique=True, max_batch_size=128)
    C = choices.cpu().numpy()
    sel = [int(np.argmin(np.abs(C - row).sum(1))) for row in cand.cpu().numpy()]
    return allocate_wells(pool[sel], rng)


# --------------------------------------------------------------------------
# GIT-BO (Yu, Picard & Ahmed, ICLR 2026)
# --------------------------------------------------------------------------
def strat_gitbo(rng, surrogate, X_obs, y_obs, seed=0, **_):
    """Gradient-informed subspace search on a frozen tabular foundation model.

    Adapter, stated plainly: the original estimates an active subspace from the
    gradient of a frozen TabPFN's predictive mean and runs *sequential* BO inside
    it.  TabPFN exposes no gradients through this wrapper, so the subspace is
    estimated by finite differences around the incumbent, and the batch is filled
    by Kriging Believer within the subspace-restricted pool.  Two departures from
    the original — finite differences instead of autograd, and a batch rule the
    original does not specify — so this is a GIT-BO-*style* arm and must be
    labelled as such, never as a reproduction.
    """
    from strategies import expected_improvement

    Xo = np.asarray(X_obs, float)
    yo = np.asarray(y_obs, float)
    centre = Xo[int(np.argmax(yo))]

    # finite-difference sensitivity of the surrogate mean at the incumbent
    eps = 0.05
    base, _ = surrogate.mean_std(centre[None, :])
    grad = np.zeros(N_SUB)
    for i in range(N_SUB):
        xp = centre.copy(); xp[i] = np.clip(xp[i] + eps, 0, 1)
        mp, _ = surrogate.mean_std(xp[None, :])
        grad[i] = abs(float(mp[0]) - float(base[0])) / eps
    active = np.argsort(-grad)[:min(GITBO_DIM, N_SUB)]

    pool = candidate_pool(rng, X_obs, y_obs)
    # restrict to the active subspace: inactive coordinates are pinned to the
    # incumbent, so the search varies only the directions the model responds to
    P = pool.copy()
    inactive = np.setdiff1d(np.arange(N_SUB), active)
    P[:, inactive] = centre[inactive]

    q = n_unique_picks()
    chosen, best = [], yo.max()
    surrogate.fit(Xo, yo)
    work = P.copy()
    while len(chosen) < q:
        mu, sd = surrogate.mean_std(work)
        ei = expected_improvement(mu, sd, best)
        idx = np.argsort(-ei)[:4]
        chosen.extend(work[idx])
        surrogate.condition(work[idx], mu[idx])
        best = max(best, float(mu[idx].max()))
        work = np.delete(work, idx, axis=0)
    return allocate_wells(np.array(chosen[:q]), rng)


STRATEGIES.update({
    "turbo": ("botorch", strat_turbo),
    "saasbo": ("botorch", strat_saasbo),
    "gitbo_tabpfn": ("tabpfn", strat_gitbo),
})
