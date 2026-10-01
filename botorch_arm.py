"""BoTorch qLogNoisyExpectedImprovement arm (Phase A-2).

Rules out "the sklearn qEI implementation is simply too weak" as the explanation
for qEI losing to kb/lp/ts.  Uses BoTorch's own SingleTaskGP with the library's
defaults and outcome standardisation, and the sequential-greedy discrete
optimiser over the SAME candidate pool and the SAME well allocation as every
other arm, so only the model + acquisition differ.

Note on the model: `SingleTaskGP`'s defaults are version dependent.  Verified on
botorch 0.18.1, they are an **RBF** kernel with the dimension-scaled LogNormal
lengthscale prior of Hvarfner et al. (2024), l_i ~ LN(sqrt(2)+log(D)/2, sqrt(3)) --
not the Matern-5/2 an earlier version of this comment claimed.  This arm therefore
already carries the strong high-dimensional prior; the resolved kernel is recorded
per run under protocol v2 rather than assumed.
"""
import os
import numpy as np

from simulator import N_SUB
from strategies import STRATEGIES, candidate_pool, allocate_wells, n_unique_picks

MC_SAMPLES = int(os.environ.get("PBO_BOTORCH_MC", "128"))


def _one_hot(X):
    X = np.asarray(X, float)
    return np.column_stack([X[:, :N_SUB], np.eye(3)[X[:, N_SUB].astype(int)]])


def strat_qlognei(rng, X_obs, y_obs, seed=0, device=None, **_):
    """device=None は従来どおり CPU 単スレッド。device="cuda" を渡すと
    GP のフィットと獲得関数最適化を GPU 上で行う。TabPFN アームとの
    計算コスト比較を「GPU 対 GPU」で取るために追加した経路で、
    既定 (gp_qlognei) の挙動は変えていない。"""
    import torch
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim.optimize import optimize_acqf_discrete
    from botorch.sampling.normal import SobolQMCNormalSampler
    from gpytorch.mlls import ExactMarginalLogLikelihood

    if device == "cuda" and not torch.cuda.is_available():
        device = None                       # GPU が無い環境では黙って CPU に落とす
    dev = torch.device(device or "cpu")
    if dev.type == "cpu":
        torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    dt = torch.double

    train_X = torch.as_tensor(_one_hot(X_obs), dtype=dt, device=dev)
    train_Y = torch.as_tensor(np.asarray(y_obs, float), dtype=dt, device=dev).unsqueeze(-1)

    model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

    pool = candidate_pool(rng, X_obs, y_obs)
    choices = torch.as_tensor(_one_hot(pool), dtype=dt, device=dev)

    acq = qLogNoisyExpectedImprovement(
        model=model, X_baseline=train_X, prune_baseline=True,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES])))

    q = n_unique_picks()
    cand, _ = optimize_acqf_discrete(acq, q=q, choices=choices, unique=True,
                                     max_batch_size=256)
    # map the chosen one-hot rows back to pool rows (exact rows of `choices`)
    C = choices.cpu().numpy()
    sel = [int(np.argmin(np.abs(C - row).sum(1))) for row in cand.cpu().numpy()]
    return allocate_wells(pool[sel], rng)


def strat_qlognei_gpu(rng, X_obs, y_obs, seed=0, **kw):
    return strat_qlognei(rng, X_obs, y_obs, seed=seed, device="cuda")


STRATEGIES["gp_qlognei"] = ("botorch", strat_qlognei)
# GPU 経路。TabPFN と同じ A10G 上で回し、ハードウェアを揃えた
# 計算コスト比較を取るためのアーム (探索の挙動は gp_qlognei と同一)。
STRATEGIES["gp_qlognei_gpu"] = ("botorch", strat_qlognei_gpu)
