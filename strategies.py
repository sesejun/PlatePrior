"""Batch-selection strategies for one 96-well round.

Every strategy returns a (96, 9) design matrix (rows = wells, already shuffled to
decorrelate the plate edge effect).  Learning strategies receive the surrogate
already fitted on all data so far.
"""
import os
import numpy as np
from scipy.stats import norm
from simulator import N_SUB, sample_designs, lhs_designs

# --- ablation hooks (env vars; defaults reproduce the original fixed設定) ---
N_WELLS = int(os.environ.get("PBO_N_WELLS", "96"))
K_TOP = int(os.environ.get("PBO_K_TOP", "8"))        # acquisition picks that get replicated
N_REP_TOP = int(os.environ.get("PBO_N_REP", "3"))    # replicates each


def n_unique_picks():
    """Number of distinct designs a strategy must choose for one plate."""
    return N_WELLS - K_TOP * (N_REP_TOP - 1)


N_GLOBAL = int(os.environ.get("PBO_N_GLOBAL", "1200"))
N_LOCAL = int(os.environ.get("PBO_N_LOCAL", "300"))


def candidate_pool(rng, X_obs, y_obs, n_global=None, n_local=None, radius=0.12):
    n_global = N_GLOBAL if n_global is None else n_global
    n_local = N_LOCAL if n_local is None else n_local
    """Global uniform pool + local perturbations around the best observed designs."""
    pool = [sample_designs(n_global, rng)]
    if X_obs is not None and len(X_obs):
        top = X_obs[np.argsort(-y_obs)[:10]]
        loc = top[rng.integers(0, len(top), n_local)].copy()
        loc[:, :N_SUB] = np.clip(loc[:, :N_SUB] + radius * rng.standard_normal((n_local, N_SUB)), 0, 1)
        flip = rng.random(n_local) < 0.15
        loc[flip, N_SUB] = rng.integers(0, 3, flip.sum())
        pool.append(loc)
    return np.vstack(pool)


def allocate_wells(ranked, rng, k_top=None, n_rep_top=None):
    """Top-k acquisition picks get n_rep_top replicates; the rest single wells.
    ranked: candidates ordered by acquisition (best first)."""
    k_top = K_TOP if k_top is None else k_top
    n_rep_top = N_REP_TOP if n_rep_top is None else n_rep_top
    n_unique = N_WELLS - k_top * (n_rep_top - 1)
    uniq = ranked[:n_unique]
    plate = np.vstack([np.repeat(uniq[:k_top], n_rep_top, axis=0), uniq[k_top:]])
    assert len(plate) == N_WELLS
    return plate[rng.permutation(N_WELLS)]


def expected_improvement(mu, sd, best, xi=0.0):
    sd = np.maximum(sd, 1e-9); z = (mu - best - xi) / sd
    return (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)


# ---------- strategies ----------
def strat_random(rng, **_):
    return sample_designs(N_WELLS, rng)


def strat_lhs(rng, **_):
    return lhs_designs(N_WELLS, rng)


def strat_kb_ei(rng, surrogate, X_obs, y_obs, kb_chunk=4, **_):
    """Kriging Believer with EI: pick, believe the predicted mean, re-condition, repeat."""
    pool = candidate_pool(rng, X_obs, y_obs)
    best = y_obs.max(); chosen = []
    n_unique = n_unique_picks()
    while len(chosen) < n_unique:
        mu, sd = surrogate.mean_std(pool)
        ei = expected_improvement(mu, sd, best)
        idx = np.argsort(-ei)[:kb_chunk]
        chosen.extend(pool[idx])
        surrogate.condition(pool[idx], mu[idx])       # believe the mean
        best = max(best, mu[idx].max())
        pool = np.delete(pool, idx, axis=0)
    return allocate_wells(np.array(chosen[:n_unique]), rng)


def strat_ts(rng, surrogate, X_obs, y_obs, ts_chunk=8, **_):
    """Batch Thompson sampling: one posterior draw per pick (GP joint / TabPFN marginal)."""
    pool = candidate_pool(rng, X_obs, y_obs)
    n_unique = n_unique_picks()
    S = surrogate.sample(pool, n=n_unique)             # (n_unique, n_pool)
    chosen, used = [], set()
    for s in S:
        for idx in np.argsort(-s):
            if idx not in used:
                used.add(idx); chosen.append(pool[idx]); break
    # order by posterior mean so replicates go to the most promising picks
    chosen = np.array(chosen)
    mu, _ = surrogate.mean_std(chosen)
    return allocate_wells(chosen[np.argsort(-mu)], rng)


STRATEGIES = {
    "random": ("none", strat_random),
    "lhs":    ("none", strat_lhs),
    "gp_kb":  ("gp", strat_kb_ei),
    "gp_ts":  ("gp", strat_ts),
    "tabpfn_kb": ("tabpfn", strat_kb_ei),
    "tabpfn_ts": ("tabpfn", strat_ts),
}


# ---------- proper batch acquisition: Local Penalization (Gonzalez et al. 2016) ----------
def strat_lp_ei(rng, surrogate, X_obs, y_obs, **_):
    """Batch EI with local penalisation (BB-LP style, as used by Lapierre et al. 2025).
    Surrogate-agnostic: needs only mean/std, so it works for GP and TabPFN alike.
    Each chosen point x_j penalises candidates within a ball whose radius is
    r_j = (M - mu_j) / L, M = current best, L = estimated Lipschitz constant."""
    pool = candidate_pool(rng, X_obs, y_obs)
    Xc = np.column_stack([pool[:, :N_SUB], np.eye(3)[pool[:, N_SUB].astype(int)]])   # metric space
    mu, sd = surrogate.mean_std(pool)
    ei = expected_improvement(mu, sd, y_obs.max())
    # Lipschitz estimate: max finite-difference slope among random candidate pairs
    i, j = rng.integers(0, len(Xc), (2, 4000))
    d = np.linalg.norm(Xc[i] - Xc[j], axis=1); d = np.where(d < 1e-9, 1e-9, d)
    L = max(np.percentile(np.abs(mu[i] - mu[j]) / d, 99), 1e-3)
    M = max(y_obs.max(), mu.max())
    n_unique = n_unique_picks()
    log_pen = np.zeros(len(pool)); chosen = []
    for _ in range(n_unique):
        k = int(np.argmax(np.log(np.maximum(ei, 1e-12)) + log_pen))
        chosen.append(pool[k])
        r = (M - mu[k]) / L
        dist = np.linalg.norm(Xc - Xc[k], axis=1)
        z = (dist - r) / (sd[k] / L + 1e-9)                # soft "outside the ball" probability
        log_pen += np.log(np.clip(norm.cdf(z), 1e-12, 1.0))
        log_pen[k] = -np.inf
    return allocate_wells(np.array(chosen), rng)


STRATEGIES.update({
    "gp_lp": ("gp", strat_lp_ei),
    "tabpfn_lp": ("tabpfn", strat_lp_ei),
})


# ---------- Monte-Carlo qEI, greedy-sequential (Wilson et al. 2018 / BoTorch sequential=True) ----------
def strat_qei(rng, surrogate, X_obs, y_obs, n_mc=256, independent=False, acq_rng=None, **_):
    """qEI over a candidate pool with fixed base samples.
    independent=False : joint posterior samples (exact for GP)  -> true batch qEI
    independent=True  : samples decorrelated across candidates (what a marginal-only
                        surrogate such as TabPFN can offer) -> 'marginal qEI' approximation.

    acq_rng: generator for the decorrelation draws.  Under protocol v2 this is a
    stream separate from `rng`, so the n_pool permutations the independent variant
    consumes cannot shift the candidate pool of any later round — in v1 they did,
    which confounded the independent-vs-joint contrast with a different pool."""
    pool = candidate_pool(rng, X_obs, y_obs)
    S = surrogate.sample(pool, n=n_mc)                    # (n_mc, n_pool)
    if independent:                                        # keep each marginal, destroy correlation
        r = acq_rng if acq_rng is not None else rng
        S = np.stack([S[r.permutation(n_mc), c] for c in range(S.shape[1])], axis=1)
    best = y_obs.max()
    batch_max = np.full(n_mc, -np.inf); chosen, used = [], np.zeros(len(pool), bool)
    n_unique = n_unique_picks()
    for _ in range(n_unique):
        gain = np.maximum(np.maximum(batch_max[:, None], S) - best, 0.0).mean(0)   # qEI of batch ∪ {c}
        gain[used] = -np.inf
        k = int(np.argmax(gain)); used[k] = True; chosen.append(pool[k])
        batch_max = np.maximum(batch_max, S[:, k])
    return allocate_wells(np.array(chosen), rng)


def strat_qei_indep(rng, **kw):
    return strat_qei(rng, independent=True, **kw)


STRATEGIES.update({
    "gp_qei": ("gp", strat_qei),            # joint samples  (GP reference)
    "gp_qei_indep": ("gp", strat_qei_indep),  # marginal-only proxy for TabPFN's limitation
    "tabpfn_qei": ("tabpfn", strat_qei),    # TabPFN samples are marginal by construction
})

# Strong-GP arms: BoTorch SingleTaskGP, whose default lengthscale prior since
# 0.12 is the dimension-scaled LogNormal of Hvarfner et al. (2024).  They run the
# SAME acquisitions as the sklearn GP and TabPFN arms, so `gp_bo_kb - tabpfn_kb`
# isolates the surrogate and `gp_bo_kb - gp_kb` isolates the GP configuration.
STRATEGIES.update({
    "gp_bo_kb": ("gp_bo", strat_kb_ei),
    "gp_bo_lp": ("gp_bo", strat_lp_ei),
    "gp_bo_ts": ("gp_bo", strat_ts),
})
