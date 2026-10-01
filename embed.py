"""P1-B: separate the effect of dimension from the effect of a harder objective.

Design doc §3.  Raising `PBO_N_SUB` changes the ambient dimension, but it also
changes the objective itself: more substrates means more essential ones (each
drawn at p=0.35), a Dirichlet weight vector over more components, and a volume
budget `V_MAX = 0.5 * N_SUB` that scales with it.  So "d=120 beats d=8" cannot
be attributed to dimension — the d=120 problem is a different problem.

This module keeps ONE 8-substrate objective and varies only how it is embedded
in a d-dimensional search space:

  ambient  x -> take the first 8 coordinates; the other d-8 are ignored.
               Intrinsic dimension stays 8; only the ambient dimension grows.
               Tests robustness to irrelevant variables.
  rotate   x -> A x, with A a random 8 x d matrix with orthonormal rows.
               Intrinsic dimension stays 8 but is no longer axis-aligned, so
               ARD lengthscales and axis-aligned subspace methods cannot exploit
               it.  Tests robustness to axis misalignment.
  intrinsic  the existing behaviour: a genuinely d-substrate organism.
               This is the group the earlier runs actually measured.

The three groups share the scenario's organism constants, the categorical
carbon source, the noise level, the well budget and the round budget, so within
a dimension they differ only in the embedding.

`PBO_EMBED` selects the group; `PBO_EMBED_DIM` sets the ambient dimension.  With
`PBO_EMBED=intrinsic` (the default) nothing changes, so existing results and
code paths are untouched.
"""
import os

import numpy as np

EMBED = os.environ.get("PBO_EMBED", "intrinsic")
EMBED_DIM = int(os.environ.get("PBO_EMBED_DIM", "0"))   # 0 = そのまま
BASE_SUB = int(os.environ.get("PBO_BASE_SUB", "8"))     # intrinsic 次元


def active_dim():
    """Search-space dimension the strategies see."""
    return EMBED_DIM if (EMBED != "intrinsic" and EMBED_DIM) else BASE_SUB


def _rotation(seed, d_out, d_in):
    """Random matrix with orthonormal rows (d_out x d_in), fixed per scenario."""
    rng = np.random.default_rng(10_000 + int(seed))
    A = rng.standard_normal((d_in, d_out))
    Q, _ = np.linalg.qr(A)                       # (d_in, d_out) with orthonormal cols
    return Q.T                                    # (d_out, d_in)


def project(conc, seed):
    """Map a design in the search space to the 8 substrate concentrations.

    Returns a vector of length BASE_SUB in [0, 1].
    """
    conc = np.asarray(conc, float)
    if EMBED == "intrinsic" or not EMBED_DIM:
        return conc
    if EMBED == "ambient":
        return conc[:BASE_SUB]
    if EMBED == "rotate":
        A = _rotation(seed, BASE_SUB, len(conc))
        z = A @ (conc - 0.5)                     # 中心を原点に置いてから回転
        # 設計ごとの min-max 正規化は使わない。それだと各設計が必ず [0,1] 全体に
        # 広がり、必ず 1 つの基質が 0 になる。必須基質の最小値則に引っかかって
        # 「回転の効果」ではなく「常にどれかが枯渇する」効果を測ってしまう。
        # 代わりに、直交行列が分散を 1/sqrt(d_in/d_out) 倍する分だけを戻す
        # 固定スケールを使い、分布の形を baseline に合わせる。
        z = z * np.sqrt(len(conc) / BASE_SUB)
        return np.clip(z + 0.5, 0.0, 1.0)
    raise ValueError(f"unknown PBO_EMBED: {EMBED}")


def describe():
    return {"embed": EMBED, "embed_dim": EMBED_DIM if EMBED != "intrinsic" else 0,
            "base_sub": BASE_SUB, "active_dim": active_dim()}
