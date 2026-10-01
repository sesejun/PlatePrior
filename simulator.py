"""In silico 96-well culture simulator (Monod-type, 8 substrates + 1 categorical).

Design space (mixed variables):
  x[0:8]  : substrate concentrations in [0, 1] (normalised, scaled to g/L internally)
  x[8]    : carbon-source type, categorical {0,1,2}  (changes mu_max / yield)
Constraint: sum of substrate concentrations <= V_MAX (total-volume constraint,
            handled by proportional rescaling as in Lapierre et al. 2025).
Objective : final biomass after T_END hours (maximise). Measurement noise = CV%
            plus optional plate edge effect.
"""
import os
import numpy as np
from scipy.integrate import solve_ivp

# --- ablation hooks (env vars; defaults reproduce the original fixed設定) ---
N_SUB = int(os.environ.get("PBO_N_SUB", "8"))
# total-volume budget.  Original: 4.0 with N_SUB=8, i.e. half of the maximum
# attainable sum.  Kept proportional so the constraint binds equally often when
# the dimension is raised (otherwise 20/40 substrates would always be rescaled).
V_MAX = float(os.environ.get("PBO_V_MAX", str(0.5 * N_SUB)))
CV = float(os.environ.get("PBO_CV", "0.06"))
# P1-B 追試 (格子実験): 「基質数が多いと TabPFN 有利」の機序を切り分ける 2 要因。
# どちらも従来ハードコードされていた値で、既定はその値のまま。
#   ESS_P  必須基質の割合。Liebig の最小律で効くので、多いほど目的関数が
#          min() の折れ線になり非平滑になる
#   W_CONC Dirichlet の濃度パラメータ。小さいほど重みが少数基質へ集中し、
#          大きいほど全基質へ均等に分散する
ESS_P = float(os.environ.get("PBO_ESS_P", "0.35"))
W_CONC = float(os.environ.get("PBO_W_CONC", "0.7"))
T_END = 24.0
S_SCALE = 10.0       # g/L at x=1


def plate_shape(n_wells):
    """Standard microplate geometries used for the batch-width ablation."""
    return {96: (8, 12), 48: (6, 8), 24: (4, 6), 12: (3, 4)}[int(n_wells)]


class MonodScenario:
    """One random 'organism' (process constants). 10 scenarios ~ Lapierre et al."""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        # 生物としての定数は「実際に代謝する基質数」で作る。embed モードでは
        # これが intrinsic 次元 (既定 8) に固定され、探索空間の次元 N_SUB を
        # 上げても目的関数そのものは変わらない。intrinsic モードでは
        # N_SUB と一致するので、従来の挙動と同一。
        import embed as _emb
        n = _emb.BASE_SUB if _emb.EMBED != "intrinsic" else N_SUB
        self.n_metab = n
        self.seed = int(seed)
        self.mu_max = rng.uniform(0.2, 0.6)                   # 1/h
        self.kd = rng.uniform(0.005, 0.03)                    # death rate
        self.Ks = rng.uniform(0.2, 2.0, n)                    # half-sat (g/L)
        self.Ki = rng.uniform(15.0, 60.0, n)                  # substrate inhibition
        self.Y = rng.uniform(0.2, 0.6, n)                     # yield gX/gS
        self.w = rng.dirichlet(np.ones(n) * W_CONC)           # substrate importance
        self.essential = rng.random(n) < ESS_P                # essential (min-law)
        if not self.essential.any():
            self.essential[rng.integers(n)] = True
        self.carbon_gain = rng.uniform(0.7, 1.3, 3)           # categorical effect
        self.carbon_gain[rng.integers(3)] = 1.3

    def _rhs(self, t, z, c_gain):
        X, S = z[0], np.maximum(z[1:], 0.0)
        # Haldane term per substrate
        f = S / (self.Ks + S + S**2 / self.Ki)
        # essential substrates: Liebig minimum; non-essential: weighted sum
        f_ess = f[self.essential].min()
        f_non = (self.w[~self.essential] * f[~self.essential]).sum() / max(self.w[~self.essential].sum(), 1e-9) if (~self.essential).any() else 1.0
        mu = self.mu_max * c_gain * f_ess * (0.5 + 0.5 * f_non)
        dX = (mu - self.kd) * X
        uptake = mu * X * self.w / self.Y
        dS = -uptake * (S > 1e-6)
        return np.concatenate([[dX], dS])

    def true_f(self, x):
        """Noise-free final biomass for one design vector x (len 9)."""
        import embed as _emb
        x = np.asarray(x, float)
        conc = np.clip(x[:N_SUB], 0, 1)
        if conc.sum() > V_MAX:                                # volume constraint
            conc = conc * V_MAX / conc.sum()
        # 埋め込みモードでは、制約を課した「探索空間の設計」を代謝する
        # 基質濃度へ射影する。制約は探索空間側に、ODE は intrinsic 側に
        # かかるので、次元を上げても目的関数の難しさは変わらない。
        conc = _emb.project(conc, self.seed)
        c_gain = self.carbon_gain[int(round(x[N_SUB]))]
        z0 = np.concatenate([[0.05], conc * S_SCALE])
        sol = solve_ivp(self._rhs, (0, T_END), z0, args=(c_gain,), rtol=1e-5, atol=1e-7)
        return float(sol.y[0, -1])


class Plate96:
    """Runs one 96-well plate of designs with noise + edge effect."""

    def __init__(self, scenario: MonodScenario, cv=None, edge_penalty=0.08, seed=0, n_wells=96):
        cv = CV if cv is None else cv
        self.sc, self.cv, self.edge, self.rng = scenario, cv, edge_penalty, np.random.default_rng(seed)
        self.nrow, self.ncol = plate_shape(n_wells)

    def is_edge(self, well):
        r, c = divmod(well, self.ncol)
        return r in (0, self.nrow - 1) or c in (0, self.ncol - 1)

    def run(self, X):
        """X: (n<=n_wells, N_SUB+1). Returns noisy measurements y (n,). Well i -> position i."""
        y = np.array([self.sc.true_f(x) for x in X])
        edge = np.array([self.is_edge(i) for i in range(len(X))])
        y = y * (1 - self.edge * edge)
        y = y * (1 + self.cv * self.rng.standard_normal(len(y)))
        return np.maximum(y, 0.0)


def sample_designs(n, rng):
    """Uniform random feasible designs (used for candidate pools)."""
    conc = rng.random((n, N_SUB))
    cat = rng.integers(0, 3, n)
    return np.column_stack([conc, cat]).astype(float)


def lhs_designs(n, rng):
    """Latin hypercube for the continuous part + balanced categorical."""
    cut = np.linspace(0, 1, n + 1)
    u = rng.random((n, N_SUB))
    pts = cut[:-1, None] + u * (1.0 / n)
    for j in range(N_SUB):
        pts[:, j] = pts[rng.permutation(n), j]
    cat = np.tile(np.arange(3), n // 3 + 1)[:n]
    rng.shuffle(cat)
    return np.column_stack([pts, cat]).astype(float)


if __name__ == "__main__":
    sc = MonodScenario(0)
    rng = np.random.default_rng(0)
    X = lhs_designs(96, rng)
    y = Plate96(sc).run(X)
    print("scenario 0: biomass range on LHS plate", y.min().round(3), y.max().round(3))
