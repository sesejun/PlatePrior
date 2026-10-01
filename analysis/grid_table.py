"""Factorial grid: essential-component probability x contribution-weight concentration.

Writes the cell contrasts (TabPFN KB minus DS GP KB, normalized score) and the
standardized regression with interaction to analysis/tables/ (Table 5).

    uv run python analysis/grid_table.py
"""
import csv
import glob
import itertools
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
PAIR = ("tabpfn_kb", "gp_bo_kb")
BOOT = 20000


# 格子は 20 シナリオ x 3 シード x 6 手法で回した。中央セル (0.35, 0.7) は
# 既定値なので config_tag に接尾辞が付かず、同じキーの P1A/P3 走行
# (25 シナリオ x 5 シード x 10 手法) と同居する。揃えずに使うと、その
# セルだけ n とアーム集合が違い、正規化の基準も変わってしまう。
GRID_ARMS = ["tabpfn_kb", "tabpfn_lp", "gp_kb", "gp_bo_kb", "gp_qlognei", "lhs"]
GRID_SCENARIOS = range(20, 40)
GRID_SEEDS = range(3)


def load():
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "results" / "v2a_runs_*.csv")))],
                   ignore_index=True)
    df = df.drop_duplicates(
        ["exp_id", "method", "config", "scenario", "seed", "round"], keep="last")
    g = df[(df.n_sub == 120) & ~df.config.str.contains("ambient|rotate", na=False)].copy()
    g["ess_p"] = g.ess_p.fillna(0.35)
    g["w_conc"] = g.w_conc.fillna(0.7)
    # 全セルを同一条件に切り揃える
    g = g[g.method.isin(GRID_ARMS)
          & g.scenario.isin(GRID_SCENARIOS)
          & g.seed.isin(GRID_SEEDS)]
    return g


def gap(sub):
    last = int(sub["round"].max())
    t = sub[sub["round"] == last].pivot_table(
        index="scenario", columns="method", values="best_true")
    if PAIR[0] not in t or PAIR[1] not in t:
        return None
    n = t.sub(t.min(axis=1), axis=0).div(
        (t.max(axis=1) - t.min(axis=1)).replace(0, np.nan), axis=0)
    return (n[PAIR[0]] - n[PAIR[1]]).dropna()


def main():
    g = load()
    rng = np.random.default_rng(20260921)
    ess = sorted(g.ess_p.unique())
    wc = sorted(g.w_conc.unique())

    cells, rows = {}, []
    for e, w in itertools.product(ess, wc):
        d = gap(g[(g.ess_p == e) & (g.w_conc == w)])
        if d is None or len(d) < 3:
            continue
        cells[(e, w)] = d
        v = d.to_numpy()
        b = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(BOOT)])
        wins = int((v > 0).sum())
        rows.append(dict(ess_p=e, w_conc=w, mean=v.mean(), median=float(np.median(v)),
                         ci_lo=np.percentile(b, 2.5), ci_hi=np.percentile(b, 97.5),
                         wins=wins, n=len(v),
                         p=binomtest(wins, len(v), 0.5).pvalue))
    # Holm 補正は 9 セル族の中で行う
    order = np.argsort([r["p"] for r in rows])
    m, prev = len(rows), 0.0
    for rank, i in enumerate(order):
        prev = rows[i]["holm"] = max(prev, min(1.0, (m - rank) * rows[i]["p"]))

    with (OUT / "grid_contrasts.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0]))
        wri.writeheader(); wri.writerows(rows)

    # 交互作用つき回帰。係数の不確実性はブートストラップ (誤差の正規性を仮定しない)
    X, y = [], []
    for (e, w), d in cells.items():
        for val in d.to_numpy():
            X.append([e, w]); y.append(val)
    X = np.asarray(X, float); y = np.asarray(y)
    Xs = (X - X.mean(0)) / X.std(0)
    design = np.column_stack([np.ones(len(Xs)), Xs, Xs[:, 0] * Xs[:, 1]])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    bs = []
    for _ in range(BOOT // 4):
        idx = rng.integers(0, len(y), len(y))
        b, *_ = np.linalg.lstsq(design[idx], y[idx], rcond=None)
        bs.append(b)
    bs = np.asarray(bs)

    with (OUT / "grid_regression.csv").open("w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["term", "coef", "ci_lo", "ci_hi", "r2", "n"])
        for i, nm in enumerate(["intercept", "essential_fraction",
                                "weight_concentration", "interaction"]):
            lo, hi = np.percentile(bs[:, i], [2.5, 97.5])
            wri.writerow([nm, f"{beta[i]:.6f}", f"{lo:.6f}", f"{hi:.6f}",
                          f"{r2:.6f}", len(y)])

    print(f"wrote grid_contrasts.csv ({len(rows)} cells) and grid_regression.csv")
    print(f"  essential fraction levels: {ess}")
    print(f"  weight concentration levels: {wc}")
    print(f"  R^2 = {r2:.4f}, n = {len(y)} scenario-cells")


if __name__ == "__main__":
    main()
