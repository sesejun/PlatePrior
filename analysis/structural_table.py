"""Contrast TabPFN KB against the two baselines that assume search structure.

TuRBO assumes the promising region is local; SAASBO assumes few coordinates
matter.  Both narrow the search using that assumption, so they gain where it
holds and lose where it does not.  TabPFN assumes neither.  The question this
table answers is what that costs: if dropping the assumptions were expensive, a
structured baseline should beat TabPFN somewhere in the sweep (Table 6).

Uses the same per-scenario normalized contrast and the same main cohort filter
as reanalyze.py, so the numbers sit on the same scale.

    uv run python analysis/structural_table.py
"""
import csv
import glob
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
BOOT = 20000
DIMS = (8, 20, 40, 80, 120)


def load():
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "results" / "v2a_runs_*.csv")))],
                   ignore_index=True)
    df = df.drop_duplicates(
        ["exp_id", "method", "config", "scenario", "seed", "round"], keep="last")
    # 埋め込み群と格子セルは主コホートではない
    return df[~df.config.str.contains("ambient|rotate|_e0", na=False)]


def contrast(df, d, a, b):
    sub = df[df.n_sub == d]
    if sub.empty:
        return None
    last = int(sub["round"].max())
    t = sub[sub["round"] == last].pivot_table(
        index="scenario", columns="method", values="best_true")
    if a not in t or b not in t:
        return None
    n = t.sub(t.min(axis=1), axis=0).div(
        (t.max(axis=1) - t.min(axis=1)).replace(0, np.nan), axis=0)
    return (n[a] - n[b]).dropna()


def main():
    df = load()
    rng = np.random.default_rng(20260922)
    rows = []
    for base in ("turbo", "saasbo"):
        for d in DIMS:
            g = contrast(df, d, "tabpfn_kb", base)
            if g is None or len(g) < 5:
                continue
            v = g.to_numpy()
            b = np.array([rng.choice(v, len(v), replace=True).mean()
                          for _ in range(BOOT)])
            wins = int((v > 0).sum())
            rows.append(dict(baseline=base, d=d, mean=v.mean(),
                             ci_lo=np.percentile(b, 2.5),
                             ci_hi=np.percentile(b, 97.5),
                             wins=wins, n=len(v),
                             p=binomtest(wins, len(v), 0.5).pvalue))
    # Holm 補正はこの族 (6 比較) の中で行う
    order = np.argsort([r["p"] for r in rows])
    m, prev = len(rows), 0.0
    for rank, i in enumerate(order):
        prev = rows[i]["holm"] = max(prev, min(1.0, (m - rank) * rows[i]["p"]))

    with (OUT / "structural_contrasts.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0]))
        wri.writeheader()
        wri.writerows(rows)
    print(f"wrote structural_contrasts.csv ({len(rows)} contrasts)")
    for r in rows:
        print(f"  d={r['d']:<4} vs {r['baseline']:<7} {r['mean']:+.3f} "
              f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  {r['wins']}/{r['n']}  "
              f"p={r['p']:.4f}  holm={r['holm']:.4f}")


if __name__ == "__main__":
    main()
