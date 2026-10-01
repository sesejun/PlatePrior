"""Parallel runner for the rapid-plate benchmark, with 80%-interval calibration logging.

Same protocol as run_benchmark.py (round 0 = shared LHS plate, rounds 1..R-1 =
strategy proposes one plate), plus:
  * multiprocessing over the (method, scenario, seed) grid
  * per-round calibration: fraction of the proposed plate whose NOISY observation
    falls inside the surrogate's 80% predictive interval, measured with the
    surrogate re-conditioned on real data only (KB pseudo-observations removed)
  * configuration recorded in every row so ablations can share one CSV

Ablation knobs are environment variables read at import time (see simulator.py /
strategies.py): PBO_N_SUB, PBO_N_WELLS, PBO_K_TOP, PBO_N_REP, PBO_CV,
PBO_N_GLOBAL, PBO_N_LOCAL.
"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# HOME is not writable in this sandbox; keep every library cache inside the workspace.
_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "cache"))
for _k, _v in [("SKRUB_DATA_DIRECTORY", "skrub"), ("HF_HOME", "hf"),
               ("TABPFN_MODEL_CACHE_DIR", "tabpfn"), ("XDG_CACHE_HOME", ""),
               ("MPLCONFIGDIR", "mpl")]:
    os.environ.setdefault(_k, os.path.join(_CACHE, _v) if _v else _CACHE)

import numpy as np
import pandas as pd

import simulator
import strategies
from simulator import MonodScenario, Plate96, lhs_designs
from surrogates import make_surrogate
from strategies import STRATEGIES, N_WELLS


def config_dict():
    import surrogates
    import embed
    model = surrogates.TabPFNSurrogate.MODEL
    tag = model.replace("tabpfn-", "").replace("-regressor", "").split(".ckpt")[0].split(".safetensors")[0]
    return dict(n_sub=simulator.N_SUB, n_wells=strategies.N_WELLS, k_top=strategies.K_TOP,
                n_rep=strategies.N_REP_TOP, cv=simulator.CV, tabpfn_model=tag,
                n_global=strategies.N_GLOBAL, n_local=strategies.N_LOCAL,
                embed=embed.EMBED, ess_p=simulator.ESS_P, w_conc=simulator.W_CONC)


def config_tag():
    c = config_dict()
    t = f"d{c['n_sub']}_w{c['n_wells']}_k{c['k_top']}x{c['n_rep']}_cv{c['cv']}"
    # 埋め込み群はタグに含める。含めないと ambient/rotate/intrinsic が同じ
    # キーになり、Volume の resume が別群の結果を再利用してしまう。
    if c.get("embed", "intrinsic") != "intrinsic":
        t += "_" + c["embed"]
    # 格子実験の 2 要因もタグに含める。含めないと 9 セルが同じキーになり
    # Volume の resume が別セルの結果を再利用する。
    if abs(c.get("ess_p", 0.35) - 0.35) > 1e-9 or abs(c.get("w_conc", 0.7) - 0.7) > 1e-9:
        t += f"_e{c['ess_p']}w{c['w_conc']}"
    return t if c["tabpfn_model"] == "v2" else t + "_" + c["tabpfn_model"]


def run_one(method, scen_id, seed, rounds, device="cpu", calibrate=True):
    kind, strat = STRATEGIES[method]
    sc = MonodScenario(scen_id)
    plate = Plate96(sc, seed=seed, n_wells=N_WELLS)
    rng = np.random.default_rng(1000 * scen_id + seed)
    cfg, tag = config_dict(), config_tag()
    true_cache = {}

    def truth(x):
        k = tuple(np.round(x, 6))
        if k not in true_cache:
            true_cache[k] = sc.true_f(x)
        return true_cache[k]

    X_all, y_all, rows = None, None, []
    for r in range(rounds):
        t0 = time.time()
        cal = {}
        if r == 0:
            Xr = lhs_designs(N_WELLS, np.random.default_rng(seed))   # shared initial plate
        elif kind == "none":
            Xr = strat(rng)
        elif kind == "botorch":
            Xr = strat(rng, X_obs=X_all, y_obs=y_all, seed=seed)
        else:
            sur = make_surrogate(kind, seed, device=device)
            sur.fit(X_all, y_all)
            Xr = strat(rng, surrogate=sur, X_obs=X_all, y_obs=y_all)
        wall = time.time() - t0          # proposal only: excludes instrumentation and the ODE plate
        if r > 0 and kind not in ("none", "botorch") and calibrate:
            cal = calibration(sur, X_all, y_all, Xr, plate)
        t1 = time.time()
        yr = plate.run(Xr)
        if cal:                       # coverage against the realised noisy plate
            inside = (yr >= cal.pop("_lo")) & (yr <= cal.pop("_hi"))
            edge = np.array([plate.is_edge(i) for i in range(len(Xr))])
            cal["cal_cov80"] = float(inside.mean())
            cal["cal_cov80_inner"] = float(inside[~edge].mean()) if (~edge).any() else np.nan
            cal["cal_cov80_edge"] = float(inside[edge].mean()) if edge.any() else np.nan
        X_all = Xr if X_all is None else np.vstack([X_all, Xr])
        y_all = yr if y_all is None else np.concatenate([y_all, yr])
        rows.append(dict(method=method, scenario=scen_id, seed=seed, round=r,
                         n_obs=len(y_all), best_true=max(truth(x) for x in X_all),
                         best_obs=float(y_all.max()), wall_s=wall,
                         plate_s=time.time() - t1, config=tag, **cfg, **cal))
    return rows


def calibration(sur, X_all, y_all, Xr, plate):
    """80% predictive interval on the proposed plate, real-data conditioning only."""
    try:
        sur.reset_to(X_all, y_all)
        lo, hi = sur.interval(Xr, level=0.8)
        width = float(np.mean(hi - lo))
        return {"_lo": np.asarray(lo, float), "_hi": np.asarray(hi, float),
                "cal_width": width, "cal_width_rel": width / max(float(np.std(y_all)), 1e-12)}
    except Exception as e:                       # never let instrumentation kill a run
        print("calibration failed:", type(e).__name__, e, flush=True)
        return {}


def _work(args):
    method, s, k, rounds, device = args
    try:
        t = time.time()
        rows = run_one(method, s, k, rounds, device)
        print(f"done {method} sc{s} seed{k} best={rows[-1]['best_true']:.3f} "
              f"({time.time()-t:.0f}s)", flush=True)
        return rows
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"FAILED {method} sc{s} seed{k}: {type(e).__name__}: {e}", flush=True)
        return []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--scenarios", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if any(m.startswith("gp_qlognei") for m in a.methods):
        import botorch_arm  # noqa: F401  (registers the arm)
    if any(m in ("turbo", "saasbo", "gitbo_tabpfn") for m in a.methods):
        import highdim_arms  # noqa: F401  (registers TuRBO / SAASBO / GIT-BO)

    grid = [(m, s, k, a.rounds, a.device)
            for m in a.methods for s in range(a.scenarios) for k in range(a.seeds)]
    done = set()
    if os.path.exists(a.out):                                   # resume support
        prev = pd.read_csv(a.out)
        prev = prev[prev["config"] == config_tag()] if "config" in prev else prev.iloc[:0]
        full = prev.groupby(["method", "scenario", "seed"])["round"].max()
        done = {k for k, v in full.items() if v == a.rounds - 1}
        grid = [g for g in grid if (g[0], g[1], g[2]) not in done]
        print(f"resuming: {len(done)} runs already complete, {len(grid)} to go", flush=True)

    print(f"config {config_tag()} | {len(grid)} runs | {a.workers} workers", flush=True)
    t0 = time.time()
    rows = []
    if a.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
        with ctx.Pool(a.workers) as pool:
            for i, rr in enumerate(pool.imap_unordered(_work, grid), 1):
                rows += rr
                if i % 10 == 0:
                    print(f"[{i}/{len(grid)}] {time.time()-t0:.0f}s", flush=True)
                    pd.DataFrame(rows).to_csv(a.out + ".partial", index=False)
    else:
        for g in grid:
            rows += _work(g)

    df = pd.DataFrame(rows)
    if os.path.exists(a.out) and len(df):
        df = pd.concat([pd.read_csv(a.out), df], ignore_index=True)
    elif os.path.exists(a.out):
        df = pd.read_csv(a.out)
    df = df.drop_duplicates(["method", "config", "scenario", "seed", "round"], keep="last")
    df.to_csv(a.out, index=False)
    print(f"saved {a.out}: {len(df)} rows in {time.time()-t0:.0f}s", flush=True)
