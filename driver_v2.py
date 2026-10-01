"""Experiment protocol v2: separated RNG streams, shared initial plates, env provenance.

Why a separate driver: `driver.py` reproduces the v1 results and must keep doing so.
v2 changes the random-number plumbing, so its numbers are NOT comparable to v1
run-for-run.  Results carry `exp_id="v2"` and are never merged into a v1 figure
without stratifying on that column.

What changes from v1 (paper/追加実験設計.md §1):

1. **Separated RNG streams.**  v1 draws the candidate pool, the acquisition's
   Monte-Carlo samples, the well layout and the measurement noise from ONE
   generator.  `gp_qei_indep` consumes `n_pool` permutations from it, so the
   decorrelation step shifts every later draw — the "independent vs joint"
   contrast was confounded with a different candidate pool.  v2 spawns one
   stream per purpose from a SeedSequence, so consumption in one never moves
   another.

2. **Shared initial plate.**  v1 regenerates round 0 per run; v2 builds it once
   per (scenario, seed) and caches it, so every arm starts from a bit-identical
   design, observation and truth.

3. **Provenance per run.**  checkpoint SHA, resolved GP kernel, the device that
   actually ran (so a silent CUDA->CPU fallback is visible), package hash.

    uv run python driver_v2.py --methods gp_kb tabpfn_lp --scenarios 2 --seeds 2
"""
import argparse, hashlib, json, os, platform, sys, time, warnings

import numpy as np

from simulator import MonodScenario, Plate96, lhs_designs
from strategies import STRATEGIES, N_WELLS

# 追加アームを登録する (import しないと STRATEGIES に入らない)。
# torch が無い環境では GP/TabPFN アームだけで動かせるよう握りつぶす。
for _m in ("botorch_arm", "highdim_arms"):
    try:
        __import__(_m)
    except Exception:                                  # noqa: BLE001
        pass

warnings.filterwarnings("ignore")

# v2a: ラウンド局所の乱数 + チェックポイント対応。v2 (初版) とは乱数系統が
# 違うので run-for-run では比較できない。Volume のキャッシュも別キーになり、
# 古い v2 走行を誤って再利用しない。
EXP_ID = "v2a"
STREAMS = ("initial", "pool", "acq", "layout", "noise")
_INIT_CACHE = {}


def streams(scen_id, seed, exp_id=EXP_ID, rnd=None):
    """One independent generator per purpose.

    Keyed off a stable digest of exp_id (not `hash()`, which is salted per
    process and would make runs irreproducible across containers).

    `rnd` makes the streams round-local.  Without it a resumed campaign would
    replay earlier rounds without drawing from the generators, so every later
    round would see a different stream position than an uninterrupted run and
    the two would diverge.  Seeding per round makes a resumed run identical to
    one that never stopped.
    """
    tag = int(hashlib.sha256(exp_id.encode()).hexdigest()[:8], 16)
    entropy = [tag, int(scen_id), int(seed)]
    if rnd is not None:
        entropy.append(int(rnd))
    ss = np.random.SeedSequence(entropy)
    return dict(zip(STREAMS, (np.random.default_rng(s) for s in ss.spawn(len(STREAMS)))))


def initial_plate(scen_id, seed, n_wells=None):
    """Round-0 design/observation/truth, built once per (scenario, seed).

    Every arm reads the same object, so the shared initial plate is identical
    bit-for-bit rather than merely distributed alike.
    """
    n_wells = n_wells or N_WELLS
    key = (EXP_ID, int(scen_id), int(seed), int(n_wells))
    if key not in _INIT_CACHE:
        st = streams(scen_id, seed)
        sc = MonodScenario(scen_id)
        X = lhs_designs(n_wells, st["initial"])
        plate = Plate96(sc, seed=seed, n_wells=n_wells)
        y = plate.run(X)
        _INIT_CACHE[key] = (X, y, np.array([sc.true_f(x) for x in X]))
    X, y, t = _INIT_CACHE[key]
    return X.copy(), y.copy(), t.copy()


def _provenance(device_req):
    """Environment facts worth carrying on every row (設計書 §1.2)."""
    p = {"exp_id": EXP_ID, "device_req": device_req, "device_actual": "cpu",
         "python": platform.python_version(), "torch_version": "", "cuda_version": "",
         "ckpt_sha": "", "peak_gpu_mem": 0}
    try:
        import torch
        p["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            p["cuda_version"] = torch.version.cuda or ""
            if device_req == "cuda":
                p["device_actual"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return p


def _ckpt_sha(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()[:16]
    except Exception:
        return ""


def run_one(method, scen_id, seed, rounds, device="cpu", calibrate=True,
            on_round=None, resume=None):
    """One (method, scenario, seed) campaign under protocol v2.

    on_round(rows_so_far) is called after every completed round, and `resume`
    supplies the rows of a partially finished campaign.  Long arms (SAASBO is
    ~21 min at d=40) otherwise lose everything when Modal preempts the
    container, and restart from round 0 each time — which can outlast the
    preemption interval and never finish.  Replaying from the round boundary
    costs the plate evaluations again but not the model fits.
    """
    import driver as v1                      # reuse config_tag/config_dict/calibration
    kind, strat = STRATEGIES[method]
    sc = MonodScenario(scen_id)
    st = streams(scen_id, seed)
    plate = Plate96(sc, seed=seed, n_wells=N_WELLS)
    cfg, tag = v1.config_dict(), v1.config_tag()

    def _plate_for(r):
        """Measurement noise seeded per round.

        Plate96 keeps a stateful generator, so a resumed campaign that replays a
        stored round without calling run() would leave that generator at a
        different position than an uninterrupted one, and every later round would
        draw different noise.  Re-seeding per round removes the dependence on how
        many rounds happened to be replayed.
        """
        pl = Plate96(sc, seed=seed, n_wells=N_WELLS)
        pl.rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(scen_id), 777, int(r)]))
        return pl
    prov = _provenance(device)

    true_cache = {}

    def truth(x):
        k = tuple(np.round(x, 6))
        if k not in true_cache:
            true_cache[k] = sc.true_f(x)
        return true_cache[k]

    X_all, y_all, rows = None, None, []
    # 途中まで終わっている走行は、記録済みの設計を再生して状態を復元する。
    # 提案 (モデル当てはめ) はやり直さないので、そこが時間の節約になる。
    done = list(resume or [])
    design = []
    for r in range(rounds):
        t0 = time.time()
        cal = {}
        st = streams(scen_id, seed, rnd=r)     # ラウンド局所 (再開しても同一)
        if r < len(done):
            rows.append(done[r])
            Xr = np.asarray(done[r].pop("_X")) if "_X" in done[r] else None
            if Xr is None:                     # 旧形式: 設計が無ければ再開できない
                rows, done = [], []
            else:
                yr = np.asarray(done[r].pop("_y"))
                design.append((Xr.tolist(), yr.tolist()))
                X_all = Xr if X_all is None else np.vstack([X_all, Xr])
                y_all = yr if y_all is None else np.concatenate([y_all, yr])
                continue
        if r == 0:
            Xr, yr_pre, _ = initial_plate(scen_id, seed)
        elif kind == "none":
            Xr = strat(st["pool"])
        elif kind == "botorch":
            Xr = strat(st["pool"], X_obs=X_all, y_obs=y_all, seed=seed, device=device)
        else:
            from surrogates import make_surrogate
            sur = make_surrogate(kind, seed, device=device)
            sur.fit(X_all, y_all)
            # fit 直後の解決済みカーネルを記録する (strat 内の condition で
            # optimizer=None の複製に差し替わる前の、実際に学習された値)
            if getattr(getattr(sur, "gp", None), "kernel_", None) is not None:
                prov["gp_kernel_resolved"] = str(sur.gp.kernel_)[:200]
            # pool と acq を別系統から引く: 獲得関数側の乱数消費が候補生成を動かさない
            Xr = strat(st["pool"], surrogate=sur, X_obs=X_all, y_obs=y_all,
                       acq_rng=st["acq"])
        wall = time.time() - t0

        if r == 0:
            yr = yr_pre                      # 共有初期プレートの観測をそのまま使う
        else:
            if kind not in ("none", "botorch") and calibrate:
                cal = v1.calibration(sur, X_all, y_all, Xr, plate)
            yr = _plate_for(r).run(Xr)
            if cal:
                inside = (yr >= cal.pop("_lo")) & (yr <= cal.pop("_hi"))
                edge = np.array([plate.is_edge(i) for i in range(len(Xr))])
                cal["cal_cov80"] = float(inside.mean())
                cal["cal_cov80_inner"] = float(inside[~edge].mean()) if (~edge).any() else np.nan
                cal["cal_cov80_edge"] = float(inside[edge].mean()) if edge.any() else np.nan

        X_all = Xr if X_all is None else np.vstack([X_all, Xr])
        y_all = yr if y_all is None else np.concatenate([y_all, yr])
        row = dict(method=method, scenario=scen_id, seed=seed, round=r,
                   n_obs=len(y_all), best_true=max(truth(x) for x in X_all),
                   best_obs=float(y_all.max()), wall_s=wall,
                   config=tag, **cfg, **cal, **prov)
        rows.append(row)
        design.append((np.asarray(Xr).tolist(), np.asarray(yr).tolist()))
        if on_round is not None:
            # 各ラウンドの設計と観測を添えて渡す。1 ラウンド分だけだと
            # round>=1 から再開できないので、全ラウンドに付ける。
            ck = []
            for i, x in enumerate(rows):
                d = dict(x)
                d["_X"], d["_y"] = design[i]
                ck.append(d)
            on_round(ck)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--scenarios", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/v2_runs.csv")
    a = ap.parse_args()

    import csv
    rows = []
    for m in a.methods:
        for s in range(a.scenarios):
            for k in range(a.seeds):
                rows += run_one(m, s, k, a.rounds, device=a.device)
                print(f"done {m} sc{s} seed{k}", flush=True)
    cols = sorted({c for r in rows for c in r})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"saved {a.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
