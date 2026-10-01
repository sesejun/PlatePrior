"""Run the whole rapid-plate benchmark on Modal.

One Modal container input = one benchmark run = (method, scenario, seed) under one
configuration.  TabPFN arms are routed to a GPU function, everything else to CPU.
Every finished run is written to a Modal Volume as its own JSON file, so a rerun
skips completed work and a crash costs at most the runs in flight.

    pip install modal && modal setup

    modal run modal_app.py                      # A, B, C (Phase D は含まない)
    modal run modal_app.py --phase A            # GP arms only
    modal run modal_app.py --phase B            # TabPFN arms only
    modal run modal_app.py --phase C1           # full-triplicate plate
    modal run modal_app.py --phase A,B --out results/modal_runs.csv
    modal run modal_app.py --phase B --dry-run  # print the job count and exit

Phase D (TabPFN 世代比較) は v2.5 以降の重みを使うため、Prior Labs で
ライセンスに同意して API Key を取得し、Modal Secret に登録したうえで
世代を焼き込んだイメージが要る:

    modal secret create tabpfn-token TABPFN_TOKEN=<your-key>
    TABPFN_GENS=v2,v2.5,v3 modal run modal_app.py --phase D

Cost control: MAX_CONTAINERS (default 30) bounds CPU fan-out and
MAX_GPU_CONTAINERS (default 10) bounds GPU fan-out -- the GPU cap is a plan
limit, so raising MAX_CONTAINERS alone will not widen it.  GPU_TYPE selects the
accelerator.  Both are read from the environment when the app is built:

    MAX_CONTAINERS=100 GPU_TYPE=L4 modal run modal_app.py
"""
import json
import os
import pathlib

import modal

HERE = pathlib.Path(__file__).parent
APP_NAME = "rapid-plate"
GPU_TYPE = os.environ.get("GPU_TYPE", "A10G")
MAX_CONTAINERS = int(os.environ.get("MAX_CONTAINERS", "30"))
# GPU の同時実行数はプラン上限で別に決まる (この環境では 10)。CPU と同じ
# 値を使うと上限超過でキューが詰まるため、GPU 側だけ独立に制限する。
MAX_GPU_CONTAINERS = int(os.environ.get("MAX_GPU_CONTAINERS", "10"))
MODEL_DIR = "/root/tabpfn_models"
DATA_DIR = "/data"

# TabPFN generations baked into the image so no run needs HuggingFace at runtime.
#
# v2.5 以降は非商用ライセンスへの同意と TABPFN_TOKEN が必要で、トークンなしの
# ビルドでは取得できない (Phase D 参照)。既定では v2 のみを焼き込む。
# トークンを Modal Secret に登録したら TABPFN_GENS で世代を足せる:
#   TABPFN_GENS=v2,v2.5,v3 modal run modal_app.py --phase D
TABPFN_VERSIONS = [v.strip() for v in
                   os.environ.get("TABPFN_GENS", "v2").split(",") if v.strip()]


def _tabpfn_secrets():
    """TABPFN_TOKEN を載せた Secret。v2 のみなら不要なので空で返す。"""
    if TABPFN_VERSIONS == ["v2"]:
        return []
    name = os.environ.get("TABPFN_SECRET", "tabpfn-token")
    return [modal.Secret.from_name(name, required_keys=["TABPFN_TOKEN"])]


def _prefetch_tabpfn():
    """Download every TabPFN regressor checkpoint into the image layer.

    A generation the installed package cannot resolve is skipped rather than
    failing the build: the version list is forward-looking, and only the
    generations actually named by a phase need to be present.  The resolved
    filenames are written to versions.json so the entrypoint can verify the
    generations a run requests before spending GPU time.
    """
    import numpy as np
    from tabpfn import TabPFNRegressor
    from tabpfn.model_loading import ModelType, ModelVersion, _get_model_source

    pathlib.Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    names = {}
    X = np.random.default_rng(0).random((16, 3))
    y = np.random.default_rng(0).random(16)
    for v in TABPFN_VERSIONS:
        try:
            fn = _get_model_source(ModelVersion(v), ModelType("regressor")).default_filename
            TabPFNRegressor(device="cpu", n_estimators=1, model_path=fn).fit(X, y)
        except Exception as e:                      # unknown generation, or download failed
            print(f"SKIP {v}: {type(e).__name__}: {e}", flush=True)
            continue
        names[v] = fn
        print("prefetched", v, fn, flush=True)
    if not names:
        raise RuntimeError("no TabPFN checkpoint could be fetched")
    pathlib.Path(MODEL_DIR, "versions.json").write_text(json.dumps(names))
    print("available generations:", sorted(names), flush=True)


image = (
    modal.Image.debian_slim(python_version="3.12")
    # ローカル (pyproject.toml) は TabPFN 2.x 固定だが、Phase D の世代比較には
    # v2.5 以降のチェックポイントを解決できる新しい tabpfn が必要なため、
    # Modal イメージ側は世代を跨げる 9.x を使う。
    #
    # 上限を切らずに "tabpfn" とすると、イメージを再ビルドした日によって
    # 入る世代が変わり Phase D の比較が揺れるため、明示的にピンする。
    # (パッケージ 6.x=v2.5 / 7.x=v2.6 / 8.x=v3 / 9.x=v3.5 と対応)
    # なお全アームのデフォルトは surrogates.py 側で v2 チェックポイントに
    # 固定されているので、Phase A/B の基準はパッケージ版数に依存しない。
    .pip_install(
        "numpy", "scipy", "pandas", "scikit-learn", "matplotlib",
        "torch>=2.5", "tabpfn==9.0.0",
        # SAASBO は fully-Bayesian extra (JAX/jaxlib/NumPyro) を要求する。
        # 素の botorch では SaasFullyBayesianSingleTaskGP が ImportError になる。
        "botorch[fully_bayesian]",
    )
    .env({
        "TABPFN_MODEL_CACHE_DIR": MODEL_DIR,
        "HF_HOME": "/root/hf",
        "SKRUB_DATA_DIRECTORY": "/root/skrub",
        "MPLCONFIGDIR": "/root/mpl",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    .add_local_dir(
        HERE, "/root/bench", copy=True,
        # .DS_Store は Finder が随時書き換えるため、含めるとビルドが
        # "modified during build process" で落ちる。paper/ と output/ は
        # 実行に不要なので除外する。
        ignore=["results/*", "logs/*", "__pycache__/*", "*.pyc", "cache/*",
                "**/.DS_Store", ".git/*", "paper/*", "output/*", ".venv/*"],
    )
    # TABPFN_GENS で v2.5 以降を要求したときだけ Secret を要求する。
    # 既定 (v2 のみ) では Secret 未作成の環境でもビルドできる。
    .run_function(_prefetch_tabpfn, secrets=_tabpfn_secrets())
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(f"{APP_NAME}-results", create_if_missing=True)


# --------------------------------------------------------------------------
# experiment grid — mirrors run_main.sh / run_ablation.sh
# --------------------------------------------------------------------------
GP_ARMS = ["random", "lhs", "gp_kb", "gp_lp", "gp_ts", "gp_qei", "gp_qei_indep", "gp_qlognei"]
TABPFN_ARMS = ["tabpfn_kb", "tabpfn_lp", "tabpfn_ts", "tabpfn_qei"]
CORE = ["random", "lhs", "gp_kb", "gp_lp", "gp_ts", "tabpfn_kb", "tabpfn_lp", "tabpfn_ts"]
SMALL = ["random", "lhs", "gp_kb", "gp_lp", "tabpfn_kb", "tabpfn_lp"]
CHEAP = ["lhs", "gp_kb", "gp_lp"]

# Phase D のチェックポイント名。イメージ内の tabpfn が解決した実名は
# versions.json に書き出してあるが、ジョブ表はローカル (tabpfn 未インストール)
# で組み立てるため、ここは既知の名前を定数として持つ。
# 実行前に _verify_generations がイメージ側と突き合わせる。
V25 = "tabpfn-v2.5-regressor-v2.5_default.ckpt"
V3 = "tabpfn-v3-regressor-v3_default.ckpt"


def _block(methods, cfg=None, scenarios=10, seeds=5, rounds=5, exp_id="v1",
           scenario_offset=0):
    """scenario_offset は未使用シナリオで追試するための起点 (設計書 §2.3)。
    既存 10 シナリオで手法や閾値を選び直さないための仕組み。"""
    return dict(methods=methods, cfg=cfg or {}, scenarios=scenarios, seeds=seeds,
                rounds=rounds, exp_id=exp_id, scenario_offset=scenario_offset)


PHASES = {
    # Phase A/B: the main comparison, 10 scenarios x 5 seeds x 5 plates
    "A": [_block(GP_ARMS)],
    "B": [_block(TABPFN_ARMS)],
    # C1: every proposed condition in triplicate -> 32 conditions per 96-well plate
    "C1": [_block(CORE, {"PBO_K_TOP": "32"})],
    # C2: replicate-allocation sweep at two noise levels
    "C2": (
        [_block(CHEAP, {"PBO_K_TOP": k}, seeds=3) for k in ("0", "4", "16")]
        + [_block(CHEAP, {"PBO_K_TOP": k, "PBO_CV": "0.15"}, seeds=3)
           for k in ("0", "4", "8", "16", "32")]
        + [_block(["tabpfn_kb", "tabpfn_lp"], {"PBO_K_TOP": k, "PBO_CV": "0.15"}, seeds=3)
           for k in ("0", "8", "32")]
    ),
    # C3: design dimension
    "C3": [_block(SMALL, {"PBO_N_SUB": "20"}, seeds=3),
           _block(SMALL, {"PBO_N_SUB": "40"}, seeds=3)],
    # C4: plate width (replicate allocation scaled proportionally)
    "C4": [_block(CORE, {"PBO_N_WELLS": "48", "PBO_K_TOP": "4"}, seeds=3),
           _block(CORE, {"PBO_N_WELLS": "24", "PBO_K_TOP": "2"}, seeds=3)],
    # P1A: 本番 (設計書 §2.3)。未使用 scenario 20 (id 20-39) x seed 5 x
    # d{8,20,40} x 8 アーム。既存 10 scenario は使わない。
    # 主比較は d=40 の tabpfn_kb vs gp_bo_kb (走行前に固定済み)。
    # SAASBO は費用が大きいため d=40 のみ (設計書 §2.1)。
    "P1A": [_block(["tabpfn_kb", "tabpfn_lp", "gp_kb", "gp_lp",
                    "gp_bo_kb", "gp_qlognei", "turbo", "gitbo_tabpfn", "lhs"],
                   {"PBO_N_SUB": d}, scenarios=20, seeds=5,
                   exp_id="v2a", scenario_offset=20)
            for d in ("8", "20", "40")]
           + [_block(["saasbo"], {"PBO_N_SUB": "40"}, scenarios=20, seeds=5,
                     exp_id="v2a", scenario_offset=20)],
    # S: 新アーム (TuRBO / SAASBO / GIT-BO) の動作確認。2シナリオ x 1シード。
    # 設計書 §2.5「新規3アームの費用は外挿せず少数走行で実測する」に対応。
    "S": [_block(["turbo", "gitbo_tabpfn"], {"PBO_N_SUB": "40"},
                 scenarios=2, seeds=1, exp_id="v2a", scenario_offset=15),
          _block(["saasbo"], {"PBO_N_SUB": "40"},
                 scenarios=2, seeds=1, exp_id="v2a", scenario_offset=15)],
    # SM: SAASBO を高次元で 1 走行ずつ計測する。d=40 では 1 走行 64 分
    # だったが、NUTS の所要時間が次元に対してどう伸びるかは分からない。
    # d=40 のときも 2 ラウンドからの外挿が 3.8 倍外れたので、200 走行を
    # 流す前に実測する。
    "SM": [_block(["saasbo"], {"PBO_N_SUB": d},
                  scenarios=1, seeds=1, exp_id="v2a", scenario_offset=15)
           for d in ("80", "120")],
    # P2: 次元を伸ばした予備比較 (d=80, 120)。パイロット P が d=40 で
    # 「強い GP に対しても TabPFN 優位」を示したため、次元の効きを確かめる。
    "P2": [_block(["gp_kb", "gp_bo_kb", "tabpfn_kb", "gp_lp", "gp_bo_lp", "tabpfn_lp"],
                  {"PBO_N_SUB": d}, scenarios=5, seeds=3,
                  exp_id="v2a", scenario_offset=10)
           for d in ("80", "120")],
    # G1: 格子実験 (P1-B 追試)。「基質数が多いと TabPFN 有利」の機序を
    # 必須基質率 x 重み集中度 の 3x3 で切り分ける。d=120 固定。
    # 既存データの相関分析では必須基質数 r=+0.52 (p=0.018)、
    # 重み分散 r=+0.36 (p=0.121) と両方が候補で、かつ 2 要因は
    # ほぼ無相関 (r=-0.19) だったため交差させる必要がある。
    "G1": [_block(["tabpfn_kb", "tabpfn_lp", "gp_kb", "gp_bo_kb",
                   "gp_qlognei", "lhs"],
                  {"PBO_N_SUB": "120", "PBO_ESS_P": e, "PBO_W_CONC": w},
                  scenarios=20, seeds=3, exp_id="v2a", scenario_offset=20)
           for e in ("0.05", "0.35", "0.7") for w in ("0.1", "0.7", "5.0")],
    # B0: B1 の動作確認。3群 x d=40 を 2 scenario だけ回し、
    # 埋め込みが効いているか (config tag が分かれるか) を先に確かめる。
    "B0": [_block(["gp_kb", "tabpfn_kb"],
                  {"PBO_N_SUB": "40", "PBO_EMBED": e, "PBO_EMBED_DIM": "40"},
                  scenarios=2, seeds=1, exp_id="v2a", scenario_offset=20)
           for e in ("ambient", "rotate", "intrinsic")],
    # B1: P1-B (設計書 §3)。次元そのものの効果と、目的関数が難しくなる効果を
    # 分ける。ambient / rotate は intrinsic 次元を 8 に固定したまま探索空間の
    # 次元だけを上げる。intrinsic は従来どおり基質数自体を増やす。
    # 3 群 x d{40,120} x 20 scenario x 3 seed x 6 アーム。
    "B1": [_block(["tabpfn_kb", "tabpfn_lp", "gp_kb", "gp_bo_kb",
                   "gp_qlognei", "lhs"],
                  {"PBO_N_SUB": d, "PBO_EMBED": e, "PBO_EMBED_DIM": d},
                  scenarios=20, seeds=3, exp_id="v2a", scenario_offset=20)
           for e in ("ambient", "rotate", "intrinsic") for d in ("40", "120")],
    # P3: 高次元 (d=80,120) を P1A と同じ規模・同じアームに揃える。
    # P2 は n=5 シナリオで、唯一 TabPFN 優位が見えた領域なのに検出力が
    # 足りなかった。scenario/seed/アームを P1A と一致させ、次元だけが
    # 違う比較にする。SAASBO は d=40 限定のまま (費用と時間のため)。
    "P3": [_block(["tabpfn_kb", "tabpfn_lp", "gp_kb", "gp_lp",
                   "gp_bo_kb", "gp_qlognei", "turbo", "gitbo_tabpfn", "lhs"],
                  {"PBO_N_SUB": d}, scenarios=20, seeds=5,
                  exp_id="v2a", scenario_offset=20)
           for d in ("80", "120")],
    # P: 強い GP の予備比較 (設計書 §2.1)。protocol v2、未使用シナリオ 10-14。
    # gp_bo_* は BoTorch SingleTaskGP (次元スケーリング prior)、gp_* は sklearn。
    # d=40 で「TabPFN の優位は強い GP に対しても残るか」を最初に確かめる。
    "P": [_block(["gp_kb", "gp_bo_kb", "tabpfn_kb", "gp_lp", "gp_bo_lp", "tabpfn_lp"],
                 {"PBO_N_SUB": "40"}, scenarios=5, seeds=3,
                 exp_id="v2a", scenario_offset=10)],
    # E: 計算コストの公平比較。gp_qlognei を TabPFN と同じ GPU 上で回し、
    # 「TabPFN が速い」のがサロゲートの差かハードウェアの差かを切り分ける。
    "E": [_block(["gp_qlognei_gpu", "tabpfn_lp", "tabpfn_kb"], seeds=5)],
    # D: TabPFN generations.  'all' には含まれない (ALL_PHASES 参照):
    # v2.5/v3 の重みはライセンス同意と TABPFN_TOKEN が要るため、
    # TABPFN_GENS で世代を焼き込んだイメージでのみ動く。
    "D": [_block(["tabpfn_kb", "tabpfn_lp"], {"PBO_TABPFN_MODEL": V25}, seeds=3),
          _block(["tabpfn_kb", "tabpfn_lp"], {"PBO_TABPFN_MODEL": V3}, seeds=3)],
}
# "all" が回す対象。Phase D (世代比較) は v2.5 以降の重みが必要で、
# ライセンス同意と TABPFN_TOKEN が揃った環境でのみ明示指定で回す:
#   modal run modal_app.py --phase D
ALL_PHASES = ["A", "B", "C1", "C2", "C3", "C4"]
OPT_IN_PHASES = ["D", "E", "P", "P2", "P3", "P1A", "B1", "B0", "G1", "S", "SM"]


def build_jobs(phases):
    jobs = []
    for ph in phases:
        for b in PHASES[ph]:
            for m in b["methods"]:
                off = b.get("scenario_offset", 0)
                for s in range(off, off + b["scenarios"]):
                    for k in range(b["seeds"]):
                        jobs.append(dict(phase=ph, method=m, scenario=s, seed=k,
                                         rounds=b["rounds"], cfg=b["cfg"],
                                         exp_id=b.get("exp_id", "v1")))
    return jobs


# --------------------------------------------------------------------------
# remote execution
# --------------------------------------------------------------------------
_ACTIVE_CFG = None


def _ensure_path():
    import sys
    if "/root/bench" not in sys.path:
        sys.path.insert(0, "/root/bench")


def _scale_one(s):
    """Module-level so multiprocessing can pickle it (containers fork, so the
    reloaded simulator configuration is inherited)."""
    import numpy as np
    from simulator import MonodScenario, sample_designs
    sc = MonodScenario(s)
    X = sample_designs(1000, np.random.default_rng(99))
    y = np.array([sc.true_f(x) for x in X])
    return str(s), [float(y.min()), float(y.max())]


def _apply_config(cfg):
    """Rebind the module-level ablation constants for this configuration.

    simulator/strategies read their knobs from the environment at import time, so a
    container that switches configuration must reload them in dependency order.
    Containers process inputs of one configuration at a time, so this fires at most
    once per container in practice.
    """
    global _ACTIVE_CFG
    import importlib
    import sys

    if sys.path[0] != "/root/bench":
        sys.path.insert(0, "/root/bench")
    keys = ["PBO_EMBED", "PBO_EMBED_DIM", "PBO_BASE_SUB", "PBO_ESS_P", "PBO_W_CONC",
            "PBO_N_SUB", "PBO_N_WELLS", "PBO_K_TOP", "PBO_N_REP", "PBO_CV",
            "PBO_TABPFN_MODEL", "PBO_N_GLOBAL", "PBO_N_LOCAL"]
    if cfg == _ACTIVE_CFG:
        return
    for k in keys:
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in cfg.items()})

    import embed                                                  # noqa: F401
    importlib.reload(embed)
    import simulator, surrogates, strategies                      # noqa: E401
    for mod in (simulator, surrogates, strategies):
        importlib.reload(mod)
    import surrogate_botorch                                      # noqa: F401
    importlib.reload(surrogate_botorch)
    import botorch_arm                                            # noqa: F401
    importlib.reload(botorch_arm)
    import highdim_arms                                           # TuRBO/SAASBO/GIT-BO
    importlib.reload(highdim_arms)
    import driver
    importlib.reload(driver)
    import driver_v2                                              # protocol v2
    importlib.reload(driver_v2)
    _ACTIVE_CFG = cfg


def _run(job, device="cpu"):
    _ensure_path()
    _apply_config(job["cfg"])
    import driver
    exp_id = job.get("exp_id", "v1")
    # v2 は RNG 系統を分離し初期プレートを共有するため、v1 とは run-for-run で
    # 比較できない。保存キーに exp_id を入れ、resume が混ざらないようにする。
    runner = driver if exp_id == "v1" else __import__("driver_v2")
    tag = driver.config_tag()
    key = f"{tag}__{job['method']}__sc{job['scenario']}__seed{job['seed']}"
    if exp_id != "v1":
        key = f"{exp_id}__{key}"
    out = pathlib.Path(DATA_DIR, "runs", f"{key}.json")
    if out.exists():                                   # resume: already finished
        return json.loads(out.read_text())

    # ラウンド単位のチェックポイント。Modal のプリエンプションは走行途中でも
    # 起きるため、完走時にしか保存しないと長いアーム (d=40 の SAASBO は
    # 1走行 ~21 分) が毎回ゼロからやり直しになり、いつまでも終わらない。
    part = pathlib.Path(DATA_DIR, "partial", f"{key}.json")
    part.parent.mkdir(parents=True, exist_ok=True)
    resume = None
    if part.exists():
        try:
            resume = json.loads(part.read_text())
        except Exception:                              # 壊れていたら捨てて最初から
            resume = None

    if exp_id == "v1":
        rows = runner.run_one(job["method"], job["scenario"], job["seed"],
                              job["rounds"], device=device)
    else:
        def _ck(rows_so_far):
            part.write_text(json.dumps(rows_so_far))
            vol.commit()

        rows = runner.run_one(job["method"], job["scenario"], job["seed"],
                              job["rounds"], device=device,
                              on_round=_ck, resume=resume)
    for r in rows:
        r["phase"] = job["phase"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
    if part.exists():
        part.unlink()                                  # 完走したら部分結果は不要
    vol.commit()
    return rows


@app.function(volumes={DATA_DIR: vol}, cpu=2.0, memory=4096, timeout=7200,
              max_containers=MAX_CONTAINERS, retries=1)
def run_cpu(job):
    return _run(job, device="cpu")


@app.function(volumes={DATA_DIR: vol}, gpu=GPU_TYPE, cpu=2.0, memory=8192, timeout=7200,
              max_containers=MAX_GPU_CONTAINERS, retries=1, secrets=_tabpfn_secrets())
def run_gpu(job):
    import torch
    return _run(job, device="cuda" if torch.cuda.is_available() else "cpu")


@app.function(timeout=600)
def available_generations():
    """Checkpoint filenames baked into the image, as {generation: filename}."""
    f = pathlib.Path(MODEL_DIR, "versions.json")
    return json.loads(f.read_text()) if f.exists() else {}


@app.function(volumes={DATA_DIR: vol}, cpu=8.0, timeout=3600)
def compute_scales(n_sub: int, scenarios: int = 10):
    """Normalisation scale per scenario for one design dimension."""
    _ensure_path()
    _apply_config({"PBO_N_SUB": str(n_sub)})
    from multiprocessing import get_context
    import simulator

    out = pathlib.Path(DATA_DIR, f"scales_d{simulator.N_SUB}.json")
    scales = json.loads(out.read_text()) if out.exists() else {}
    todo = [s for s in range(scenarios) if str(s) not in scales]
    if todo:
        with get_context("fork").Pool(8) as p:
            for k, v in p.map(_scale_one, todo):
                scales[k] = v
    out.write_text(json.dumps(scales, indent=1))
    vol.commit()
    return {simulator.N_SUB: scales}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
@app.local_entrypoint()
def main(phase: str = "all", out: str = "results/modal_runs.csv", dry_run: bool = False):
    """`out` は走らせたフェーズ名を挟んだ実ファイル名に展開される
    (results/modal_runs.csv + phase A -> results/modal_runs_A.csv)。
    フェーズごとに別ファイルへ書くことで、続けて別フェーズを回しても
    先に書いた結果を truncate しない。analysis.py は複数 CSV を
    まとめて読んで (method, config, scenario, seed, round) で
    重複排除するので、分かれていても集計はそのまま通る。"""
    import csv

    phases = ALL_PHASES if phase == "all" else [p.strip() for p in phase.split(",")]
    bad = [p for p in phases if p not in PHASES]
    if bad:
        raise SystemExit(f"unknown phase(s) {bad}; "
                         f"choose from {ALL_PHASES + OPT_IN_PHASES} or 'all' "
                         f"({OPT_IN_PHASES} は 'all' に含まれず明示指定が要る)")

    jobs = build_jobs(phases)

    # Phase D などが指すチェックポイントがイメージに無ければ、GPU を掴む前に落とす。
    wanted = {j["cfg"]["PBO_TABPFN_MODEL"] for j in jobs if "PBO_TABPFN_MODEL" in j["cfg"]}
    if wanted:
        have = set(available_generations.remote().values())
        missing = sorted(wanted - have)
        if missing:
            raise SystemExit(
                f"checkpoint(s) not in the image: {missing}\n"
                f"  available: {sorted(have)}\n"
                "  イメージの tabpfn ピンと TABPFN_VERSIONS を確認してください。")

    # GPU に載せるアーム: TabPFN 全般と、GPU 経路の BoTorch アーム。
    # 後者はハードウェアを揃えた計算コスト比較 (GPU 対 GPU) のためのもの。
    def _wants_gpu(m):
        # TabPFN を使うアームは GPU (gitbo_tabpfn は TabPFN サロゲート)。
        # GP 系は CPU に置き、sklearn GP と同じハードで比較する。
        return m.startswith("tabpfn") or m.endswith("_gpu") or "tabpfn" in m

    gpu_jobs = [j for j in jobs if _wants_gpu(j["method"])]
    cpu_jobs = [j for j in jobs if not _wants_gpu(j["method"])]
    dims = sorted({int(j["cfg"].get("PBO_N_SUB", 8)) for j in jobs})
    print(f"phases={phases}  runs={len(jobs)}  (cpu={len(cpu_jobs)}, gpu={len(gpu_jobs)})  "
          f"dimensions={dims}  cpu_containers={MAX_CONTAINERS}  "
          f"gpu_containers={MAX_GPU_CONTAINERS}  gpu={GPU_TYPE}")
    if dry_run:
        return

    # 正規化スケールは次元ごとに必要。analysis.py は results/scales_d*.json を
    # 読むので、Volume に置くだけでなくローカルにも書き出す。
    outdir = pathlib.Path(out).parent
    outdir.mkdir(parents=True, exist_ok=True)
    for d in dims:
        for n_sub, sc in compute_scales.remote(d, 10).items():
            f = outdir / f"scales_d{n_sub}.json"
            merged = json.loads(f.read_text()) if f.exists() else {}
            merged.update(sc)
            f.write_text(json.dumps(merged, indent=1))
            print(f"  scales -> {f} ({len(merged)} scenarios)")

    rows, failed = [], 0
    for label, fn, batch in (("cpu", run_cpu, cpu_jobs), ("gpu", run_gpu, gpu_jobs)):
        if not batch:
            continue
        for i, rr in enumerate(fn.map(batch, order_outputs=False, return_exceptions=True), 1):
            if isinstance(rr, Exception):
                failed += 1
                print(f"  [{label}] run failed: {type(rr).__name__}: {rr}")
                continue
            rows += rr
            if i % 25 == 0:
                print(f"  [{label}] {i}/{len(batch)}")

    if not rows:
        raise SystemExit("no rows produced")
    cols = sorted({k for r in rows for k in r})
    outp = pathlib.Path(out)
    tag = "all" if phases == ALL_PHASES else "-".join(phases)
    outp = outp.with_name(f"{outp.stem}_{tag}{outp.suffix}")
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {outp}: {len(rows)} rows from {len(jobs) - failed}/{len(jobs)} runs"
          + (f"  ({failed} failed)" if failed else ""))
    print("analyse with:  uv run python analysis/reanalyze.py   (reads results/v2a_runs_<phase>.csv)")
