"""Figures 1-3 of the manuscript, drawn from the summary tables in analysis/tables/.

The files follow the PLOS figure specifications: TIFF with LZW compression, RGB,
at most 7.5 in wide, Arial at 8-12 pt.  A PNG of each figure is written too.
Run reanalyze.py and grid_table.py first.

    uv run python analysis/make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/tables"
OUT = ROOT / "analysis/figures"
DPI = 400
WIDTH = 7.5   # PLOS の最大幅 (in)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})
COLORS = {"tabpfn_kb": "#0068A8", "gp_bo_kb": "#C65A16", "gp_qlognei": "#A8326E",
          "tabpfn_lp": "#009E73", "lhs": "#AAAAAA"}
LABELS = {"tabpfn_kb": "TabPFN KB", "gp_bo_kb": "DS GP KB", "gp_qlognei": "GP qLogNEI",
          "tabpfn_lp": "TabPFN LP", "lhs": "LHS"}


def save(fig, name):
    """PNG (for the review PDF) and LZW-compressed RGB TIFF (for submission)."""
    png = OUT / f"{name}.png"
    fig.savefig(png, dpi=DPI, facecolor="white")
    plt.close(fig)
    # matplotlib は RGBA で書くので、PLOS の要件 (RGB 8 bit) に合わせて変換する
    Image.open(png).convert("RGB").save(OUT / f"{name}.tif", compression="tiff_lzw",
                                        dpi=(DPI, DPI))
    w, h = Image.open(png).size
    print(f"{name}: {w}x{h}px = {w / DPI:.2f}x{h / DPI:.2f} in at {DPI} dpi")


def panel(ax, letter):
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=11)


def fig1():
    """Overview: (A) campaign, (B) growth model, (C) methods, (D) the three experiments.

    Everything is drawn in inch coordinates on one full-figure axes so that the
    layout can be read off the numbers.  Blue and orange are reserved for TabPFN
    and the Gaussian process, as in Figs 2 and 3, so the plates in (A) are gray.
    """
    H = 6.8
    ink, muted, faint, rule = "#1a1a1a", "#555555", "#bdbdbd", "#d9d9d9"
    fig = plt.figure(figsize=(WIDTH, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH); ax.set_ylim(0, H); ax.axis("off")

    def text(x, y, s, size=8, **kw):
        kw.setdefault("color", ink)
        ax.text(x, y, s, fontsize=size, **kw)

    def title(x, y, letter, s):
        text(x, y, letter, 11, fontweight="bold", va="center")
        text(x + 0.22, y, s, 9, fontweight="bold", va="center")

    def arrow(p, q, color=muted, lw=1.0, ms=8):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms,
                                     linewidth=lw, color=color, shrinkA=0, shrinkB=0))

    def box(x, y, w, h, face="white", edge=ink, lw=0.8, r=0.04):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    linewidth=lw, edgecolor=edge, facecolor=face))

    def square(x, y, size, essential):
        ax.add_patch(plt.Rectangle((x, y), size, size, linewidth=0.7, edgecolor=ink,
                                   facecolor=ink if essential else "white"))

    def bar(x, y, length, frac_ess, h=0.09):
        """A row of components: the essential share filled, the rest open."""
        ax.add_patch(plt.Rectangle((x, y), length * frac_ess, h, color=ink, lw=0))
        ax.add_patch(plt.Rectangle((x + length * frac_ess, y), length * (1 - frac_ess), h,
                                   facecolor="white", edgecolor=ink, lw=0.6))

    # --- A: 5 枚のプレートとその間の培養 ---------------------------------------
    title(0.08, 6.64, "A", "A simulated campaign: five plates, four opportunities to learn")
    pw, ph, py = 0.95, 0.60, 5.40
    xs = [0.32 + i * 1.47 for i in range(5)]
    for i, x in enumerate(xs):
        box(x, py, pw, ph, face="#efefef" if i == 0 else "white", r=0.05)
        gx = np.linspace(x + 0.09, x + pw - 0.09, 12)
        gy = np.linspace(py + 0.08, py + ph - 0.08, 8)
        X, Y = np.meshgrid(gx, gy)
        ax.scatter(X.ravel(), Y.ravel(), s=5, color=faint if i == 0 else muted, linewidths=0)
        text(x + pw / 2, py + ph + 0.07, f"Round {i}", 8.5, ha="center", va="bottom",
             fontweight="bold")
        text(x + pw / 2, py - 0.06, "96 conditions\n(Latin hypercube)" if i == 0
             else "80 conditions\nin 96 wells", 8, ha="center", va="top", color=muted,
             linespacing=1.1)
    for a, b in zip(xs, xs[1:]):
        arrow((a + pw + 0.05, py + ph / 2), (b - 0.05, py + ph / 2))
        text((a + pw + b) / 2, py + ph / 2 - 0.05, "culture\ncycle", 8, ha="center", va="top",
             color=muted, linespacing=1.05)
    # 適応プレートの上に、測定前に全条件を決めることを示す括弧
    by = py + ph + 0.30
    ax.plot([xs[1], xs[1], xs[4] + pw, xs[4] + pw], [by - 0.05, by, by, by - 0.05],
            color=muted, lw=0.8)
    text((xs[1] + xs[4] + pw) / 2, by + 0.04,
         "on each plate, all conditions are chosen before any of them is measured",
         8, ha="center", va="bottom", color=muted)
    ax.plot([0.08, WIDTH - 0.08], [4.98, 4.98], color=rule, lw=0.6)

    # --- B: 成長モデル --------------------------------------------------------
    title(0.08, 4.80, "B", "Growth model")
    ess = [True, False, True, True, False, False, True, False]
    sq, sx, top = 0.13, 0.30, 4.30
    ys = [top - i * 0.19 for i in range(len(ess))]
    text(sx, top + sq + 0.08, "Components", 8, va="bottom")
    minc, meanc, r = (1.05, 3.95), (1.05, 3.30), 0.17
    for y, e in zip(ys, ess):
        square(sx, y, sq, e)
        tgt = minc if e else meanc
        ax.plot([sx + sq, tgt[0] - r], [y + sq / 2, tgt[1]], color=ink if e else faint,
                lw=0.7, zorder=0)
    for (cx, cy), lab in ((minc, "min"), (meanc, "mean")):
        ax.add_patch(plt.Circle((cx, cy), r, facecolor="white", edgecolor=ink, lw=0.8))
        text(cx, cy, lab, 8, ha="center", va="center")
    text(minc[0], minc[1] + r + 0.04, "Liebig's law", 8, ha="center", va="bottom", color=muted)
    text(meanc[0], meanc[1] - r - 0.04, "weighted", 8, ha="center", va="top", color=muted)
    gw, gh = 0.62, 0.38
    gx0, gy0 = 1.50, 3.45
    box(gx0, gy0, gw, gh)
    text(gx0 + gw / 2, gy0 + gh / 2, "growth\nrate", 8, ha="center", va="center",
         linespacing=1.05)
    arrow((minc[0] + r, minc[1]), (gx0, gy0 + gh * 0.72))
    arrow((meanc[0] + r, meanc[1]), (gx0, gy0 + gh * 0.28))
    bx0, by0 = 1.50, 2.85
    box(bx0, by0, gw, gh)
    text(bx0 + gw / 2, by0 + gh / 2, "biomass\nat 24 h", 8, ha="center", va="center",
         linespacing=1.05)
    arrow((gx0 + gw / 2, gy0), (bx0 + gw / 2, by0 + gh))
    # 凡例: 塗りつぶしが必須成分
    square(0.30, 2.56, sq, True); text(0.49, 2.625, "essential (probability 0.35)", 8, va="center")
    square(2.15, 2.56, sq, False); text(2.34, 2.625, "non-essential", 8, va="center")
    # 挿入図: 必須成分が 2 つのとき、律速成分が入れ替わる点で傾きが急に変わる
    ins = fig.add_axes([2.62 / WIDTH, 3.00 / H, 1.00 / WIDTH, 1.10 / H])
    S = np.linspace(0, 10, 400)
    h1 = S / (1 + S + S ** 2 / 40)
    g = np.minimum(h1, 0.55)
    ins.plot(S, h1, color=faint, lw=1.0, ls="--")
    ins.axhline(0.55, color=faint, lw=1.0, ls="--")
    ins.plot(S, g, color=ink, lw=1.6)
    k = np.argmax(h1 >= 0.55)
    ins.plot(S[k], g[k], "o", ms=5, mfc="white", mec=ink, mew=1.0)
    ins.set_xticks([]); ins.set_yticks([])
    ins.set_xlabel("component 1", fontsize=8, labelpad=2)
    ins.set_ylabel("growth rate", fontsize=8, labelpad=2)
    ins.set_ylim(0, 0.8); ins.set_xlim(0, 10)
    ins.annotate("limiting\ncomponent\nswitches", xy=(S[k], g[k]), xytext=(4.6, 0.10),
                 fontsize=8, color=ink, linespacing=1.05,
                 arrowprops=dict(arrowstyle="-", color=muted, lw=0.7))
    text(2.45, 4.22, "Growth follows the scarcest\nessential component", 8, va="bottom",
         linespacing=1.05)

    # --- C: 比較した手法 --------------------------------------------------------
    title(3.86, 4.80, "C", "Methods compared")
    cx0, hw, cw = 3.86, 1.16, 0.80
    cols = ["Within-plate\nupdating (KB)", "Spreading\n(LP)", "Joint\nselection"]
    colx = [cx0 + hw + i * cw for i in range(3)]
    for x, c in zip(colx, cols):
        text(x + cw / 2, 4.36, c, 8, ha="center", va="bottom", linespacing=1.05)
    rows = [("TabPFN", "pretrained,\nnot retrained", COLORS["tabpfn_kb"],
             ["TabPFN KB", "TabPFN LP", "—"]),
            ("Gaussian process", "assumes a\nsmooth response", COLORS["gp_bo_kb"],
             ["DS GP KB\nMatérn GP KB", "Matérn GP LP", "GP qLogNEI"])]
    rh, ry = 0.52, [3.76, 3.20]
    for (name, sub, col, cells), y in zip(rows, ry):
        ax.add_patch(plt.Rectangle((cx0, y + rh - 0.13), 0.09, 0.09, color=col, lw=0))
        text(cx0 + 0.14, y + rh - 0.085, name, 8, va="center", fontweight="bold")
        text(cx0 + 0.14, y + rh - 0.17, sub, 8, va="top", color=muted, linespacing=1.05)
        for x, c in zip(colx, cells):
            ax.add_patch(plt.Rectangle((x, y), cw, rh, facecolor="white", edgecolor=rule, lw=0.6))
            text(x + cw / 2, y + rh / 2, c, 8, ha="center", va="center", linespacing=1.05,
                 color=muted if c == "—" else ink)
    # 中心の比較 (予測モデルだけが違う対) を太枠で囲む
    ax.add_patch(FancyBboxPatch((colx[0] + 0.02, ry[1] + 0.02), cw - 0.04,
                                ry[0] + rh - ry[1] - 0.04,
                                boxstyle="round,pad=0,rounding_size=0.04", linewidth=1.4,
                                edgecolor=ink, facecolor="none"))
    text(cx0, 3.04, "Bold outline: the central comparison, in which only the model differs.",
         8, va="top")
    text(cx0, 2.84, "Structural assumptions: Fixed TR TS (searches near the best medium),",
         8, va="top", color=muted)
    text(cx0, 2.69, "SAAS qLogNEI (few components matter). Control: LHS (no learning).",
         8, va="top", color=muted)
    ax.plot([0.08, WIDTH - 0.08], [2.42, 2.42], color=rule, lw=0.6)

    # --- D: 3 つのシミュレーション実験 -------------------------------------------
    title(0.08, 2.24, "D", "Three simulation experiments")
    x0s = [0.10, 2.58, 5.06]
    heads = [("Main evaluation", "Does TabPFN find better media?"),
             ("Embedding controls", "Is it the number of variables?"),
             ("Factorial simulation", "Which property of the growth model?")]
    answers = ["Better media with 80 and\n120 components (Fig 2A, Table 2)",
               "No: the advantage disappears\n(Fig 2B, Table 4)",
               "The proportion of essential\ncomponents (Table 5)"]
    for x, (h, q), a in zip(x0s, heads, answers):
        text(x, 1.98, h, 8.5, fontweight="bold", va="center")
        text(x, 1.82, q, 8, va="center", style="italic")
        text(x, 0.44, "→ " + a, 8, va="top", linespacing=1.1)
    # D1: 成分数を増やすと、変数の数と必須成分の数が一緒に増える
    x, full = x0s[0], 1.85
    for i, d in enumerate((8, 20, 40, 80, 120)):
        y = 1.55 - i * 0.15
        bar(x + 0.32, y, full * d / 120, 0.35)
        text(x + 0.27, y + 0.045, str(d), 8, ha="right", va="center")
    text(x + 0.32, 0.78, "variables and essential components\nincrease together", 8,
         va="center", color=muted, linespacing=1.05)
    # D2: 8 成分のモデルを 120 変数として見せる
    x, full2 = x0s[1], 1.65
    bx = x + 0.68
    for name, y in (("Padded", 1.50), ("Mixed", 1.26), ("120 comp.", 1.02)):
        text(bx - 0.06, y + 0.045, name, 8, ha="right", va="center")
    w8 = full2 * 8 / 120
    bar(bx, 1.50, w8, 0.35)
    ax.add_patch(plt.Rectangle((bx + w8, 1.50), full2 - w8, 0.09, facecolor="#efefef",
                               edgecolor=faint, lw=0.6))
    ax.add_patch(plt.Rectangle((bx, 1.26), full2, 0.09, facecolor="#efefef", edgecolor=faint,
                               lw=0.6, hatch="////"))
    bar(bx, 1.02, full2, 0.35)
    text(bx, 0.82, "all three have 120 variables", 8, va="center", color=muted)
    # D3: 3x3 の格子。セルの濃さは表 5 の平均差 (TabPFN KB − DS GP KB)
    grid = pd.read_csv(DATA / "grid_contrasts.csv")
    x = x0s[2]
    cwid, chgt, gx, gy = 0.34, 0.24, x + 0.86, 0.66
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list(
        "advantage", ["#f2f7fb", COLORS["tabpfn_kb"]])
    for i, e in enumerate((0.05, 0.35, 0.70)):
        for j, w in enumerate((0.1, 0.7, 5.0)):
            v = float(grid[(grid.ess_p == e) & (grid.w_conc == w)]["mean"].iloc[0])
            ax.add_patch(plt.Rectangle((gx + j * cwid, gy + i * chgt), cwid, chgt,
                                       facecolor=ramp(max(v, 0) / 0.40), edgecolor="white", lw=1.0))
            text(gx + j * cwid + cwid / 2, gy + i * chgt + chgt / 2,
                 f"{v:+.2f}".replace("-", "−"), 8, ha="center", va="center",
                 color="white" if v > 0.2 else ink)
        text(gx - 0.05, gy + i * chgt + chgt / 2, f"{e:.2f}", 8, ha="right", va="center")
    for j, w in enumerate((0.1, 0.7, 5.0)):
        text(gx + j * cwid + cwid / 2, gy + 3 * chgt + 0.03, f"{w:g}", 8, ha="center",
             va="bottom")
    text(gx + 1.5 * cwid, gy + 3 * chgt + 0.19, "weight concentration α", 8, ha="center",
         va="bottom")
    text(gx - 0.42, gy + 1.5 * chgt, "essential prob.", 8, ha="center", va="center",
         rotation=90)
    save(fig, "Fig1")


def fig2():
    """(A) gain by number of components; (B) 120 variables with and without 120 components."""
    raw = pd.read_csv(DATA / "raw_sensitivity.csv")
    emb = pd.read_csv(DATA / "embedding_contrasts.csv")
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.9))
    for ref, shift in (("gp_bo_kb", 0.0), ("gp_qlognei", 0.10)):
        q = raw[(raw.reference == ref) & (raw.metric == "relative biomass")].sort_values("d")
        x = np.arange(len(q)) + shift
        ax[0].errorbar(x, 100 * q["mean"],
                       yerr=100 * np.array([q["mean"] - q.ci_lo, q.ci_hi - q["mean"]]),
                       marker="o", ms=4, capsize=3, label="vs " + LABELS[ref], color=COLORS[ref])
    ax[0].axhline(0, color="black", lw=0.7)
    ax[0].set_xticks(np.arange(5), [8, 20, 40, 80, 120])
    ax[0].set_xlabel("Number of components")
    ax[0].set_ylabel("Gain of TabPFN KB in best biomass (%)")
    ax[0].legend(frameon=False, loc="upper left")
    panel(ax[0], "A")

    e = emb[emb.d == 120].set_index("embedding").loc[["ambient", "rotate", "intrinsic"]]
    ax[1].errorbar(range(3), e["mean"], yerr=np.array([e["mean"] - e.ci_lo, e.ci_hi - e["mean"]]),
                   fmt="o", ms=4, color=COLORS["tabpfn_kb"], capsize=4)
    ax[1].axhline(0, color="black", lw=0.7)
    ax[1].set_xticks(range(3), ["8 components,\npadded", "8 components,\nmixed",
                                "120 components"])
    ax[1].set_xlim(-0.5, 2.5)
    ax[1].set_xlabel("Growth model presented as 120 variables")
    ax[1].set_ylabel("TabPFN KB − DS GP KB\n(normalized score)")
    panel(ax[1], "B")
    fig.tight_layout()
    save(fig, "Fig2")


def fig3():
    """Learning curves over the five plates at 40, 80 and 120 components."""
    c = pd.read_csv(DATA / "curves.csv")
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 3.0), sharey=True)
    for ax, dim, letter in zip(axs, (40, 80, 120), "ABC"):
        for m in COLORS:
            q = c[(c.d == dim) & (c.method == m)].sort_values("round")
            ax.plot(q["round"] + 1, q["mean"], marker="o", ms=3, lw=1.3, color=COLORS[m],
                    label=LABELS[m])
            ax.fill_between(q["round"] + 1, q["mean"] - q.se, q["mean"] + q.se,
                            color=COLORS[m], alpha=0.12, linewidth=0)
        panel(ax, letter)
        ax.set_title(f"{dim} components", loc="center")
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("Completed plates")
    axs[0].set_ylabel("Normalized best biomass")
    # どのパネルにも曲線が通らない空きがないので、凡例は軸の外に 1 行で置く
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "Fig3")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig1(); fig2(); fig3()
