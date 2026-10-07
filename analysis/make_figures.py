"""Figures 1-3 of the manuscript, drawn from the summary tables in analysis/tables/
and, for the scenario-level points of Fig 2, from the run-level results in results/.

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
    """Biological overview; panel B is explicitly a conceptual illustration."""
    H = 7.25
    fig = plt.figure(figsize=(WIDTH, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(0, H)); ax.axis("off")
    ink, muted = "#202C36", "#53616D"
    def text(x, y, label, size=9, **kw):
        ax.text(x, y, label, fontsize=size, color=ink, **kw)
    def title(y, letter, label):
        text(.18, y, letter, 12, weight="bold")
        text(.44, y, label, 10, weight="bold")
    def box(x, y, w, h, color="#F1F5F7"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=.07",
                                  facecolor=color,edgecolor="#C7D1D9",lw=.8))
    def arrow(start,end):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=11,
                                    color=muted,lw=1))
    def plate(x,y,w=.65,h=.38):
        box(x,y,w,h,"white")
        xx,yy=np.meshgrid(np.linspace(x+.06,x+w-.06,12),np.linspace(y+.05,y+h-.05,8))
        ax.scatter(xx,yy,s=2,color=muted,linewidths=0)
    title(6.96,"A","Choose recipes, culture cells, measure growth, then learn")
    steps=[("Choose recipes","All conditions selected\nbefore measurements"),
           ("Culture cells","96 wells per plate\n24 h in the simulator"),
           ("Measure growth","Biomass observations\ninclude measurement noise"),
           ("Update the model","Use results to choose\nthe next plate")]
    for i,(head,desc) in enumerate(steps):
        x=.18+i*1.85
        box(x,5.88,1.60,.78)
        text(x+.80,6.44,head,9,ha="center",weight="bold")
        text(x+.80,6.15,desc,8,ha="center",va="center",linespacing=1.5)
        if i<3: arrow((x+1.62,6.27),(x+1.82,6.27))
    ax.plot([6.53,6.53,.98,.98],[5.87,5.67,5.67,5.85],color=muted,lw=.8)
    arrow((.98,5.67),(.98,5.87))
    text(3.75,5.44,"Each new round waits for the preceding culture and measurement cycle",8,ha="center")
    text(.25,5.07,"Fixed budget",9,weight="bold")
    text(1.42,5.07,"1 initial plate  +  4 adaptive plates  =  5 culture rounds",9)
    text(1.42,4.85,"96 initial recipes; 80 recipes per adaptive plate, with selected replicates",8)
    ax.axhline(4.62,xmin=.025,xmax=.975,color="#CFD8DE",lw=.8)
    title(4.35,"B","An essential ingredient is not always the limiting ingredient")
    text(.25,4.10,"Essential: required for growth.   Limiting: constrains growth in the current recipe.",9)
    for x,vals,lim,recipe in [(.55,[.25,.75,.65],0,"Recipe 1"),(4.20,[.75,.25,.65],1,"Recipe 2")]:
        text(x+1.10,3.77,recipe,9,ha="center",weight="bold")
        for j,v in enumerate(vals):
            bx=x+.25+j*.72
            col="#AF4D25" if j==lim else "#AEBCC5"
            ax.add_patch(plt.Rectangle((bx,2.65),.43,v,facecolor=col,edgecolor="none"))
            text(bx+.215,2.48,"ABC"[j],9,ha="center",weight="bold" if j==lim else "normal")
            if j==lim:
                ax.annotate("limiting",(bx+.215,2.65+v),xytext=(bx+.215,3.55),
                            ha="center",fontsize=8,color=col,
                            arrowprops=dict(arrowstyle="->",color=col,lw=.8))
        ax.plot([x+.12,x+2.24],[2.65,2.65],color=muted,lw=.6)
    text(3.75,3.16,"Change the\nrecipe",8,ha="center",va="center")
    arrow((3.2,2.91),(4.02,2.91))
    text(.25,2.17,"A, B and C are all essential; the ingredient limiting growth can change between recipes.",8.5)
    text(.25,1.95,"Schematic: bar heights represent growth-support terms, not measured concentrations or results.",8)
    ax.axhline(1.77,xmin=.025,xmax=.975,color="#CFD8DE",lw=.8)
    title(1.50,"C","Higher best-discovered biomass within the same five-plate budget")
    for y,name,color in [(.98,"DS GP KB",COLORS["gp_bo_kb"]),(.42,"TabPFN KB",COLORS["tabpfn_kb"])]:
        text(.25,y+.14,name,9,weight="bold")
        for i in range(5): plate(1.35+i*.72,y)
    arrow((5.00,.91),(5.36,.91))
    box(5.47,.33,1.78,.94,"#EEF6FB")
    raw=pd.read_csv(DATA/"raw_sensitivity.csv")
    for y,d in [(1.01,80),(.64,120)]:
        v=raw[(raw.d==d)&(raw.reference=="gp_bo_kb")&(raw.metric=="relative biomass")]["mean"].iloc[0]
        text(5.60,y,f"{d} components: +{100*v:.1f}%",9,weight="bold")
    text(.25,.12,"Mean relative gain in best tested biomass; simulated outcomes, not laboratory validation (Fig 2).",8)
    save(fig,"Fig1")


def fig2():
    """Show all 20 scenario effects, using the manuscript's seed-first estimator."""
    raw = pd.read_csv(DATA / "raw_sensitivity.csv")
    emb = pd.read_csv(DATA / "embedding_contrasts.csv")
    runs=pd.concat([pd.read_csv(ROOT/f"results/v2a_runs_{p}.csv") for p in ("P1A","P3")])
    last=runs[runs["round"]==4]
    assert not last.duplicated(["n_sub","method","scenario","seed"]).any()
    assert last.groupby(["n_sub","method","scenario"]).seed.nunique().eq(5).all()
    means=last.groupby(["n_sub","scenario","method"]).best_true.mean().unstack("method")
    effects=[]
    fig, ax = plt.subplots(1,2,figsize=(WIDTH,3.8),gridspec_kw={"width_ratios":[1.6,1]})
    rng=np.random.default_rng(20261001)
    for ref,shift,marker in (("gp_bo_kb",-.18,"o"),("gp_qlognei",.18,"s")):
        q=raw[(raw.reference==ref)&(raw.metric=="relative biomass")].sort_values("d")
        for i,row in enumerate(q.itertuples()):
            m=means.loc[row.d]
            v=100*(m.tabpfn_kb/m[ref]-1)
            assert len(v)==20 and set(v.index)==set(range(20,40))
            assert np.isclose(v.mean(),100*row.mean,atol=1e-8)
            jitter=rng.uniform(-.085,.085,len(v))
            ax[0].scatter(i+shift+jitter,v,s=15,marker=marker,color=COLORS[ref],alpha=.42,
                          linewidths=0,zorder=2)
            effects.extend(dict(d=row.d,scenario=int(sc),reference=ref,relative_gain_pct=float(value))
                           for sc,value in v.items())
        ax[0].errorbar(np.arange(len(q))+shift,100*q["mean"],
                       yerr=100*np.array([q["mean"]-q.ci_lo,q.ci_hi-q["mean"]]),
                       fmt=marker,ms=5,mec="white",mew=.6,capsize=3,elinewidth=1.4,
                       label="vs "+LABELS[ref],color=COLORS[ref],zorder=4)
    pd.DataFrame(effects).to_csv(OUT/"Fig2_scenario_effects.csv",index=False)
    ax[0].axhline(0,color="#555555",lw=.7,zorder=1)
    ax[0].set_xticks(np.arange(5),[8,20,40,80,120])
    ax[0].set_xlim(-.6,4.6)
    ax[0].set_xlabel("Number of components")
    ax[0].set_ylabel("Gain of TabPFN KB in best biomass (%)")
    ax[0].legend(frameon=False,loc="upper left")
    panel(ax[0],"A")
    ax[0].set_title("Individual growth scenarios",loc="center",fontsize=9)
    e=emb[emb.d==120].set_index("embedding").loc[["ambient","rotate","intrinsic"]]
    ax[1].errorbar(range(3),e["mean"],yerr=np.array([e["mean"]-e.ci_lo,e.ci_hi-e["mean"]]),
                   fmt="o",ms=5,color=COLORS["tabpfn_kb"],capsize=4)
    ax[1].axhline(0,color="#555555",lw=.7)
    ax[1].set_xticks(range(3),["8 comp.,\npadded","8 comp.,\nmixed","120 comp."])
    ax[1].set_xlim(-.5,2.5)
    ax[1].set_xlabel("Growth model (120 search variables)",fontsize=8)
    ax[1].set_ylabel("TabPFN KB − DS GP KB\n(normalized score)")
    panel(ax[1],"B")
    ax[1].set_title("Embedding controls",loc="center",fontsize=9)
    fig.text(.08,.04,"A: small points = 20 scenarios per comparison; large markers = mean; bars = 95% bootstrap CI.",fontsize=8)
    fig.tight_layout(rect=(0,.09,1,1),w_pad=1.4)
    save(fig,"Fig2")


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
