"""One visual style for every figure of the paper.

Font: Arial, embedded (pdf.fonttype 42). Text 7.5 to 8 pt at print size, panel letters 10 pt bold.
Width: the text block of the paper is 122 mm (4.8 in); every main figure is drawn at that width.

Colours (one meaning each, Nature Publishing Group palette):
  OURS     dark blue   our methods (FLiPAlign, split labelling)
  BASE     light blue  published tools and one-to-one labelling
  RIGHT    green       a right answer
  WRONG    red         a wrong answer
  REFUSED  grey        a refused answer, background, "other"
  ACCENT   orange      truth and highlights (true placement, landmarks)
Region labels on tissue maps use REGION (12 colours); they carry no meaning across sections unless stated.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

OURS, BASE, RIGHT, WRONG, REFUSED, ACCENT = "#3C5488", "#4DBBD5", "#00A087", "#E64B35", "#B0B0B0", "#F39B7F"
INK, MUTED, LIGHT = "#222222", "#6B6B6B", "#E6E6E6"
REGION = ["#4DBBD5", "#E64B35", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC9E3C",
          "#7E6148", "#B09C85", "#C4A5DE", "#6FB07F"]
TEXTW = 4.8          # inches
FS, FS_SMALL, FS_LETTER = 8, 7.5, 10

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"], "font.size": FS,
    "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
    "legend.fontsize": FS_SMALL, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlepad": 4, "axes.labelpad": 2,
    "legend.frameon": False, "legend.handlelength": 1.2, "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2,
    "lines.linewidth": 1.2, "lines.markersize": 4, "axes.unicode_minus": True, "savefig.dpi": 600,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
})
region_cmap = ListedColormap(REGION)


def letter(fig, ax, s, dx=-0.02, dy=0.0):
    """bold panel letter at the top left of an axes, in figure coordinates"""
    bb = ax.get_position()
    fig.text(bb.x0 + dx, bb.y1 + dy, s, fontsize=FS_LETTER, fontweight="bold", va="bottom", ha="right")


def tissue(ax, xy, lab, cmap=None, s=0.6, alpha=1.0, colors=None):
    """scatter of cells or spots coloured by label, equal aspect, no axes"""
    import numpy as np
    if colors is None:
        cols = np.array(REGION)[np.asarray(lab) % len(REGION)]
    else:
        cols = colors
    ax.scatter(xy[:, 0], xy[:, 1], c=cols, s=s, lw=0, alpha=alpha, rasterized=True)
    ax.set_aspect("equal"); ax.axis("off")


def save(fig, path):
    fig.savefig(path + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path + ".png", bbox_inches="tight", pad_inches=0.02, dpi=300)
