"""Shared figure style, colours and saving (PDF + PNG + data CSV)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]

CLASS_COLOR = {"fire": "#D55E00", "smoke": "#56B4E9", "normal": "#009E73"}
STEP_COLOR = {"median (P0→P1)": "#0072B2", "CLAHE (P1→P2)": "#E69F00", "full (P0→P2)": "#CC79A7"}
METHOD_COLOR = {"median": "#0072B2", "gaussian": "#E69F00", "bilateral": "#009E73",
                "he_rgb": "#D55E00", "he_l": "#CC79A7", "clahe_l": "#0072B2"}
MODEL_STYLE = {
    "A_svm_rf": {"color": "#999999", "marker": "s"}, "B_mobilenetv3": {"color": "#E69F00", "marker": "^"},
    "C_mobilevit": {"color": "#56B4E9", "marker": "o"}, "D_mobilevit_concat": {"color": "#009E73", "marker": "D"},
    "E_mobilevit_diptokens": {"color": "#D55E00", "marker": "*"},
}

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "savefig.dpi": 300,
})


def save(fig, out_base, data=None) -> None:
    """Save PDF + PNG, and the exact plotted numbers as CSV next to them (guide §15)."""
    fig.savefig(f"{out_base}.pdf")
    fig.savefig(f"{out_base}.png")
    if data is not None:
        data.to_csv(f"{out_base}_data.csv", index=False)
