"""Evaluation figures: confusion matrix, look-alike false alarms."""
import numpy as np
import pandas as pd

from src.viz import style
import matplotlib.pyplot as plt


def confusion_figure(cm: list, labels: list, title: str, out_base) -> None:
    cm = np.array(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(labels)), labels), ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted"), ax.set_ylabel("true"), ax.set_title(title, fontsize=8)
    ax.spines[:].set_visible(True)
    style.save(fig, out_base, pd.DataFrame(cm, index=labels, columns=labels).reset_index(names="true"))
    plt.close(fig)


def lookalike_figure(table: pd.DataFrame, out_base) -> None:
    """table: columns model, lookalike_type, false_alarm_rate."""
    types = sorted(table["lookalike_type"].unique())
    models = list(dict.fromkeys(table["model"]))
    fig, ax = plt.subplots(figsize=(6.2, 2.3))
    w = 0.8 / max(len(models), 1)
    for j, m in enumerate(models):
        d = table[table["model"] == m].set_index("lookalike_type").reindex(types)
        col = style.MODEL_STYLE.get(m, {}).get("color", style.OKABE_ITO[j % 8])
        x = np.arange(len(types)) + (j - (len(models) - 1) / 2) * w
        ax.bar(x, d["false_alarm_rate"], w, label=m, color=col)
        for xi, v in zip(x, d["false_alarm_rate"]):
            if pd.notna(v):
                ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=5.5)
    ax.set_xticks(range(len(types)), [t.replace("_", " ") for t in types], fontsize=7)
    ax.set_ylabel("false-alarm rate"), ax.set_ylim(0, 1), ax.legend(fontsize=7, frameon=False)
    style.save(fig, out_base, table)
    plt.close(fig)
