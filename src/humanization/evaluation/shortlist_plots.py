"""Grafici PNG dalle tabelle salvate delle analisi sulle feature compatte."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from humanization.evaluation.shortlist_statistics import HISTORICAL_FEATURES


LABELS = {
    "C1_full105": "C1: tutte le 105",
    "C2_fixed4": "C2: quattro storiche*",
    "C3_training4": "C3: quattro dal training",
    "C4_random": "C4: casuali uniformi",
    "C4_random_matched": "C4: casuali, stesse famiglie",
    "C1_full100": "Riferimento: 100",
    "C1_style93": "Riferimento: 93",
    "word_count_only": "Solo numero di parole",
}
MAIN = tuple(LABELS)[:5]
PROTOCOLS = {"cv": "Validazione ripetuta", "leave_domain_out": "Dominio escluso",
             "leave_generator_out": "Generatore escluso"}


def plot_shortlist(results: Path | str, protocol_path: Path | str | None = None) -> list[Path]:
    """Salva confronto delle AUC e frequenze di selezione come immagini PNG."""
    results = Path(results)
    metrics = pd.read_csv(results / "metrics.csv")
    frequencies = pd.read_csv(results / "selection_frequency.csv")
    protocol = json.loads((Path(protocol_path) if protocol_path else results / "protocol.json").read_text())

    def metric(name, config):
        return metrics[(metrics.protocol == name) & (metrics.scope == "macro") &
                       (metrics.configuration == config)].iloc[0]

    colors = ["#22577a", "#b6483e", "#21816b", "#777777", "#7953a3"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharey=True)
    y = np.arange(len(MAIN))
    for ax, (name, title) in zip(axes, PROTOCOLS.items()):
        for i, (config, color) in enumerate(zip(MAIN, colors)):
            row = metric(name, config)
            ax.errorbar(row.auc, i, xerr=[[row.auc-row.ci_low], [row.ci_high-row.auc]],
                        fmt="o", color=color, capsize=3, markersize=5)
            ax.text(row.ci_high + .009, i+.1, f"{row.auc:.3f}", fontsize=8)
        ax.axvline(.5, ls="--", color="#aaaaaa", lw=1)
        ax.set_xlim(.48, .98)
        ax.set_ylim(len(MAIN)-.5, -.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("AUC macro e IC 95% per gruppo", fontsize=9)
        ax.grid(axis="x", alpha=.15)
    axes[0].set_yticks(y, [LABELS[c] for c in MAIN], fontsize=9)
    fig.tight_layout()
    auc_path = results / "auc_comparison.png"
    fig.savefig(auc_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    observed = frequencies[frequencies.protocol == "cv"].copy()
    names = observed.sort_values(["selection_count", "feature"], ascending=[False, True]).head(8).feature.tolist()
    for name in protocol["fixed_features"]:
        if name not in names:
            names.append(name)
    rates = observed.set_index("feature").loc[names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(np.arange(len(names)), rates.selection_rate,
            color=["#b6483e" if flag else "#22577a" for flag in rates.historical_shortlist])
    ax.set_yticks(np.arange(len(names)), names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    for i, row in enumerate(rates.itertuples()):
        ax.text(row.selection_rate+.015, i+.1, f"{row.selection_count}/{row.n_fits}", fontsize=9)
    ax.set_xlabel("Frazione dei training CV in cui C3 seleziona la feature")
    ax.set_title("Stabilità descrittiva: rosso = shortlist storica", fontsize=11)
    fig.tight_layout()
    stability_path = results / "selection_stability.png"
    fig.savefig(stability_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [auc_path, stability_path]


def plot_robustness(results: Path | str) -> list[Path]:
    """Salva il confronto fra le selezioni da 105 e da 100 candidati."""
    results = Path(results)
    metrics = pd.read_csv(results / "metrics.csv")
    comparison = pd.read_csv(results / "selection_comparison.csv")
    configurations = ("C3_105", "C3_100")

    def metric(name, config):
        return metrics[(metrics.protocol == name) & (metrics.scope == "macro") &
                       (metrics.configuration == config)].iloc[0]

    count_names = comparison[(comparison.total_C3_100_count > 0) | (comparison.total_C3_105_count > 0)].copy()
    count_names = count_names.sort_values(["total_C3_100_count", "total_C3_105_count", "feature"], ascending=[False, False, True])
    count_names = count_names.feature.tolist()
    for feature in HISTORICAL_FEATURES:
        if feature not in count_names:
            count_names.append(feature)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharey=True)
    for ax, (name, title) in zip(axes, PROTOCOLS.items()):
        for i, config in enumerate(configurations):
            row = metric(name, config)
            ax.errorbar(row.auc, i, xerr=[[row.auc-row.ci_low], [row.ci_high-row.auc]],
                        fmt="o", color=("#22577a", "#21816b")[i], capsize=3)
            ax.text(row.ci_high+.006, i+.1, f"{row.auc:.3f}", fontsize=9)
        ax.axvline(.5, color="gray", ls="--", lw=.8)
        ax.set_xlim(.48, .93)
        ax.set_ylim(1.6, -.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("AUC macro e IC 95% per gruppo", fontsize=9)
        ax.grid(axis="x", alpha=.15)
    axes[0].set_yticks([0, 1], ["C3: 105 candidati", "C3: 100 candidati"], fontsize=9)
    fig.tight_layout()
    auc_path = results / "auc_comparison.png"
    fig.savefig(auc_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    plot_names = count_names[:12]
    for feature in HISTORICAL_FEATURES:
        if feature not in plot_names:
            plot_names.append(feature)
    indexed = comparison.set_index("feature").loc[plot_names]
    fig, ax = plt.subplots(figsize=(8.2, max(3.8, .30*len(plot_names))))
    y = np.arange(len(plot_names))
    for offset, config, color in ((-.18,"C3_105","#22577a"),(.18,"C3_100","#21816b")):
        rate = indexed[f"cv_{config}_count"] / indexed.cv_n_fits
        ax.barh(y+offset, rate, height=.33, color=color, label=config.replace("_", ": ")+" candidati")
    ax.set_yticks(y, plot_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Frequenza di selezione nei 25 training CV")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    selection_path = results / "selection_comparison.png"
    fig.savefig(selection_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [auc_path, selection_path]
