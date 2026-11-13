"""Salva grafici PNG e riepiloghi numerici dell’esperimento stilometrico."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save_style_outputs(df, effects, metrics, shortlist, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pooled = effects[effects.scope == 'pooled'].copy()
    pooled['abs_g'] = pooled.hedges_g.abs()
    top = pooled.sort_values('abs_g', ascending=False).head(18)
    names = list(top.feature)
    domain_effects = effects[(effects.scope == 'domain') & effects.feature.isin(names)]
    pivot = domain_effects.pivot(index='feature', columns='domain', values='hedges_g').reindex(names)
    if len(pivot):
        fig, (ax, heat) = plt.subplots(1, 2, figsize=(12, max(5, .30 * len(names))), gridspec_kw={'width_ratios': [1.3, 1]})
        y = np.arange(len(names))
        ax.scatter(top.hedges_g, y, s=22, color='#167c80')
        for i, row in enumerate(top.itertuples()):
            if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
                ax.plot([row.ci_low, row.ci_high], [i, i], color='#167c80', lw=1)
        ax.set_yticks(y, names, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color='gray', lw=.7)
        ax.set_xlabel('Hedges g (macchina − umano), IC 95%')
        ax.set_title('Effetti aggregati: prime 18 feature')
        limit = max(1., float(np.nanmax(np.abs(pivot.to_numpy()))))
        im = heat.imshow(pivot, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
        heat.set_yticks(y, names, fontsize=8)
        heat.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=20)
        heat.set_title('Lo stesso effetto nei tre domini')
        for i in range(len(pivot)):
            for j in range(len(pivot.columns)):
                value = pivot.iloc[i, j]
                heat.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=8,
                          color='white' if abs(value) > .6 * limit else 'black')
        fig.colorbar(im, ax=heat, shrink=.7, label='g')
        fig.tight_layout()
        fig.savefig(outdir / 'feature_effects.png', dpi=180)
        plt.close(fig)
    test = metrics[metrics.protocol == 'leave_domain_out']
    if len(test):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        domains = sorted(test.domain.unique())
        sets = list(dict.fromkeys(test.feature_set))
        width = .8 / len(sets)
        for k, feature_set in enumerate(sets):
            rows = test[test.feature_set == feature_set].set_index('domain').reindex(domains)
            x = np.arange(len(domains)) + (k - (len(sets) - 1) / 2) * width
            lower = np.maximum(rows.roc_auc - rows.roc_auc_ci_low, 0)
            upper = np.maximum(rows.roc_auc_ci_high - rows.roc_auc, 0)
            ax.bar(x, rows.roc_auc, width, label=feature_set)
            ax.errorbar(x, rows.roc_auc, yerr=[lower, upper], fmt='none', ecolor='black', lw=.8, capsize=2)
        ax.axhline(.5, ls='--', color='gray', lw=1)
        ax.set_xticks(range(len(domains)), domains)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel('ROC-AUC (classe positiva: macchina)')
        ax.set_title('Addestramento sugli altri due domini; test sul dominio indicato')
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / 'cross_domain_auc.png', dpi=180)
        plt.close(fig)

    metrics_cols = ['protocol', 'domain', 'generator', 'feature_set', 'n_test', 'roc_auc', 'roc_auc_ci_low', 'roc_auc_ci_high', 'balanced_accuracy', 'macro_f1']
    summary = {'n_texts': len(df), 'shortlist': shortlist,
               'cross_domain': test[metrics_cols].to_dict(orient='records')}
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
