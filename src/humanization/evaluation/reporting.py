"""Grafici autonomi e report """
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def markdown_table(frame, digits=3):
    
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            if np.isfinite(x) and x != 0 and abs(x) < 10 ** (-digits):
                return f'{x:.2e}'
            return f'{x:.{digits}f}' if np.isfinite(x) else 'NA'
        return str(x).replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
             '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    lines.extend('| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None))
    return '\n'.join(lines)


def write_report(df, effects, metrics, shortlist, robustness, manifest, outdir):
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
        fig.savefig(outdir / 'feature_effects.pdf')
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
        fig.savefig(outdir / 'cross_domain_auc.pdf')
        plt.close(fig)

    counts = df.groupby(['domain', 'generator', 'label']).size().rename('n').reset_index()
    selected = robustness[robustness.selected]
    metrics_cols = ['protocol', 'domain', 'generator', 'feature_set', 'n_test', 'roc_auc', 'roc_auc_ci_low', 'roc_auc_ci_high', 'balanced_accuracy', 'macro_f1']
    effect_cols = ['feature', 'human_mean', 'machine_mean', 'hedges_g', 'ci_low', 'ci_high', 'bh_q']
    lines = [
        '# Esperimento esplorativo: componente stilometrica',
        '', '## Dati e provenienza', '',
        f'Testi analizzati: {len(df)}; coppie di origine: {df.pair_id.nunique()}. '
        'Etichetta 0 = umano, 1 = macchina. Ogni umano compare una volta nel campione principale.',
        '', 'La provenienza completa, gli hash SHA-256 e i conteggi di esclusione sono in `data_manifest.json`. '
        'Il campione usa la baseline MGTBench, senza attacchi o riscritture. '
        'Non è una riproduzione dei file di attacco chiamati `clean` in TH-Bench.',
        '', '**Fonte effettiva:** ' + str(manifest.get('download_provenance', {}).get('mirror_repository', manifest.get('official_source', 'vedere manifest'))) +
        '. L’identità dei byte con la distribuzione ufficiale non è stata verificata quando si usa il mirror.',
        '', '**Pulizia applicata:** ' + '; '.join(manifest.get('cleaning', [])) + '.',
        '', markdown_table(counts),
        '', '## Protocollo', '',
        'Feature fissate prima di osservare i risultati: spaCy en_core_web_sm 3.8.0, textstat en_US e MATTR con finestra 50. '
        'Le definizioni e i denominatori sono in `feature_schema.json`. Non si usano troncamenti a un numero fisso di parole; '
        'il manifest documenta l’eventuale rimozione simmetrica delle code senza terminazione di frase plausibile. '
        'TTR, hapax e conteggi globali sono diagnostici esclusi dal modello principale di stile.',
        '', 'Effetti: Hedges g corretto per piccoli campioni, con segno macchina meno umano; '
        'g positivo indica valori maggiori nei testi macchina. IC 95% bootstrap a gruppi di origine; '
        'p-value paired t sulle differenze entro coppia e correzione BH separata entro ogni tabella. '
        'Per generatore si confrontano soltanto gli umani delle corrispondenti coppie. '
        'Le soglie di grandezza dell’effetto sono euristiche, non confini naturali.',
        '', 'Classificatore: regressione logistica L2, C=1 fisso, class_weight=balanced, soglia 0.5; '
        'imputazione mediana e standardizzazione stimate esclusivamente sul training. '
        'CV a cinque fold con gruppi di origine disgiunti entro dominio; leave-one-domain-out; '
        'leave-one-generator-out con ulteriore esclusione dal training dei gruppi sorgente nel test. '
        'Nessun tuning sul test. Le metriche usano predizioni out-of-fold aggregate. '
        'Gli IC delle metriche ricampionano le predizioni fisse per gruppo: non includono tutta l’incertezza dell’addestramento.',
        '', '## Feature con i maggiori effetti aggregati', '', markdown_table(top[effect_cols]),
        '', 'Gli effetti aggregati non dimostrano robustezza: consultare `feature_effects.csv` '
        'per tutti i domini, generatori e incroci dominio×generatore.',
        '', '![Effetti](feature_effects.png)',
        '', '## Classificazione e trasferimento', '', markdown_table(metrics[metrics_cols]),
        '', '![Trasferimento tra domini](cross_domain_auc.png)',
        '', '## Lista corta esplorativa', '',
        'Regola: stessa direzione in tutti e tre i domini, |g| ≥ 0.3 in ciascuno, '
        'segno coerente in almeno 5/7 dei generatori; massimo 12 feature; '
        'rimozione di ridondanze con |correlazione di Spearman| ≥ 0.85. '
        'La lista è ricavata dall’intero campione ed è una proposta per esperimenti successivi: '
        'non è stata retroattivamente valutata come se fosse una scelta indipendente dal test.',
        '', markdown_table(selected[['feature', 'pooled_g', 'min_domain_abs_g', 'generators_passing', 'n_generators']]) if len(selected) else
        'Nessuna feature supera tutti i criteri. Non si forza una lista arbitraria; consultare `feature_robustness.csv`.',
        '', '## Cosa si può concludere sulla guida Wikipedia', '',
        'Il modello `wikipedia` usa soltanto i proxy fissati nel codice: vocabolario AI, enfasi, '
        'conclusioni, formule conversazionali, parallelismi negativi, Markdown ed em dash. '
        'Non misura l’intera guida, né verifica superficialità, neutralità, correttezza delle fonti o intenzione dell’autore. '
        'Confrontare `wikipedia` con `style` e `combined` su dati non visti; '
        'un vantaggio solo nel dominio di addestramento non dimostra trasferibilità. '
        'Gli indicatori possono essere rari, dipendere dall’epoca dei generatori e cambiare segno.',
        '', '## Limiti', '',
        '- Circa 42–43 testi macchina per generatore e dominio: le stime fini hanno incertezza elevata.',
        '- Le feature normalizzate possono ancora dipendere dalla lunghezza; `length_only` è un controllo, non elimina ogni confondente.',
        '- I generatori MGTBench sono storici: risultati non generalizzabili automaticamente agli LLM attuali o a testi umanizzati.',
        '- La lista corta richiede conferma indipendente, idealmente su un nuovo campione e nuove riscritture.',
        '- La pulizia delle terminazioni è euristica: non certifica la completezza grammaticale né identifica tutte le generazioni fallite. Rifiuti e formule di chatbot possono restare.',
        '- Il mirror pubblico, se utilizzato, non è stato confrontato byte per byte con l’archivio originale Google Drive; vedere manifest.',
        '', 'File completi: `feature_effects.csv`, `feature_robustness.csv`, `classifier_metrics.csv`, '
        '`predictions.csv`, `run_metadata.json`, `feature_schema.json` e `data_manifest.json`.',
    ]
    (outdir / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    summary = {'n_texts': len(df), 'shortlist': shortlist,
               'cross_domain': test[metrics_cols].to_dict(orient='records')}
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
